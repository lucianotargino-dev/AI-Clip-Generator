"""Recorte e processamento dos trechos selecionados para geração de clipes.

O módulo recebe o vídeo de origem e os momentos identificados pelo módulo
de destaques e gera os respectivos arquivos de vídeo, reenquadrados numa
grade dinâmica de 1 a 4 sub-câmeras, cada uma acompanhando um rosto
detectado (YOLO), com corte seco em mudanças de cena ou entrada/saída de
pessoas, movimento suave dentro da mesma cena, zoom proporcional ao
tamanho do rosto e um período de confirmação antes de incluir um rosto
novo na grade (evita que falsos positivos passageiros entrem e saiam).
"""

import os
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .configuracao import DIRETORIO_SAIDA

# --------------------------------------------------------------------------
# Detecção de rostos
# --------------------------------------------------------------------------
YOLO_MODEL = "yolov12s-face.pt"
DEVICE_BACKEND = "cpu"      # "cpu" | "cuda" (GPU NVIDIA)
DETECTION_INTERVAL = 2      # a cada quantos quadros o YOLO roda de fato (>= 1)
CONF_MINIMA = 0.4

# --------------------------------------------------------------------------
# Resolução de saída (fixa, para as células da grade dividirem certinho;
# a largura é derivada da proporção pedida, ex.: 9:16)
# --------------------------------------------------------------------------
ALTURA_SAIDA_PADRAO = 1920

# --------------------------------------------------------------------------
# Rastreamento: associação de detecções aos rostos já rastreados
# --------------------------------------------------------------------------
DISTANCIA_MAXIMA_CORRESPONDENCIA = 260.0  # px (na resolução de origem)

# --------------------------------------------------------------------------
# Debounce de entrada/saída de rostos (evita flicker por falsos positivos
# e por falhas pontuais de detecção)
# --------------------------------------------------------------------------
SEGUNDOS_CONFIRMACAO = 1.5          # tempo detectando um rosto novo antes de incluí-lo na grade
SEGUNDOS_REMOCAO = 0.5              # tolerância antes de remover um rosto confirmado que sumiu
SEGUNDOS_DESCARTE_CANDIDATO = 0.35  # descarta um candidato que sumiu antes de ser confirmado

# --------------------------------------------------------------------------
# Suavização de câmera (posição e zoom) dentro da mesma cena
# --------------------------------------------------------------------------
FATOR_SUAVIZACAO = 0.045
VELOCIDADE_MAXIMA_POSICAO = 9.0     # px/quadro
BANDA_MORTA_POSICAO = 22.0          # px

FATOR_SUAVIZACAO_ZOOM = 0.035
VELOCIDADE_MAXIMA_ZOOM = 6.0        # px/quadro, na altura do recorte
BANDA_MORTA_TAMANHO = 18.0          # px

# --------------------------------------------------------------------------
# Zoom: altura do recorte-fonte é proporcional à altura do rosto detectado
# --------------------------------------------------------------------------
FATOR_ZOOM_ROSTO = 3.4
CROP_ALTURA_MINIMA_PX = 90.0

# --------------------------------------------------------------------------
# Detecção de corte de cena (mudança abrupta no conteúdo do quadro)
# --------------------------------------------------------------------------
TAMANHO_AMOSTRA_CENA = (64, 36)
LIMIAR_CORTE_CENA = 32.0


# ============================================================================
# Carregamento do modelo YOLO (cache a nível de módulo — carrega uma única
# vez por processo, reaproveitado entre todos os clipes de um mesmo lote)
# ============================================================================
_modelo_cache: Optional[YOLO] = None
_kwargs_inferencia_cache: Dict = {}


def _carregar_cpu():
    try:
        modelo = YOLO(YOLO_MODEL)
        modelo.to("cpu")
        print(f"[Reenquadramento] Modelo YOLO carregado: {YOLO_MODEL} — dispositivo: CPU", flush=True)
        return modelo, {}
    except Exception as exc:
        raise RuntimeError(f"Erro crítico: Impossível carregar o modelo YOLO nem mesmo na CPU: {exc}")


def _carregar_cuda():
    if not torch.cuda.is_available():
        print("[Reenquadramento] Aviso: CUDA indisponível — YOLO usando CPU", flush=True)
        return _carregar_cpu()
    
    try:
        modelo = YOLO(YOLO_MODEL)
        modelo.to("cuda")
        print(f"[Reenquadramento] Modelo YOLO carregado: {YOLO_MODEL} — dispositivo: GPU NVIDIA (CUDA)", flush=True)
        return modelo, {}
    except Exception as exc:
        print(f"[Reenquadramento] Aviso: Falha ao carregar na CUDA ({exc}) — YOLO usando CPU", flush=True)
        return _carregar_cpu()


def _obter_modelo() -> Tuple[YOLO, Dict]:
    """Carrega o modelo YOLO obrigatoriamente. Se CUDA falhar, usa CPU. Se CPU falhar, interrompe."""
    global _modelo_cache, _kwargs_inferencia_cache
    if _modelo_cache is not None:
        return _modelo_cache, _kwargs_inferencia_cache

    if DEVICE_BACKEND == "cuda":
        _modelo_cache, _kwargs_inferencia_cache = _carregar_cuda()
    else:
        _modelo_cache, _kwargs_inferencia_cache = _carregar_cpu()

    return _modelo_cache, _kwargs_inferencia_cache


def _detectar_rostos(modelo, quadro, kwargs_inferencia) -> List[Tuple[float, float, float, float, float]]:
    """Roda o YOLO no quadro e retorna [(x1, y1, x2, y2, conf), ...]."""
    try:
        resultados = modelo(quadro, conf=CONF_MINIMA, verbose=False, **kwargs_inferencia)
    except Exception:
        return []
    caixas = []
    for caixa in resultados[0].boxes:
        x1, y1, x2, y2 = caixa.xyxy[0].tolist()
        conf = float(caixa.conf[0])
        caixas.append((x1, y1, x2, y2, conf))
    return caixas


# ============================================================================
# Estado de rastreamento de um rosto entre quadros
# ============================================================================
@dataclass
class _SlotRosto:
    id: int
    confirmado: bool = False
    centro_x: float = 0.0
    centro_y: float = 0.0
    alvo_x: float = 0.0
    alvo_y: float = 0.0
    tamanho: float = 0.0        # altura suavizada do recorte-fonte (zoom)
    tamanho_alvo: float = 0.0
    frames_sem_deteccao: int = 0
    frames_detectado: int = 0   # contagem consecutiva enquanto candidato (debounce de confirmação)


def _associar_deteccoes(
    slots: Dict[int, _SlotRosto],
    deteccoes: List[Tuple[float, float, float, float, float]],
    distancia_maxima: float,
):
    """Casa cada detecção com o slot rastreado mais próximo (guloso, por
    distância). Retorna (correspondencias, novas_deteccoes, slots_sem_correspondencia)."""
    pares = []
    for id_slot, slot in slots.items():
        for i, (x1, y1, x2, y2, _conf) in enumerate(deteccoes):
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dist = ((slot.alvo_x - cx) ** 2 + (slot.alvo_y - cy) ** 2) ** 0.5
            if dist <= distancia_maxima:
                pares.append((dist, id_slot, i))
    pares.sort(key=lambda p: p[0])

    correspondencias: Dict[int, Tuple[float, float, float, float, float]] = {}
    usados_slot, usados_deteccao = set(), set()
    for _dist, id_slot, i in pares:
        if id_slot in usados_slot or i in usados_deteccao:
            continue
        correspondencias[id_slot] = deteccoes[i]
        usados_slot.add(id_slot)
        usados_deteccao.add(i)

    novas = [d for i, d in enumerate(deteccoes) if i not in usados_deteccao]
    sem_correspondencia = [id_slot for id_slot in slots if id_slot not in usados_slot]
    return correspondencias, novas, sem_correspondencia


def _suavizar(atual: float, alvo: float, fator: float, velocidade_maxima: float) -> float:
    passo = (alvo - atual) * fator
    passo = max(-velocidade_maxima, min(velocidade_maxima, passo))
    return atual + passo


def _calcular_tamanho_alvo(altura_rosto: float) -> float:
    return max(CROP_ALTURA_MINIMA_PX, altura_rosto * FATOR_ZOOM_ROSTO)


# ============================================================================
# Layout da grade (1 a 4 células) e composição de cada célula
# ============================================================================
def _calcular_grade(ids_ativos: List[int], largura_saida: int, altura_saida: int) -> Dict[int, Tuple[int, int, int, int]]:
    """Retorna {id_slot: (x, y, largura, altura)} para cada célula da grade."""
    n = len(ids_ativos)
    celulas: Dict[int, Tuple[int, int, int, int]] = {}
    meia_altura = altura_saida // 2
    meia_largura = largura_saida // 2

    if n == 1:
        celulas[ids_ativos[0]] = (0, 0, largura_saida, altura_saida)
    elif n == 2:
        celulas[ids_ativos[0]] = (0, 0, largura_saida, meia_altura)
        celulas[ids_ativos[1]] = (0, meia_altura, largura_saida, altura_saida - meia_altura)
    elif n == 3:
        # "T": 2 lado a lado em cima, 1 ocupando a base inteira
        celulas[ids_ativos[0]] = (0, 0, meia_largura, meia_altura)
        celulas[ids_ativos[1]] = (meia_largura, 0, largura_saida - meia_largura, meia_altura)
        celulas[ids_ativos[2]] = (0, meia_altura, largura_saida, altura_saida - meia_altura)
    elif n >= 4:
        celulas[ids_ativos[0]] = (0, 0, meia_largura, meia_altura)
        celulas[ids_ativos[1]] = (meia_largura, 0, largura_saida - meia_largura, meia_altura)
        celulas[ids_ativos[2]] = (0, meia_altura, meia_largura, altura_saida - meia_altura)
        celulas[ids_ativos[3]] = (meia_largura, meia_altura, largura_saida - meia_largura, altura_saida - meia_altura)

    return celulas


def _recorte_central(quadro_fonte: np.ndarray, largura_saida: int, altura_saida: int) -> np.ndarray:
    """Recorte central simples, usado quando não há nenhum rosto confirmado."""
    altura_fonte, largura_fonte = quadro_fonte.shape[:2]
    proporcao_alvo = largura_saida / altura_saida
    if proporcao_alvo < largura_fonte / altura_fonte:
        crop_altura = altura_fonte
        crop_largura = int(crop_altura * proporcao_alvo)
    else:
        crop_largura = largura_fonte
        crop_altura = int(crop_largura / proporcao_alvo)
    x0 = (largura_fonte - crop_largura) // 2
    y0 = (altura_fonte - crop_altura) // 2
    recorte = quadro_fonte[y0:y0 + crop_altura, x0:x0 + crop_largura]
    return cv2.resize(recorte, (largura_saida, altura_saida), interpolation=cv2.INTER_LINEAR)


def _recortar_e_redimensionar(
    quadro_fonte: np.ndarray, centro_x: float, centro_y: float, tamanho_altura: float,
    celula_largura: int, celula_altura: int,
) -> np.ndarray:
    """Recorta a região (centro, tamanho) do quadro fonte, reposicionando
    para caber nos limites da imagem, e redimensiona para a célula alvo."""
    altura_fonte, largura_fonte = quadro_fonte.shape[:2]
    proporcao_celula = celula_largura / celula_altura

    crop_altura = min(tamanho_altura, altura_fonte)
    crop_largura = crop_altura * proporcao_celula
    if crop_largura > largura_fonte:
        crop_largura = largura_fonte
        crop_altura = crop_largura / proporcao_celula

    x0 = int(round(centro_x - crop_largura / 2))
    y0 = int(round(centro_y - crop_altura / 2))
    x1 = x0 + int(round(crop_largura))
    y1 = y0 + int(round(crop_altura))

    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > largura_fonte:
        x0 -= (x1 - largura_fonte)
        x1 = largura_fonte
    if y1 > altura_fonte:
        y0 -= (y1 - altura_fonte)
        y1 = altura_fonte
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(largura_fonte, x1), min(altura_fonte, y1)

    recorte = quadro_fonte[y0:y1, x0:x1]
    if recorte.size == 0 or recorte.shape[0] < 2 or recorte.shape[1] < 2:
        recorte = quadro_fonte

    return cv2.resize(recorte, (celula_largura, celula_altura), interpolation=cv2.INTER_LINEAR)


# ============================================================================
# Rastreador de grade dinâmica — mantém o estado entre quadros de um clipe
# ============================================================================
class _RastreadorGradeDinamica:
    """Detecta rostos, mantém identidade entre quadros, aplica o debounce de
    entrada/saída, suaviza posição e zoom dentro da mesma cena, corta seco em
    mudanças de cena ou de layout, e monta a composição final de cada quadro."""

    def __init__(self, fps: float, largura_saida: int, altura_saida: int):
        self.fps = max(1.0, fps)
        self.largura_saida = largura_saida
        self.altura_saida = altura_saida

        self._proximo_id = 1
        self._slots: Dict[int, _SlotRosto] = {}
        self._ordem_confirmados: List[int] = []       # ordem estável (evita reembaralhar células)
        self._selecionados_acima_de_4: List[int] = []  # subconjunto fixo quando há > 4 confirmados

        self._quadro_anterior_cena: Optional[np.ndarray] = None
        self._contador_quadros = 0

        self.model, self.infer_kwargs = _obter_modelo()

        self._frames_confirmacao = max(1, round(SEGUNDOS_CONFIRMACAO * self.fps))
        self._frames_remocao = max(1, round(SEGUNDOS_REMOCAO * self.fps))
        self._frames_descarte_candidato = max(1, round(SEGUNDOS_DESCARTE_CANDIDATO * self.fps))

    def processar_quadro(self, quadro: np.ndarray) -> np.ndarray:
        self._contador_quadros += 1

        corte_de_cena = self._checar_corte_de_cena(quadro)
        if corte_de_cena:
            self._slots.clear()
            self._ordem_confirmados.clear()
            self._selecionados_acima_de_4.clear()

        deve_detectar = self.model is not None and (self._contador_quadros % max(1, DETECTION_INTERVAL) == 0)
        if deve_detectar:
            deteccoes = _detectar_rostos(self.model, quadro, self.infer_kwargs)
            self._atualizar_rastreamento(deteccoes)

        mudou_layout = self._promover_e_remover()
        self._avancar_suavizacao(forcar_snap=corte_de_cena or mudou_layout)

        return self._montar_saida(quadro)

    # ------------------------------------------------------------------
    def _checar_corte_de_cena(self, quadro: np.ndarray) -> bool:
        pequeno = cv2.resize(quadro, TAMANHO_AMOSTRA_CENA, interpolation=cv2.INTER_AREA)
        pequeno_cinza = cv2.cvtColor(pequeno, cv2.COLOR_BGR2GRAY)
        houve_corte = False
        if self._quadro_anterior_cena is not None:
            diferenca = cv2.absdiff(self._quadro_anterior_cena, pequeno_cinza)
            houve_corte = float(np.mean(diferenca)) > LIMIAR_CORTE_CENA
        self._quadro_anterior_cena = pequeno_cinza
        return houve_corte

    def _atualizar_rastreamento(self, deteccoes):
        correspondencias, novas, sem_correspondencia = _associar_deteccoes(
            self._slots, deteccoes, DISTANCIA_MAXIMA_CORRESPONDENCIA
        )

        for id_slot, (x1, y1, x2, y2, _conf) in correspondencias.items():
            slot = self._slots[id_slot]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            tam = _calcular_tamanho_alvo(y2 - y1)
            if abs(cx - slot.alvo_x) > BANDA_MORTA_POSICAO:
                slot.alvo_x = cx
            if abs(cy - slot.alvo_y) > BANDA_MORTA_POSICAO:
                slot.alvo_y = cy
            if abs(tam - slot.tamanho_alvo) > BANDA_MORTA_TAMANHO:
                slot.tamanho_alvo = tam
            slot.frames_sem_deteccao = 0
            if not slot.confirmado:
                slot.frames_detectado += 1

        for id_slot in sem_correspondencia:
            self._slots[id_slot].frames_sem_deteccao += 1

        for x1, y1, x2, y2, _conf in novas:
            novo = _SlotRosto(id=self._proximo_id)
            self._proximo_id += 1
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            tam = _calcular_tamanho_alvo(y2 - y1)
            novo.centro_x = novo.alvo_x = cx
            novo.centro_y = novo.alvo_y = cy
            novo.tamanho = novo.tamanho_alvo = tam
            novo.frames_detectado = 1
            self._slots[novo.id] = novo

    def _promover_e_remover(self) -> bool:
        ids_confirmados_antes = set(self._ordem_confirmados)

        for slot in list(self._slots.values()):
            if not slot.confirmado and slot.frames_detectado >= self._frames_confirmacao:
                slot.confirmado = True
                self._ordem_confirmados.append(slot.id)

        for id_slot in list(self._slots.keys()):
            slot = self._slots[id_slot]
            if not slot.confirmado and slot.frames_sem_deteccao >= self._frames_descarte_candidato:
                del self._slots[id_slot]

        for id_slot in list(self._slots.keys()):
            slot = self._slots[id_slot]
            if slot.confirmado and slot.frames_sem_deteccao >= self._frames_remocao:
                del self._slots[id_slot]
                if id_slot in self._ordem_confirmados:
                    self._ordem_confirmados.remove(id_slot)
                if id_slot in self._selecionados_acima_de_4:
                    self._selecionados_acima_de_4.remove(id_slot)

        return set(self._ordem_confirmados) != ids_confirmados_antes

    def _avancar_suavizacao(self, forcar_snap: bool):
        for slot in self._slots.values():
            if forcar_snap:
                slot.centro_x, slot.centro_y, slot.tamanho = slot.alvo_x, slot.alvo_y, slot.tamanho_alvo
                continue
            slot.centro_x = _suavizar(slot.centro_x, slot.alvo_x, FATOR_SUAVIZACAO, VELOCIDADE_MAXIMA_POSICAO)
            slot.centro_y = _suavizar(slot.centro_y, slot.alvo_y, FATOR_SUAVIZACAO, VELOCIDADE_MAXIMA_POSICAO)
            slot.tamanho = _suavizar(slot.tamanho, slot.tamanho_alvo, FATOR_SUAVIZACAO_ZOOM, VELOCIDADE_MAXIMA_ZOOM)

    def _determinar_ids_para_grade(self) -> List[int]:
        confirmados = list(self._ordem_confirmados)
        if len(confirmados) <= 4:
            self._selecionados_acima_de_4.clear()
            return confirmados

        # Mais de 4 rostos confirmados: mantém um subconjunto fixo de 4,
        # trocando só quem saiu (evita reembaralhar tudo a cada quadro)
        self._selecionados_acima_de_4 = [i for i in self._selecionados_acima_de_4 if i in confirmados]
        while len(self._selecionados_acima_de_4) < 4:
            candidatos = [i for i in confirmados if i not in self._selecionados_acima_de_4]
            if not candidatos:
                break
            self._selecionados_acima_de_4.append(random.choice(candidatos))
        return self._selecionados_acima_de_4

    def _montar_saida(self, quadro_fonte: np.ndarray) -> np.ndarray:
        ids_ativos = self._determinar_ids_para_grade()
        if not ids_ativos:
            return _recorte_central(quadro_fonte, self.largura_saida, self.altura_saida)

        saida = np.zeros((self.altura_saida, self.largura_saida, 3), dtype=np.uint8)
        celulas = _calcular_grade(ids_ativos, self.largura_saida, self.altura_saida)
        for id_slot, (cx, cy, cl, ca) in celulas.items():
            slot = self._slots[id_slot]
            recorte = _recortar_e_redimensionar(quadro_fonte, slot.centro_x, slot.centro_y, slot.tamanho, cl, ca)
            saida[cy:cy + ca, cx:cx + cl] = recorte
        return saida


# ============================================================================
# Funções auxiliares gerais
# ============================================================================
def _calcular_proporcao(proporcao: str) -> float:
    """Converte uma proporção no formato 'largura:altura' em valor decimal."""
    try:
        largura, altura = proporcao.split(":")
        return float(largura) / float(altura)
    except (ValueError, ZeroDivisionError):
        print(f"[Reenquadramento] Proporção inválida: {proporcao!r}. Utilizando a proporção padrão 9:16.")
        return 9.0 / 16.0


def _dimensoes_saida(proporcao: str, altura_base: int = ALTURA_SAIDA_PADRAO) -> Tuple[int, int]:
    razao = _calcular_proporcao(proporcao)
    altura = altura_base - (altura_base % 4)
    largura = int(round(altura * razao))
    largura -= largura % 4
    return largura, altura


def _recortar_trecho(caminho_video: str, inicio: float, fim: float) -> str:
    """Recorta um trecho do vídeo utilizando FFmpeg."""
    caminho_recorte = caminho_video + ".recorte.mp4"
    comando = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", caminho_video,
        "-ss", f"{inicio:.3f}",
        "-to", f"{fim:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-hide_banner", "-stats",
        caminho_recorte,
    ]
    subprocess.run(comando, check=True)
    return caminho_recorte


def _reenquadrar_vertical(caminho_recorte: str, proporcao: str, caminho_recorte_reenquadrado: str) -> str:
    """Reenquadra o vídeo numa grade dinâmica de 1 a 4 sub-câmeras, cada uma
    acompanhando um rosto detectado, com corte seco em mudanças de cena ou de
    layout e movimento suave (posição e zoom) dentro da mesma cena."""
    largura_saida, altura_saida = _dimensoes_saida(proporcao)

    captura = cv2.VideoCapture(caminho_recorte)
    if not captura.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {caminho_recorte}")

    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
    rastreador = _RastreadorGradeDinamica(fps, largura_saida, altura_saida)

    caminho_recorte_reenquadrado_sem_audio = caminho_recorte + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    gravador = cv2.VideoWriter(caminho_recorte_reenquadrado_sem_audio, fourcc, fps, (largura_saida, altura_saida))

    while True:
        ret, quadro = captura.read()
        if not ret:
            break
        gravador.write(rastreador.processar_quadro(quadro))

    captura.release()
    gravador.release()

    # Reanexa o áudio com FFmpeg
    comando = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", caminho_recorte_reenquadrado_sem_audio,
        "-i", caminho_recorte,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        caminho_recorte_reenquadrado,
    ]
    subprocess.run(comando, check=True)

    if os.path.exists(caminho_recorte_reenquadrado_sem_audio):
        os.remove(caminho_recorte_reenquadrado_sem_audio)

    return caminho_recorte_reenquadrado


def processar_destaques(caminho_video: str, destaques: List[Dict], proporcao: str = "9:16", video_id: Optional[str] = None) -> List[Dict]:
    """Recorta e reenquadra os destaques, retornando os caminhos dos clipes gerados."""
    caminho = Path(caminho_video)
    diretorio_saida = caminho.parent

    if not diretorio_saida or str(diretorio_saida) == ".":
        diretorio_saida = Path(DIRETORIO_SAIDA)

    resultados: List[Dict] = []

    for indice, destaque in enumerate(destaques, 1):
        if video_id:
            caminho_salvamento_recorte = os.path.join(diretorio_saida, f"corte-{indice:02d}_{video_id}.mp4")
        else:
            caminho_salvamento_recorte = os.path.join(diretorio_saida, f"corte-{indice:02d}.mp4")
        print(f"[Reenquadramento] {indice}/{len(destaques)}: {destaque.get('titulo', '(sem título)')}", flush=True)

        caminho_recorte = None
        try:
            caminho_recorte = _recortar_trecho(caminho_video, float(destaque["inicio"]), float(destaque["fim"]))
            caminho_recorte_reenquadrado = _reenquadrar_vertical(caminho_recorte, proporcao, caminho_salvamento_recorte)
            resultados.append({**destaque, "clip_url": caminho_recorte_reenquadrado})
        except Exception as e:
            print(f"[Reenquadramento] {indice} falhou: {e}", flush=True)
            resultados.append({**destaque, "clip_url": None, "error": str(e)})
        finally:
            if caminho_recorte and os.path.exists(caminho_recorte):
                os.remove(caminho_recorte)

    return resultados