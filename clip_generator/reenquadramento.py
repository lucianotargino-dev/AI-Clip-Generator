"""Recorte e processamento dos trechos selecionados para geração de clipes.

Utiliza YOLO para detecção de pessoas/cenas, aplica enquadramento dinâmico 
estilo OpusClip (grids de 1 a 4 câmeras) e garante o cálculo estrito de aspect 
ratio (crop estilo 'cover' sem achatamento) com acompanhamento suave e cortes secos.
"""

import os
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .configuracao import DIRETORIO_SAIDA


def _calcular_proporcao(proporcao: str) -> float:
    """Converte uma proporção no formato 'largura:altura' em valor decimal."""
    try:
        largura, altura = proporcao.split(":")
        return float(largura) / float(altura)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


def _detectar_corte_de_cena(quadro_atual: np.ndarray, quadro_anterior: Optional[np.ndarray], limiar: float = 0.45) -> bool:
    """Identifica se houve uma mudança brusca de ângulo/câmera no vídeo original."""
    if quadro_anterior is None:
        return True
    
    hist_atual = cv2.calcHist([quadro_atual], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_ant = cv2.calcHist([quadro_anterior], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    cv2.normalize(hist_atual, hist_atual)
    cv2.normalize(hist_ant, hist_ant)
    
    distancia = cv2.compareHist(hist_atual, hist_ant, cv2.HISTCMP_CORREL)
    return distancia < (1.0 - limiar)


def _crop_proporcional_cover(quadro: np.ndarray, centro_x: float, centro_y: float, larg_destino: int, alt_destino: int) -> np.ndarray:
    """Realiza o recorte focado no centro mantendo a proporção exata de destino (estilo object-fit: cover).
    
    Garante ZERO achatamento ou esticamento da imagem.
    """
    alt_q, larg_q = quadro.shape[:2]
    proporcao_destino = larg_destino / alt_destino

    # Calcula as dimensões do recorte ideal na imagem original
    if larg_q / alt_q > proporcao_destino:
        # A imagem de origem é mais larga do que o espaço de destino
        alt_crop = alt_q
        larg_crop = int(alt_crop * proporcao_destino)
    else:
        # A imagem de origem é mais alta do que o espaço de destino
        larg_crop = larg_q
        alt_crop = int(larg_crop / proporcao_destino)

    # Garante que as bordas fiquem dentro do quadro original
    x1 = max(0, min(larg_q - larg_crop, int(centro_x - larg_crop / 2)))
    y1 = max(0, min(alt_q - alt_crop, int(centro_y - alt_crop / 2)))
    x2 = x1 + larg_crop
    y2 = y1 + alt_crop

    crop = quadro[y1:y2, x1:x2]
    
    if crop.size == 0:
        crop = quadro

    # Redimensiona mantendo o aspect ratio perfeito
    return cv2.resize(crop, (larg_destino, alt_destino), interpolation=cv2.INTER_LANCZOS4)


def _montar_grid_dinamico(quadro: np.ndarray, centros_atuais: List[Tuple[float, float]], larg_alvo: int, alt_alvo: int) -> np.ndarray:
    """Gera o quadro final montando o layout em Grid (1 a 4 câmeras) sem achatar imagens."""
    qtd = len(centros_atuais)
    
    # Caso 0 ou 1 Pessoa: Tela Cheia 9:16
    if qtd <= 1:
        if qtd == 1:
            cx, cy = centros_atuais[0]
            return _crop_proporcional_cover(quadro, cx, cy, larg_alvo, alt_alvo)
        else:
            # Padrão centralizado se nada for detectado
            alt_orig, larg_orig = quadro.shape[:2]
            return _crop_proporcional_cover(quadro, larg_orig / 2, alt_orig / 2, larg_alvo, alt_alvo)

    canvas = np.zeros((alt_alvo, larg_alvo, 3), dtype=np.uint8)

    # Caso 2 Pessoas: Divisão Vertical (Topo e Base)
    if qtd == 2:
        h_metade = alt_alvo // 2
        p1 = _crop_proporcional_cover(quadro, centros_atuais[0][0], centros_atuais[0][1], larg_alvo, h_metade)
        p2 = _crop_proporcional_cover(quadro, centros_atuais[1][0], centros_atuais[1][1], larg_alvo, alt_alvo - h_metade)
        canvas[0:h_metade, :] = p1
        canvas[h_metade:, :] = p2

    # Caso 3 Pessoas: Layout em "T" (2 no topo lado a lado, 1 na base)
    elif qtd == 3:
        h_metade = alt_alvo // 2
        w_metade = larg_alvo // 2
        
        p1 = _crop_proporcional_cover(quadro, centros_atuais[0][0], centros_atuais[0][1], w_metade, h_metade)
        p2 = _crop_proporcional_cover(quadro, centros_atuais[1][0], centros_atuais[1][1], larg_alvo - w_metade, h_metade)
        p3 = _crop_proporcional_cover(quadro, centros_atuais[2][0], centros_atuais[2][1], larg_alvo, alt_alvo - h_metade)
        
        canvas[0:h_metade, 0:w_metade] = p1
        canvas[0:h_metade, w_metade:] = p2
        canvas[h_metade:, :] = p3

    # Caso 4 Pessoas: Grid 2x2
    else:
        h_metade = alt_alvo // 2
        w_metade = larg_alvo // 2
        
        p1 = _crop_proporcional_cover(quadro, centros_atuais[0][0], centros_atuais[0][1], w_metade, h_metade)
        p2 = _crop_proporcional_cover(quadro, centros_atuais[1][0], centros_atuais[1][1], larg_alvo - w_metade, h_metade)
        p3 = _crop_proporcional_cover(quadro, centros_atuais[2][0], centros_atuais[2][1], w_metade, alt_alvo - h_metade)
        p4 = _crop_proporcional_cover(quadro, centros_atuais[3][0], centros_atuais[3][1], larg_alvo - w_metade, alt_alvo - h_metade)
        
        canvas[0:h_metade, 0:w_metade] = p1
        canvas[0:h_metade, w_metade:] = p2
        canvas[h_metade:, 0:w_metade] = p3
        canvas[h_metade:, w_metade:] = p4

    return canvas


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
    """Recorta o vídeo adaptando dinamicamente para grids de 1 a 4 pessoas via YOLO com rastreamento suave."""
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "A biblioteca Ultralytics é necessária para o reenquadramento YOLO. Instale-a com:\n"
            "    pip install ultralytics"
        ) from e

    modelo_yolo = YOLO("yolov8n.pt")

    captura = cv2.VideoCapture(caminho_recorte)
    if not captura.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {caminho_recorte}")

    altura_origem = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0

    largura_saida = 1080
    altura_saida = 1920

    caminho_recorte_reenquadrado_sem_audio = caminho_recorte + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    gravador = cv2.VideoWriter(caminho_recorte_reenquadrado_sem_audio, fourcc, fps, (largura_saida, altura_saida))

    quadro_anterior: Optional[np.ndarray] = None
    
    # Armazena os centros dos alvos
    centros_atuais: List[Tuple[float, float]] = []
    centros_alvo: List[Tuple[float, float]] = []
    
    suavizacao = 0.08  # Fator de deslizamento suave do tracking (LERP)

    while True:
        ret, quadro = captura.read()
        if not ret:
            break

        troca_de_cena = _detectar_corte_de_cena(quadro, quadro_anterior)

        # Executa detecção YOLO
        resultados = modelo_yolo(quadro, verbose=False)[0]
        novos_centros = []

        for box in resultados.boxes:
            classe = int(box.cls[0])
            confianca = float(box.conf[0])
            
            # Classe 0 = 'person' no modelo YOLO
            if classe == 0 and confianca > 0.35:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (y2 - y1) > (altura_origem * 0.12):
                    cx = float(x1 + (x2 - x1) / 2)
                    cy = float(y1 + (y2 - y1) / 2)
                    novos_centros.append((cx, cy))

        # Se houver mais de 4 alvos, escolhe 4 aleatoriamente de forma consistente
        if len(novos_centros) > 4:
            # Caso haja corte de cena, re-sorteia os 4 alvos aleatórios
            if troca_de_cena:
                random.seed(int(captura.get(cv2.CAP_PROP_POS_FRAMES)))
            random.shuffle(novos_centros)
            novos_centros = novos_centros[:4]

        # Ordena da esquerda para a direita para manter posições lógicas no grid
        novos_centros = sorted(novos_centros, key=lambda c: c[0])

        # Se houver mudança brusca de cena ou alteração na quantidade de pessoas -> CORTE SECO
        if troca_de_cena or len(novos_centros) != len(centros_alvo):
            centros_alvo = novos_centros
            centros_atuais = list(novos_centros)  # Pula transição suave (corte seco imediato)
        else:
            centros_alvo = novos_centros

            # Acompanhamento suave (LERP) para pequenos movimentos dentro da mesma cena
            novos_atuais = []
            for idx in range(len(centros_atuais)):
                cx_atual, cy_atual = centros_atuais[idx]
                cx_alvo, cy_alvo = centros_alvo[idx]

                nx = cx_atual + (cx_alvo - cx_atual) * suavizacao
                ny = cy_atual + (cy_alvo - cy_atual) * suavizacao
                novos_atuais.append((nx, ny))

            centros_atuais = novos_atuais

        # Monta a imagem final sem achatar e com tracking aplicado
        quadro_final = _montar_grid_dinamico(quadro, centros_atuais, largura_saida, altura_saida)
        gravador.write(quadro_final)

        quadro_anterior = quadro.copy()

    captura.release()
    gravador.release()

    # Reanexa o áudio via FFmpeg
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