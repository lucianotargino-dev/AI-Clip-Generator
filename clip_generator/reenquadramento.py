"""Recorte e processamento dos trechos selecionados para geração de clipes.

Utiliza YOLO para detecção de pessoas/cenas, aplica enquadramento dinâmico 
estilo OpusClip (grids de 1 a 4 câmeras) e garante o cálculo estrito de aspect 
ratio com zoom automático em pessoas distantes e sem falsos positivos.
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


def _crop_proporcional_cover(quadro: np.ndarray, centro_x: float, centro_y: float, larg_destino: int, alt_destino: int, fator_zoom: float = 1.0) -> np.ndarray:
    """Realiza o recorte focado no centro mantendo a proporção exata de destino.
    
    Aplica fator de zoom dinâmico quando a pessoa está distante no cenário.
    """
    alt_q, larg_q = quadro.shape[:2]
    proporcao_destino = larg_destino / alt_destino

    # Calcula as dimensões base do recorte na imagem original
    if larg_q / alt_q > proporcao_destino:
        alt_crop = alt_q
        larg_crop = int(alt_crop * proporcao_destino)
    else:
        larg_crop = larg_q
        alt_crop = int(larg_crop / proporcao_destino)

    # Aplica o Zoom (reduz a janela de corte na imagem original para fechar mais o plano)
    if fator_zoom > 1.0:
        larg_crop = int(larg_crop / fator_zoom)
        alt_crop = int(alt_crop / fator_zoom)

    # Garante que as bordas fiquem estritamente dentro dos limites da mídia
    x1 = max(0, min(larg_q - larg_crop, int(centro_x - larg_crop / 2)))
    y1 = max(0, min(alt_q - alt_crop, int(centro_y - alt_crop / 2)))
    x2 = x1 + larg_crop
    y2 = y1 + alt_crop

    crop = quadro[y1:y2, x1:x2]
    
    if crop.size == 0:
        crop = quadro

    return cv2.resize(crop, (larg_destino, alt_destino), interpolation=cv2.INTER_LANCZOS4)


def _montar_grid_dinamico(quadro: np.ndarray, dados_alvos: List[Dict], larg_alvo: int, alt_alvo: int) -> np.ndarray:
    """Gera o quadro final montando o layout em Grid (1 a 4 câmeras) com zoom individual."""
    qtd = len(dados_alvos)
    
    # Caso 0 ou 1 Pessoa: Tela Cheia 9:16
    if qtd <= 1:
        if qtd == 1:
            alvo = dados_alvos[0]
            return _crop_proporcional_cover(quadro, alvo["cx"], alvo["cy"], larg_alvo, alt_alvo, alvo["zoom"])
        else:
            alt_orig, larg_orig = quadro.shape[:2]
            return _crop_proporcional_cover(quadro, larg_orig / 2, alt_orig / 2, larg_alvo, alt_alvo, 1.0)

    canvas = np.zeros((alt_alvo, larg_alvo, 3), dtype=np.uint8)

    # Caso 2 Pessoas: Divisão Vertical (Topo e Base)
    if qtd == 2:
        h_metade = alt_alvo // 2
        p1 = _crop_proporcional_cover(quadro, dados_alvos[0]["cx"], dados_alvos[0]["cy"], larg_alvo, h_metade, dados_alvos[0]["zoom"])
        p2 = _crop_proporcional_cover(quadro, dados_alvos[1]["cx"], dados_alvos[1]["cy"], larg_alvo, alt_alvo - h_metade, dados_alvos[1]["zoom"])
        canvas[0:h_metade, :] = p1
        canvas[h_metade:, :] = p2

    # Caso 3 Pessoas: Layout em "T" (2 no topo, 1 na base)
    elif qtd == 3:
        h_metade = alt_alvo // 2
        w_metade = larg_alvo // 2
        
        p1 = _crop_proporcional_cover(quadro, dados_alvos[0]["cx"], dados_alvos[0]["cy"], w_metade, h_metade, dados_alvos[0]["zoom"])
        p2 = _crop_proporcional_cover(quadro, dados_alvos[1]["cx"], dados_alvos[1]["cy"], larg_alvo - w_metade, h_metade, dados_alvos[1]["zoom"])
        p3 = _crop_proporcional_cover(quadro, dados_alvos[2]["cx"], dados_alvos[2]["cy"], larg_alvo, alt_alvo - h_metade, dados_alvos[2]["zoom"])
        
        canvas[0:h_metade, 0:w_metade] = p1
        canvas[0:h_metade, w_metade:] = p2
        canvas[h_metade:, :] = p3

    # Caso 4 Pessoas: Grid 2x2
    else:
        h_metade = alt_alvo // 2
        w_metade = larg_alvo // 2
        
        p1 = _crop_proporcional_cover(quadro, dados_alvos[0]["cx"], dados_alvos[0]["cy"], w_metade, h_metade, dados_alvos[0]["zoom"])
        p2 = _crop_proporcional_cover(quadro, dados_alvos[1]["cx"], dados_alvos[1]["cy"], larg_alvo - w_metade, h_metade, dados_alvos[1]["zoom"])
        p3 = _crop_proporcional_cover(quadro, dados_alvos[2]["cx"], dados_alvos[2]["cy"], w_metade, alt_alvo - h_metade, dados_alvos[2]["zoom"])
        p4 = _crop_proporcional_cover(quadro, dados_alvos[3]["cx"], dados_alvos[3]["cy"], larg_alvo - w_metade, alt_alvo - h_metade, dados_alvos[3]["zoom"])
        
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
    """Recorta o vídeo adaptando dinamicamente para grids via YOLO com filtro anti-falsos-positivos e zoom automático."""
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
    
    alvos_atuais: List[Dict] = []
    alvos_destino: List[Dict] = []
    
    suavizacao = 0.08

    while True:
        ret, quadro = captura.read()
        if not ret:
            break

        troca_de_cena = _detectar_corte_de_cena(quadro, quadro_anterior)

        resultados = modelo_yolo(quadro, verbose=False)[0]
        novos_alvos = []

        for box in resultados.boxes:
            classe = int(box.cls[0])
            confianca = float(box.conf[0])
            
            # --- AJUSTE 1: Limiar de Confiança mais rigoroso (60%) para ELIMINAR FALSOS POSITIVOS ---
            if classe == 0 and confianca > 0.60:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                alt_pessoa = y2 - y1
                
                # Exige altura mínima relevante para desconsiderar ruídos no cenário
                if alt_pessoa > (altura_origem * 0.15):
                    cx = float(x1 + (x2 - x1) / 2)
                    cy = float(y1 + alt_pessoa / 2)

                    # --- AJUSTE 2: Lógica de ZOOM Automático para pessoas distantes ---
                    # Se a pessoa ocupar menos de 30% da altura da tela, aplicamos zoom relativo
                    proporcao_altura = alt_pessoa / altura_origem
                    if proporcao_altura < 0.25:
                        fator_zoom = 1.6  # Zoom forte para enquadrar pessoas bem longe
                    elif proporcao_altura < 0.40:
                        fator_zoom = 1.3  # Zoom leve/médio
                    else:
                        fator_zoom = 1.0  # Sem zoom se já estiver em plano médio ou próximo

                    novos_alvos.append({"cx": cx, "cy": cy, "zoom": fator_zoom})

        if len(novos_alvos) > 4:
            if troca_de_cena:
                random.seed(int(captura.get(cv2.CAP_PROP_POS_FRAMES)))
            random.shuffle(novos_alvos)
            novos_alvos = novos_alvos[:4]

        novos_alvos = sorted(novos_alvos, key=lambda a: a["cx"])

        # Troca de cena ou alteração na quantidade de alvos -> Corte seco imediato
        if troca_de_cena or len(novos_alvos) != len(alvos_destino):
            alvos_destino = novos_alvos
            alvos_atuais = [dict(a) for a in novos_alvos]
        else:
            alvos_destino = novos_alvos
            novos_atuais = []
            
            for idx in range(len(alvos_atuais)):
                cx_at = alvos_atuais[idx]["cx"]
                cy_at = alvos_atuais[idx]["cy"]
                zoom_at = alvos_atuais[idx]["zoom"]

                cx_al = alvos_destino[idx]["cx"]
                cy_al = alvos_destino[idx]["cy"]
                zoom_al = alvos_destino[idx]["zoom"]

                # Interpolação suave para posição e zoom
                nx = cx_at + (cx_al - cx_at) * suavizacao
                ny = cy_at + (cy_al - cy_at) * suavizacao
                nz = zoom_at + (zoom_al - zoom_at) * suavizacao

                novos_atuais.append({"cx": nx, "cy": ny, "zoom": nz})

            alvos_atuais = novos_atuais

        quadro_final = _montar_grid_dinamico(quadro, alvos_atuais, largura_saida, altura_saida)
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