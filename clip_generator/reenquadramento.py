"""Recorte e processamento dos trechos selecionados para geração de clipes.

Utiliza YOLO para detecção de pessoas/cenas e aplica enquadramento dinâmico 
estilo OpusClip (telas divididas para múltiplos falantes/eventos).
"""

import os
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


def _extrair_crop_pessoa(quadro: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Extrai uma região de uma pessoa mantendo a centralização do corpo/rosto."""
    x1, y1, x2, y2 = bbox
    alt_q, larg_q = quadro.shape[:2]
    
    # Expandir margem para enquadramento de meio-corpo
    largura_box = x2 - x1
    altura_box = y2 - y1
    
    centro_x = x1 + largura_box // 2
    centro_y = y1 + altura_box // 2
    
    dimensao = max(largura_box, altura_box)
    
    nx1 = max(0, centro_x - dimensao // 2)
    ny1 = max(0, centro_y - dimensao // 2)
    nx2 = min(larg_q, centro_x + dimensao // 2)
    ny2 = min(alt_q, centro_y + dimensao // 2)
    
    crop = quadro[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return cv2.resize(quadro, (360, 640))
    return crop


def _montar_grid_dinamico(quadro: np.ndarray, pessoas_bboxes: List[Tuple[int, int, int, int]], larg_alvo: int, alt_alvo: int) -> np.ndarray:
    """Gera a imagem final montando o layout em Grid conforme a quantidade de pessoas."""
    qtd = len(pessoas_bboxes)
    
    # Caso 0 ou 1 Pessoa: Tela Cheia 9:16
    if qtd <= 1:
        if qtd == 1:
            crop = _extrair_crop_pessoa(quadro, pessoas_bboxes[0])
            return cv2.resize(crop, (larg_alvo, alt_alvo))
        else:
            # Padrão central se não houver detecção
            alt_orig, larg_orig = quadro.shape[:2]
            crop_w = int(alt_orig * (9 / 16))
            cx = larg_orig // 2
            crop = quadro[:, max(0, cx - crop_w // 2):min(larg_orig, cx + crop_w // 2)]
            return cv2.resize(crop, (larg_alvo, alt_alvo))

    canvas = np.zeros((alt_alvo, larg_alvo, 3), dtype=np.uint8)

    # Caso 2 Pessoas: Divisão Vertical (Topo e Base)
    if qtd == 2:
        h_metade = alt_alvo // 2
        p1 = cv2.resize(_extrair_crop_pessoa(quadro, pessoas_bboxes[0]), (larg_alvo, h_metade))
        p2 = cv2.resize(_extrair_crop_pessoa(quadro, pessoas_bboxes[1]), (larg_alvo, alt_alvo - h_metade))
        canvas[0:h_metade, :] = p1
        canvas[h_metade:, :] = p2

    # Caso 3 Pessoas: Layout "T" (2 em Cima, 1 Embaixo)
    elif qtd == 3:
        h_metade = alt_alvo // 2
        w_metade = larg_alvo // 2
        
        p1 = cv2.resize(_extrair_crop_pessoa(quadro, pessoas_bboxes[0]), (w_metade, h_metade))
        p2 = cv2.resize(_extrair_crop_pessoa(quadro, pessoas_bboxes[1]), (larg_alvo - w_metade, h_metade))
        p3 = cv2.resize(_extrair_crop_pessoa(quadro, pessoas_bboxes[2]), (larg_alvo, alt_alvo - h_metade))
        
        canvas[0:h_metade, 0:w_metade] = p1
        canvas[0:h_metade, w_metade:] = p2
        canvas[h_metade:, :] = p3

    # Caso 4 ou mais Pessoas: Grid 2x2
    else:
        h_metade = alt_alvo // 2
        w_metade = larg_alvo // 2
        
        p1 = cv2.resize(_extrair_crop_pessoa(quadro, pessoas_bboxes[0]), (w_metade, h_metade))
        p2 = cv2.resize(_extrair_crop_pessoa(quadro, pessoas_bboxes[1]), (larg_alvo - w_metade, h_metade))
        p3 = cv2.resize(_extrair_crop_pessoa(quadro, pessoas_bboxes[2]), (w_metade, alt_alvo - h_metade))
        p4 = cv2.resize(_extrair_crop_pessoa(quadro, pessoas_bboxes[3]), (larg_alvo - w_metade, alt_alvo - h_metade))
        
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
    """Recorta o vídeo adaptando dinamicamente para grids de 1 a 4 pessoas via YOLO."""
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "A biblioteca Ultralytics é necessária para o reenquadramento YOLO. Instale-a com:\n"
            "    pip install ultralytics"
        ) from e

    # Carrega modelo YOLO leve pré-treinado
    modelo_yolo = YOLO("yolov8n.pt")

    captura = cv2.VideoCapture(caminho_recorte)
    if not captura.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {caminho_recorte}")

    altura_origem = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0

    # Resolução padrão do vídeo final (1080x1920)
    largura_saida = 1080
    altura_saida = 1920

    caminho_recorte_reenquadrado_sem_audio = caminho_recorte + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    gravador = cv2.VideoWriter(caminho_recorte_reenquadrado_sem_audio, fourcc, fps, (largura_saida, altura_saida))

    quadro_anterior: Optional[np.ndarray] = None
    bboxes_atuais: List[Tuple[int, int, int, int]] = []

    while True:
        ret, quadro = captura.read()
        if not ret:
            break

        # Detecta troca de cena no vídeo original (Corte Seco)
        troca_de_cena = _detectar_corte_de_cena(quadro, quadro_anterior)

        # Atualiza a detecção de pessoas a cada troca de cena ou periodicamente
        if troca_de_cena or len(bboxes_atuais) == 0:
            resultados = modelo_yolo(quadro, verbose=False)[0]
            novas_boxes = []
            
            for box in resultados.boxes:
                classe = int(box.cls[0])
                confianca = float(box.conf[0])
                
                # Classe 0 = 'person' no modelo YOLO COCO
                if classe == 0 and confianca > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Filtra pessoas muito pequenas no fundo
                    if (y2 - y1) > (altura_origem * 0.15):
                        novas_boxes.append((x1, y1, x2, y2))
            
            # Ordena da esquerda para a direita na tela
            novas_boxes = sorted(novas_boxes, key=lambda b: b[0])
            bboxes_atuais = novas_boxes[:4]  # Limita a 4 pessoas max no grid

        # Gera o quadro final no layout em grid correto
        quadro_final = _montar_grid_dinamico(quadro, bboxes_atuais, largura_saida, altura_saida)
        gravador.write(quadro_final)
        
        quadro_anterior = quadro.copy()

    captura.release()
    gravador.release()

    # Reanexa o áudio usando FFmpeg
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
        
        print(f"[clipe] {indice}/{len(destaques)}: {destaque.get('titulo', '(sem título)')}", flush=True)

        caminho_recorte = None
        try:
            caminho_recorte = _recortar_trecho(caminho_video, float(destaque["inicio"]), float(destaque["fim"]))
            caminho_recorte_reenquadrado = _reenquadrar_vertical(caminho_recorte, proporcao, caminho_salvamento_recorte)
            resultados.append({**destaque, "clip_url": caminho_recorte_reenquadrado})
        except Exception as e:
            print(f"[clipe] {indice} falhou: {e}", flush=True)
            resultados.append({**destaque, "clip_url": None, "error": str(e)})
        finally:
            if caminho_recorte and os.path.exists(caminho_recorte):
                os.remove(caminho_recorte)

    return resultados