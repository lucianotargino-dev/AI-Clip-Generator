"""Recorte e processamento dos trechos selecionados para geração de clipes.

O módulo recebe o vídeo de origem e os momentos identificados pelo módulo
de destaques e gera os respectivos arquivos de vídeo.
"""

import os
import subprocess
from typing import Dict, List, Optional, Tuple

from .configuracao import DIRETORIO_SAIDA


def _calcular_proporcao(proporcao: str) -> float:
    """Converte uma proporção no formato 'largura:altura' em valor decimal."""
    try:
        largura, altura = proporcao.split(":")
        return float(largura) / float(altura)
    except (ValueError, ZeroDivisionError):
        print(f"[Reenquadramento] Proporção inválida: {proporcao!r}. Utilizando a proporção padrão 9:16.")
        return 9.0 / 16.0


def _recortar_trecho(caminho_origem: str, inicio: float, fim: float, caminho_saida: str) -> str:
    """Recorta um trecho do vídeo utilizando FFmpeg."""
    comando = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", caminho_origem,
        "-ss", f"{inicio:.3f}",
        "-to", f"{fim:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        caminho_saida,
    ]
    subprocess.run(comando, check=True)
    return caminho_saida


def _reenquadrar_vertical(caminho_entrada: str, caminho_saida: str, proporcao: str) -> str:
    """Recorta o vídeo para a proporção desejada, acompanhando rostos quando possível."""
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "A biblioteca opencv-python é necessária para o reenquadramento. "
            "Instale-a com:\n"
            "    pip install -r requirements.txt"
        ) from e

    proporcao_alvo = _calcular_proporcao(proporcao)
    captura = cv2.VideoCapture(caminho_entrada)
    if not captura.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {caminho_entrada}")

    largura_origem = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_origem = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0

    # Calcula o maior recorte possível dentro do quadro na proporção desejada.
    if proporcao_alvo < largura_origem / altura_origem:
        altura_recorte = altura_origem
        largura_recorte = int(altura_recorte * proporcao_alvo)
    else:
        largura_recorte = largura_origem
        altura_recorte = int(largura_recorte / proporcao_alvo)
    largura_recorte = max(2, largura_recorte - (largura_recorte % 2))
    altura_recorte = max(2, altura_recorte - (altura_recorte % 2))

    classificador_rostos = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    caminho_video_sem_audio = caminho_saida + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    gravador = cv2.VideoWriter(caminho_video_sem_audio, fourcc, fps, (largura_recorte, altura_recorte))

    ultimo_centro: Optional[Tuple[int, int]] = None
    suavizacao = 0.15  # Define a velocidade de acompanhamento de uma nova posição facial.
    while True:
        ret, quadro = captura.read()
        if not ret:
            break

        escala_cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)
        rostos = classificador_rostos.detectMultiScale(escala_cinza, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(rostos) > 0:
            # Seleciona o maior rosto, geralmente correspondente ao falante.
            x, y, largura, altura = max(rostos, key=lambda rosto: rosto[2] * rosto[3])
            centro_x = x + largura // 2
            centro_y = y + altura // 2
            if ultimo_centro is None:
                ultimo_centro = (centro_x, centro_y)
            else:
                ultimo_x, ultimo_y = ultimo_centro
                ultimo_centro = (int(ultimo_x + (centro_x - ultimo_x) * suavizacao), int(ultimo_y + (centro_y - ultimo_y) * suavizacao))
        if ultimo_centro is None:
            ultimo_centro = (largura_origem // 2, altura_origem // 2)

        centro_x, centro_y = ultimo_centro
        inicio_x = max(0, min(largura_origem - largura_recorte, centro_x - largura_recorte // 2))
        inicio_y = max(0, min(altura_origem - altura_recorte, centro_y - altura_recorte // 2))
        quadro_recortado = quadro[inicio_y:inicio_y + altura_recorte, inicio_x:inicio_x + largura_recorte]
        gravador.write(quadro_recortado)

    captura.release()
    gravador.release()

    # Adiciona novamente o áudio do vídeo original ao vídeo reenquadrado.
    comando = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", caminho_video_sem_audio,
        "-i", caminho_entrada,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        caminho_saida,
    ]
    subprocess.run(comando, check=True)
    os.remove(caminho_video_sem_audio)
    return caminho_saida


def _processar_clipe(caminho_arquivo_entrada: str, inicio: float, fim: float, proporcao: str, caminho_arquivo_saida: str) -> str:
    """Recorta e reenquadra um destaque, retornando o caminho do clipe."""
    caminho_recorte = caminho_arquivo_saida + ".cut.mp4"
    try:
        _recortar_trecho(caminho_arquivo_entrada, inicio, fim, caminho_recorte)
        _reenquadrar_vertical(caminho_recorte, caminho_arquivo_saida, proporcao)
    finally:
        if os.path.exists(caminho_recorte):
            os.remove(caminho_recorte)
    return caminho_arquivo_saida


def processar_destaques(caminho_arquivo_entrada: str, destaques: List[Dict], proporcao: str = "9:16", diretorio_saida: Optional[str] = None) -> List[Dict]:
    """Recorta e reenquadra os destaques, retornando os caminhos dos clipes gerados."""
    diretorio_saida = diretorio_saida or DIRETORIO_SAIDA
    os.makedirs(diretorio_saida, exist_ok=True)
    resultados: List[Dict] = []
    for indice, destaque in enumerate(destaques, 1):
        caminho_arquivo_saida = os.path.join(diretorio_saida, f"short_{indice:02d}.mp4")
        print(f"[clipe] {indice}/{len(destaques)}: {destaque.get('title', '(sem título)')}", flush=True)
        try:
            _processar_clipe(caminho_arquivo_entrada, float(destaque["start_time"]), float(destaque["end_time"]), proporcao, caminho_arquivo_saida)
            resultados.append({**destaque, "clip_url": caminho_arquivo_saida})
        except Exception as e:
            print(f"[clipe] {indice} falhou: {e}", flush=True)
            resultados.append({**destaque, "clip_url": None, "error": str(e)})
    return resultados