"""Recorte e processamento dos trechos selecionados para geração de clipes.

O módulo recebe o vídeo de origem e os momentos identificados pelo módulo
de destaques e gera os respectivos arquivos de vídeo.
"""

import os
import subprocess
from pathlib import Path
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
    """Recorta o vídeo para a proporção desejada, acompanhando o rosto de forma contínua e fluida."""
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "A biblioteca OpenCV é necessária para o reenquadramento. "
            "Instale-a com:\n"
            "    pip install -r requirements.txt"
        ) from e

    proporcao_alvo = _calcular_proporcao(proporcao)

    captura = cv2.VideoCapture(caminho_recorte)
    if not captura.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {caminho_recorte}")

    largura_origem = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_origem = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0

    # Calcula as dimensões do recorte mantendo a proporção vertical
    if proporcao_alvo < largura_origem / altura_origem:
        altura_recorte = altura_origem
        largura_recorte = int(altura_recorte * proporcao_alvo)
    else:
        largura_recorte = largura_origem
        altura_recorte = int(largura_recorte / proporcao_alvo)

    largura_recorte = max(2, largura_recorte - (largura_recorte % 2))
    altura_recorte = max(2, altura_recorte - (altura_recorte % 2))

    classificador_rostos = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    caminho_recorte_reenquadrado_sem_audio = caminho_recorte + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    gravador = cv2.VideoWriter(caminho_recorte_reenquadrado_sem_audio, fourcc, fps, (largura_recorte, altura_recorte))

    # Posições para tracking contínuo
    ultimo_centro: Optional[Tuple[int, int]] = None
    
    # Parâmetros de ajuste fino
    suavizacao = 0.08  # Fator de deslizamento (quanto menor, mais suave o acompanhamento da câmera)
    tamanho_minimo_rosto = int(altura_origem * 0.10)  # Evita falsos positivos menores
    tolerancia_ruido = 4  # Ignora tremores microscópicos do detector de face quando a pessoa está parada

    while True:
        ret, quadro = captura.read()
        if not ret:
            break
        escala_cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)
        
        rostos = classificador_rostos.detectMultiScale(
            escala_cinza,
            scaleFactor=1.08,
            minNeighbors=8,
            minSize=(tamanho_minimo_rosto, tamanho_minimo_rosto)
        )

        if len(rostos) > 0:
            # Seleciona o maior rosto em tela
            x, y, largura, altura = max(rostos, key=lambda f: f[2] * f[3])
            centro_x = x + largura // 2
            centro_y = y + altura // 2
            if ultimo_centro is None:
                ultimo_centro = (centro_x, centro_y)
            else:
                ultimo_x, ultimo_y = ultimo_centro
                
                # Só atualiza a posição se a variação for maior que o ruído insignificante do detector
                delta_x = centro_x - ultimo_x
                delta_y = centro_y - ultimo_y
                
                novo_x = int(ultimo_x + delta_x * suavizacao) if abs(delta_x) > tolerancia_ruido else ultimo_x
                novo_y = int(ultimo_y + delta_y * suavizacao) if abs(delta_y) > tolerancia_ruido else ultimo_y
                
                ultimo_centro = (novo_x, novo_y)

        # SE NÃO DETECTAR: MANTÉM O ÚLTIMO PONTO (Não reseta pro centro da tela)
        if ultimo_centro is None:
            ultimo_centro = (largura_origem // 2, altura_origem // 2)
        centro_x, centro_y = ultimo_centro

        # Garante o enquadramento dentro dos limites da mídia
        inicio_x = max(0, min(largura_origem - largura_recorte, centro_x - largura_recorte // 2))
        inicio_y = max(0, min(altura_origem - altura_recorte, centro_y - altura_recorte // 2))

        quadro_recortado = quadro[inicio_y:inicio_y + altura_recorte, inicio_x:inicio_x + largura_recorte]
        gravador.write(quadro_recortado)

    captura.release()
    gravador.release()

    # Junta o áudio com FFmpeg
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