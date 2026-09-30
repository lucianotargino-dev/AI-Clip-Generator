"""Transcrição local utilizando faster-whisper.

Lê um arquivo de mídia local e retorna os dados no formato esperado pelo
gerador de cortes:
{duracao, segmentos[{inicio, fim, texto, palavras[{inicio, fim, palavra}]}]}.
"""

import os
import json
import subprocess
import wave
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional

from .configuracao import DIRETORIO_SAIDA, DISPOSITIVO_WHISPER, MODELO_WHISPER


def _obter_caminho_para_transcricao(caminho_midia: str, video_id: Optional[str] = None) -> Path:
    """Retorna o caminho onde será salvo o arquivo JSON da transcrição."""

    caminho = Path(caminho_midia)
    diretorio_saida = caminho.parent

    if not diretorio_saida or str(diretorio_saida) == ".":
        diretorio_saida = Path(DIRETORIO_SAIDA)

    # diretorio_saida.mkdir(parents=True, exist_ok=True)

    if video_id:
        return diretorio_saida / f"transcricao_{video_id}.json"

    return diretorio_saida / "transcricao.json"


def _salvar_transcricao_json(caminho_video: str, transcricao: Dict, video_id: Optional[str] = None) -> Path:
    """Salva a transcrição completa em um arquivo JSON."""
    caminho_transcricao = _obter_caminho_para_transcricao(caminho_video, video_id)
    caminho_transcricao.write_text(json.dumps(transcricao, ensure_ascii=False, indent=4), encoding="utf-8")
    return caminho_transcricao


def _carregar_transcricao_json(caminho_transcricao: Path) -> Dict:
    """Carrega uma transcrição de um arquivo JSON."""
    return json.loads(caminho_transcricao.read_text(encoding="utf-8"))


def _obter_dispositivo_whisper() -> str:
    """Determina o dispositivo que será utilizado pelo Whisper."""
    if DISPOSITIVO_WHISPER != "auto":
        return DISPOSITIVO_WHISPER
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            # Verifica se o CUDA realmente está funcional (detecta ausência de bibliotecas como cuBLAS e cuDNN)
            torch.zeros(1, device="cuda")
            return "cuda"
    except (ImportError, OSError, RuntimeError):
        pass
    return "cpu"


def _converter_para_wav(arquivo_video: str, video_id: Optional[str]) -> str:
    caminho = Path(arquivo_video)
    diretorio_saida = caminho.parent

    if not diretorio_saida or str(diretorio_saida) == ".":
        diretorio_saida = Path(DIRETORIO_SAIDA)

    if video_id:
        caminho_audio = diretorio_saida / f"audio_{video_id}.wav"
    else:
        caminho_audio = diretorio_saida / "audio.wav"
    
    comando = [
        "ffmpeg",
        "-y",               # sobrescreve arquivo de audio se existir
        "-i", arquivo_video,

        "-ar", "16000",     # sample rate
        "-ac", "1",         # mono
        "-loglevel", "error",
        "-hide_banner",
        "-stats",

        caminho_audio
    ]

    subprocess.run(comando)

    return str(caminho_audio)


def _extrair_duracao_audio(arquivo_audio:str) -> float:
    with wave.open(arquivo_audio, "r") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        duracao = frames / float(rate)

    return duracao


def transcrever(caminho_video: str, idioma: Optional[str] = None, video_id: Optional[str] = None) -> Dict:
    """Transcreve uma mídia e reutiliza uma transcrição JSON existente quando possível."""
    caminho_transcricao = _obter_caminho_para_transcricao(caminho_video, video_id)
    if caminho_transcricao.exists():
        data_modificacao_video = os.path.getmtime(caminho_video)
        data_modificacao_transcricao = caminho_transcricao.stat().st_mtime
        if data_modificacao_transcricao >= data_modificacao_video:
            print(f"[Transcrição] Reutilizando transcrição existente: {caminho_transcricao}", flush=True)
            transcricao_salva = _carregar_transcricao_json(caminho_transcricao)
            # Considera uma transcrição vazia como inválida
            # (normalmente causada por uma execução interrompida).
            # Deleta e refaz a transcrição nesse caso.
            if not transcricao_salva["segmentos"] or transcricao_salva["duracao"] <= 0.0:
                print(f"[Transcrição] A transcrição existente está vazia ou inválida. Removendo arquivo: {caminho_transcricao}", flush=True)
                caminho_transcricao.unlink(missing_ok=True)
            else:
                print(f"[Transcrição] {len(transcricao_salva['segmentos'])} segmentos encontrados, {transcricao_salva['duracao']:.0f} segundos de áudio.", flush=True)
                return transcricao_salva

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "A biblioteca faster-whisper é necessária para realizar a transcrição. Instale-a com:\n"
            "    pip install -r requirements.txt"
        ) from e

    dispositivo_whisper = _obter_dispositivo_whisper()
    tipo_computacao = "float16" if dispositivo_whisper == "cuda" else "int8"
    print(f"[Transcrição] Faster Whisper | Modelo: {MODELO_WHISPER} | Dispositivo: {dispositivo_whisper}", flush=True)

    from .configuracao import FILTRO_WHISPER_VAD, PARAMETROS_WHISPER_VAD

    modelo_transcricao = WhisperModel(MODELO_WHISPER, device=dispositivo_whisper, compute_type=tipo_computacao)

    caminho_audio = _converter_para_wav(caminho_video, video_id)
    if not os.path.exists(caminho_audio):
        raise RuntimeError("[Trancrição] Erro ao extrair audio do video")
    else:
        print(f"[Transcrição] Áudio extraído com sucesso: {caminho_audio}", flush=True)

    parametros_transcricao = {
        "audio": caminho_audio,
        "language": idioma,
        "beam_size": 5,
        "condition_on_previous_text": False,
        "word_timestamps": True,
    }

    if FILTRO_WHISPER_VAD:
        parametros_transcricao["vad_filter"] = True
        parametros_transcricao["vad_parameters"] = PARAMETROS_WHISPER_VAD
    else:
        parametros_transcricao["vad_filter"] = False

    iterador_segmentos, informacoes = modelo_transcricao.transcribe(**parametros_transcricao)

    segmentos = []

    duracao_audio = _extrair_duracao_audio(caminho_audio)

    barra_progresso = tqdm(total=duracao_audio, unit="s", desc="Transcrevendo")

    for segmento in iterador_segmentos:

        palavras = []

        if segmento.words:
            for palavra in segmento.words:
                palavras.append({
                    "inicio": float(palavra.start),
                    "fim": float(palavra.end),
                    "palavra": (palavra.word or "").strip(),
                })

        segmentos.append({
            "inicio": float(segmento.start),
            "fim": float(segmento.end),
            "texto": (segmento.text or "").strip(),
            "palavras": palavras,
        })

        novo_valor = min(segmento.end, duracao_audio)
        progresso = novo_valor - barra_progresso.n
        barra_progresso.update(progresso)

    barra_progresso.close()

    duracao = float(getattr(informacoes, "duration", 0.0)) or (segmentos[-1]["fim"] if segmentos else 0.0)
    print(f"[Transcrição] {len(segmentos)} segmentos gerados, {duracao:.0f} segundos de áudio", flush=True)
    transcricao = {"duracao": duracao, "segmentos": segmentos}
    caminho_transcricao = _salvar_transcricao_json(caminho_video, transcricao, video_id)
    print(f"[Transcrição] Transcrição salva em: {caminho_transcricao}", flush=True)
    return transcricao