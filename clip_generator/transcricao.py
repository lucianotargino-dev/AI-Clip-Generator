"""Transcrição local utilizando faster-whisper.

Lê um arquivo de mídia local e retorna os dados no formato esperado pelo
gerador de cortes:
{duracao, segmentos[inicio, fim, texto]}.
"""

import os
import re
from pathlib import Path
from typing import Dict, Optional

from .configuracao import DIRETORIO_SAIDA, DISPOSITIVO_WHISPER, MODELO_WHISPER


def _obter_caminho_para_transcricao(caminho_midia: str) -> Path:
    """Retorna o caminho onde será salvo o arquivo de transcrição (.srt) da mídia."""
    diretorio_saida = Path(DIRETORIO_SAIDA)
    diretorio_saida.mkdir(parents=True, exist_ok=True)
    return diretorio_saida / (Path(caminho_midia).stem + ".srt")


def _converter_segundos_em_timestamp_srt(segundos: float) -> str:
    """Converte segundos para timestamp SRT."""
    total_milissegundos = max(0, int(round(segundos * 1000)))
    milissegundos = total_milissegundos % 1000
    total_segundos = total_milissegundos // 1000
    segundos = total_segundos % 60
    total_minutos = total_segundos // 60
    minutos = total_minutos % 60
    horas = total_minutos // 60
    return f"{horas:02d}:{minutos:02d}:{segundos:02d},{milissegundos:03d}"


def _converter_timestamp_srt_em_segundos(timestamp_srt: str) -> float:
    """Converte timestamp SRT para segundos."""
    resultado = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", timestamp_srt.strip())
    if not resultado:
        raise ValueError(f"Formato de timestamp SRT inválido: {timestamp_srt!r}")
    horas, minutos, segundos, milissegundos = map(int, resultado.groups())
    return horas * 3600 + minutos * 60 + segundos + (milissegundos / 1000.0)


def _salvar_transcricao_srt(caminho_video: str, transcricao: Dict) -> Path:
    """Gera um arquivo de transcrição no formato SRT."""
    caminho_transcricao = _obter_caminho_para_transcricao(caminho_video)
    linhas = []
    for indice, segmento in enumerate(transcricao.get("segmentos", []), start=1):
        inicio = _converter_segundos_em_timestamp_srt(float(segmento["inicio"]))
        fim = _converter_segundos_em_timestamp_srt(float(segmento["fim"]))
        texto = str(segmento.get("texto", "")).strip().replace("\r", "").replace("\n", " ")
        linhas.append(str(indice))
        linhas.append(f"{inicio} --> {fim}")
        linhas.append(texto)
        linhas.append("")

    caminho_transcricao.write_text("\n".join(linhas), encoding="utf-8")
    return caminho_transcricao


def _carregar_transcricao_srt(caminho_transcricao: Path) -> Dict:
    """Carrega uma transcrição de um arquivo no formato SRT."""
    conteudo = caminho_transcricao.read_text(encoding="utf-8-sig").strip()
    if not conteudo:
        return {"duracao": 0.0, "segmentos": []}

    segmentos = []
    for bloco in re.split(r"\n\s*\n", conteudo):
        linhas = [linha.strip("\ufeff") for linha in bloco.splitlines() if linha.strip()]
        if not linhas:
            continue
        if "-->" not in linhas[0] and len(linhas) > 1 and "-->" in linhas[1]:
            linhas = linhas[1:]
        if not linhas or "-->" not in linhas[0]:
            continue
        inicio_texto, fim_texto = [parte.strip() for parte in linhas[0].split("-->", 1)]
        texto = "\n".join(linhas[1:]).strip()
        segmentos.append(
            {
                "inicio": _converter_timestamp_srt_em_segundos(inicio_texto),
                "fim": _converter_timestamp_srt_em_segundos(fim_texto),
                "texto": texto,
            }
        )

    duracao = segmentos[-1]["fim"] if segmentos else 0.0
    return {"duracao": duracao, "segmentos": segmentos}


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


def transcrever(caminho_midia: str, idioma: Optional[str] = None) -> Dict:
    """Transcreve uma mídia e reutiliza uma transcrição SRT existente quando possível."""
    caminho_transcricao = _obter_caminho_para_transcricao(caminho_midia)
    if caminho_transcricao.exists():
        data_modificacao_midia = os.path.getmtime(caminho_midia)
        data_modificacao_transcricao = caminho_transcricao.stat().st_mtime
        if data_modificacao_transcricao >= data_modificacao_midia:
            print(f"[Transcrição] Reutilizando transcrição existente: {caminho_transcricao}", flush=True)
            transcricao_salva = _carregar_transcricao_srt(caminho_transcricao)
            # Considera uma transcrição vazia como inválida
            # (normalmente causada por uma execução interrompida).
            # Deleta e refaz a transcrição nesse caso.
            if not transcricao_salva["segmentos"] or transcricao_salva["duracao"] <= 0.0:
                print(f"[Transcrição] A transcrição existente está vazia ou inválida. Removendo arquivo.", flush=True)
                caminho_transcricao.unlink(missing_ok=True)
            else:
                print(f"[Transcrição] {len(transcricao_salva['segmentos'])} segmentos encontrados ({transcricao_salva['duracao']:.0f}s).", flush=True)
                return transcricao_salva

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "A biblioteca faster-whisper é necessária para realizar a transcrição.\n"
            "Instale-a com:\n"
            "    pip install -r requirements.txt"
        ) from e

    dispositivo = _obter_dispositivo_whisper()
    tipo_computacao = "float16" if dispositivo == "cuda" else "int8"
    print(f"[Transcrição] Modelo: {MODELO_WHISPER} | Dispositivo: {dispositivo}", flush=True)

    from .configuracao import FILTRO_WHISPER_VAD, PARAMETROS_WHISPER_VAD

    modelo = WhisperModel(MODELO_WHISPER, device=dispositivo, compute_type=tipo_computacao)

    parametros_transcricao = {
        "audio": caminho_midia,
        "language": idioma,
        "beam_size": 5,
        "condition_on_previous_text": False,
    }
    if FILTRO_WHISPER_VAD:
        parametros_transcricao["vad_filter"] = True
        parametros_transcricao["vad_parameters"] = PARAMETROS_WHISPER_VAD
    else:
        parametros_transcricao["vad_filter"] = False

    iterador_segmentos, informacoes = modelo.transcribe(**parametros_transcricao)

    segmentos = []
    for segmento in iterador_segmentos:
        segmentos.append({
            "inicio": float(segmento.start),
            "fim": float(segmento.end),
            "texto": (segmento.text or "").strip(),
        })

    duracao = float(getattr(informacoes, "duration", 0.0)) or (segmentos[-1]["fim"] if segmentos else 0.0)
    print(f"[Transcrição] {len(segmentos)} segmentos gerados, {duracao:.0f}s de áudio", flush=True)
    transcricao = {"duracao": duracao, "segmentos": segmentos}
    caminho_transcricao = _salvar_transcricao_srt(caminho_midia, transcricao)
    print(f"[Transcrição] Transcrição salva em: {caminho_transcricao}", flush=True)
    return transcricao