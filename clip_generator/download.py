"""
Implementação da obtenção de vídeos do YouTube.

Este módulo identifica arquivos locais, reutiliza downloads existentes e,
quando necessário, realiza o download do vídeo utilizando o yt-dlp,
retornando o caminho do arquivo para as próximas etapas do pipeline.
"""
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from typing import Optional

# from ..config import LOCAL_OUTPUT_DIR
DIRETORIO_SAIDA = "saida" # provisiorio


def _importar_yt_dlp():
    try:
        import yt_dlp   # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "A biblioteca 'yt-dlp' é necessária para baixar vídeos do YouTube.\n"
            "Instale-a com:\n"
            "   pip install -r requirements.txt"
        ) from e
    return yt_dlp


def _obter_seletor_resolucao(resolucao: str) -> str:
    """Converte a resolução informada (ex.: 720 ou 1080) no seletor de download utilizado pelo yt-dlp."""
    try:
        altura = int(resolucao)
    except (ValueError, TypeError):
        altura = 720
    return (
        f"bestvideo[height<={altura}][ext=mp4]+bestaudio[ext=m4a]/"
        f"best[height<={altura}][ext=mp4]/best"
    )


def _extrair_video_id_youtube(url: str) -> Optional[str]:
    """Extrai, quando possível, o ID de um vídeo do YouTube a partir de uma URL."""
    url_analisada = urlparse(url)
    host = (url_analisada.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]

    if host == "youtu.be":
        video_id = url_analisada.path.lstrip("/").split("/", 1)[0]
        return video_id or None

    if "youtube.com" in host:
        if url_analisada.path.startswith("/watch"):
            parametros = parse_qs(url_analisada.query)
            video_id = parametros.get("v", [""])[0]
            return video_id or None
        resultado = re.search(r"/(?:shorts|embed|live)/([^/?#&]+)", url_analisada.path)
        if resultado:
            return resultado.group(1)

    return None


def _resolver_caminho_local(origem: str) -> Optional[str]:
    """Resolve e valida um caminho local, retornando seu caminho absoluto quando existir."""
    origem_analisada = urlparse(origem)
    if origem_analisada.scheme == "file":
        caminho_decodificado = unquote(origem_analisada.path)
        if origem_analisada.netloc and origem_analisada.netloc not in ("", "localhost"):
            caminho_decodificado = f"//{origem_analisada.netloc}{caminho_decodificado}"
        caminho = Path(caminho_decodificado).expanduser()
        if caminho.exists() and caminho.is_file():
            return str(caminho.resolve())
        raise RuntimeError(f"O arquivo informado na URL não existe: {origem}")

    if origem_analisada.scheme in {"http", "https"}:
        return None

    caminho = Path(origem).expanduser()
    if caminho.exists() and caminho.is_file():
        return str(caminho.resolve())

    if any(separador in origem for separador in (os.sep, "/")) or origem.startswith("~") or origem.startswith("."):
        raise RuntimeError(f"O caminho informado não existe: {origem}")

    return None


def _obter_download_existente(diretorio_saida: str, video_id: str) -> Optional[str]:
    """Retorna o caminho de um vídeo já baixado, caso ele exista."""
    for extensao in (".mp4", ".mkv", ".webm"):
        caminho = os.path.join(diretorio_saida, f"video_{video_id}{extensao}")
        if os.path.exists(caminho):
            return caminho
    return None


def download_youtube(video_url: str, resolucao: str = "720", diretorio_saida: Optional[str] = None) -> str:
    """Baixa um vídeo do YouTube ou retorna o caminho de um arquivo já existente."""


    # 1 - Verifica se a URL é um caminho de arquivo local.
    arquivo_existente = _resolver_caminho_local(video_url)
    if arquivo_existente:
        print(f"[Download] Utilizando arquivo existente: {arquivo_existente}", flush=True)
        return arquivo_existente


    # 2 - Verifica se o vídeo já foi baixado anteriormente.
    video_id = _extrair_video_id_youtube(video_url)
    if video_id:
        diretorio_saida = diretorio_saida or os.path.join(DIRETORIO_SAIDA, f"projeto_{video_id}")
        os.makedirs(diretorio_saida, exist_ok=True)
        download_existente =_obter_download_existente(diretorio_saida, video_id)
        if download_existente:
            print(f"[Download] Reutilizando download existente: {download_existente}", flush=True)
            return download_existente
    else:
        raise RuntimeError(f"Não foi possível extrair o ID do vídeo do YouTube a partir da URL: {video_url}")


    # 3 - Baixa o vídeo do YouTube.
    yt_dlp = _importar_yt_dlp()
    print(f"[Download] {video_url} @ {resolucao}p ? {diretorio_saida}/", flush=True)
    opcoes_yt_dlp = {
    "format": _obter_seletor_resolucao(resolucao),
    "outtmpl": os.path.join(diretorio_saida, "video_%(id)s.%(ext)s"),
    "merge_output_format": "mp4",
    "quiet": True,
    "no_warnings": True,
    "noprogress": True,
    }

    with yt_dlp.YoutubeDL(opcoes_yt_dlp) as ydl:
        informacoes = ydl.extract_info(video_url, download=True)
        download_atual = ydl.prepare_filename(informacoes)
        # Após a mesclagem, a extensão do arquivo pode ser alterada.
        if not os.path.exists(download_atual):
            nome_arquivo, _ = os.path.splitext(download_atual)
            for extensao in (".mp4", ".mkv", ".webm"):
                if os.path.exists(nome_arquivo + extensao):
                    download_atual = nome_arquivo + extensao
                    break
    print(f"[Download] Download concluído: {download_atual}", flush=True)
    return download_atual