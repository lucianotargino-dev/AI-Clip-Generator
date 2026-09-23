"""
Implementação da obtenção de vídeos do YouTube.

Este módulo identifica arquivos locais, reutiliza downloads existentes e,
quando necessário, realiza o download do vídeo utilizando o yt-dlp,
retornando o caminho do arquivo para as próximas etapas do pipeline.
"""
import os
import re
import platform
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from typing import Optional

# from ..config import LOCAL_OUTPUT_DIR
DIRETORIO_SAIDA = "saida" # provisiorio


def _importar_yt_dlp():
    try:
        import yt_dlp  # type: ignore
        from yt_dlp.utils import DownloadError
    except ImportError as e:
        raise RuntimeError(
            "A biblioteca 'yt-dlp' é necessária para baixar vídeos do YouTube.\n"
            "Instale-a com:\n"
            "    pip install -r requirements.txt"
        ) from e

    return yt_dlp, DownloadError


def _obter_titulo_video(video_url: str) -> str:
    """Obtém o título atual de um vídeo do YouTube sem realizar o download."""
    yt_dlp, DownloadError = _importar_yt_dlp()
    opcoes_yt_dlp = {
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(opcoes_yt_dlp) as ydl:
            informacoes = ydl.extract_info(video_url, download=False)
        return informacoes["title"]
    
    except DownloadError as e:
        raise RuntimeError(f"Falha ao obter informações do vídeo do YouTube: {e}") from e
    

def _salvar_link_video(video_url: str, caminho_video: str) -> Optional[str]:
    """
    Cria um arquivo de link para o vídeo original, utilizando
    o formato nativo do sistema operacional.

    Windows -> .url
    Linux   -> .desktop
    macOS   -> .webloc

    Args:
        video_url: URL original do vídeo.
        caminho_video: Caminho do vídeo baixado.

    Returns:
        Caminho do arquivo de link criado.
    """

    sistema = platform.system()

    caminho = Path(caminho_video)
    nome_link = f"link_{caminho.stem.removeprefix('video_')}"

    if sistema == "Windows":
        caminho_link = caminho.with_name(f"{nome_link}.url")

        conteudo = (
            "[InternetShortcut]\n"
            f"URL={video_url}\n"
        )

    elif sistema == "Linux":
        caminho_link = caminho.with_name(f"{nome_link}.desktop")

        conteudo = (
            "[Desktop Entry]\n"
            "Type=Link\n"
            f"URL={video_url}\n"
        )

    elif sistema == "Darwin":
        caminho_link = caminho.with_name(f"{nome_link}.webloc")

        conteudo = f"""<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
        "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>URL</key>
            <string>{video_url}</string>
        </dict>
        </plist>
        """

    else:
        raise OSError(
            f"Sistema operacional não suportado: {sistema}"
        )

    caminho_link.write_text(conteudo, encoding="utf-8")

    if caminho_link.exists():
        return str(caminho_link)
    
    return None


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

    
    # A partir daqui, sabemos que estamos tratando de um vídeo do YouTube
    video_id = _extrair_video_id_youtube(video_url)
    if not video_id:
        raise RuntimeError(f"URL do YouTube inválida: {video_url}")

    diretorio_saida = diretorio_saida or os.path.join(DIRETORIO_SAIDA, f"projeto_{video_id}")
    os.makedirs(diretorio_saida, exist_ok=True)

    # 2 - Verifica se o vídeo já foi baixado anteriormente.
    download_existente = _obter_download_existente(diretorio_saida, video_id)
    if download_existente:
        caminho_video = download_existente
        print(f"[Download] Reutilizando download existente: {download_existente}", flush=True)


    # 3 - Baixa o vídeo do YouTube.
    else:
        print(f"[Download] {video_url} @ {resolucao}p → "f"{diretorio_saida}/", flush=True)
        opcoes_yt_dlp = {
        "format": _obter_seletor_resolucao(resolucao),
        "outtmpl": os.path.join(diretorio_saida, "video_%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        }

        try:
            yt_dlp, DownloadError = _importar_yt_dlp()
            with yt_dlp.YoutubeDL(opcoes_yt_dlp) as ydl:
                informacoes = ydl.extract_info(video_url, download=True)
                download_atual = ydl.prepare_filename(informacoes)
                # Após a mesclagem, a extensão pode ser alterada.
                if not os.path.exists(download_atual):
                    nome_arquivo, _ = os.path.splitext(download_atual)
                    for extensao in (".mp4", ".mkv", ".webm"):
                        caminho_final = nome_arquivo + extensao
                        if os.path.exists(caminho_final):
                            caminho_video = download_atual = caminho_final
                            break

        except DownloadError as e:
            raise RuntimeError(f"Falha ao baixar o vídeo do YouTube: {e}") from e

        print(f"[Download] Download concluído: {caminho_video}", flush=True)

    # Vídeo veio do YouTube em ambos os casos:
    # - download existente
    # - download atual
    caminho_link = _salvar_link_video(video_url, caminho_video)
    print(f"[Download] Link para o vídeo original salvo em {caminho_link}",flush=True)
    return caminho_video