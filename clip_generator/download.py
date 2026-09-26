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

from .configuracao import DIRETORIO_SAIDA


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


def _preparar_nome_para_diretorio(nome: str) -> str:
    """
    Prepara um texto para ser utilizado como nome de pasta.
    Remove caracteres inválidos em nomes de arquivos e pastas
    e ajusta espaços e caracteres problemáticos.

    Args:
        nome: Nome original da pasta.

    Returns:
        Nome preparado para utilização como pasta.
    """

    # Remove caracteres inválidos para Windows/Linux/macOS.
    nome = re.sub(r'[<>:"/\\|?*]', '', nome)

    # Remove caracteres de controle.
    nome = re.sub(r'[\x00-\x1F]', '', nome)

    # Substitui sequências de espaços por um único espaço.
    nome = re.sub(r'\s+', ' ', nome)

    # Remove espaços e pontos no final.
    nome = nome.rstrip(' .')

    return nome


def _obter_diretorio_projeto(diretorio_saida: str, video_id: str) -> Optional[str]:
    """
    Localiza o diretório de um projeto pelo ID do vídeo.
    O diretório deve começar com 'projeto_{id_video}'.
    
    Args:
        diretorio_saida: Diretório onde os projetos são armazenados.
        video_id: ID do vídeo do YouTube.

    Returns:
        Caminho do diretório encontrado ou None caso não exista.
    """

    prefixo = f"projeto_{video_id}"

    for caminho in Path(diretorio_saida).iterdir():
        if not caminho.is_dir():
            continue

        if caminho.name == prefixo or caminho.name.startswith(f"{prefixo} - "):
            return str(caminho)

    return None


def _preparar_diretorio_projeto(diretorio_saida: str, video_id: str, titulo: str) -> str:
    """
    Cria ou atualiza o diretório de um projeto do YouTube.

    O diretório é identificado pelo ID do vídeo e recebe o título
    atual do vídeo em seu nome.

    Args:
        diretorio_saida: Diretório onde os projetos são armazenados.
        video_id: ID do vídeo do YouTube.
        titulo: Título atual do vídeo.

    Returns:
        Caminho do diretório do projeto.
    """

    titulo = _preparar_nome_para_diretorio(titulo)

    nome_diretorio = f"projeto_{video_id} - {titulo}"
    novo_diretorio = Path(diretorio_saida) / nome_diretorio

    diretorio_existente = _obter_diretorio_projeto(diretorio_saida, video_id)

    # Projeto ainda não existe.
    if not diretorio_existente:
        novo_diretorio.mkdir(parents=True, exist_ok=True)
        return str(novo_diretorio)

    diretorio_existente = Path(diretorio_existente)

    # O diretório já possui o nome correto.
    if diretorio_existente == novo_diretorio:
        return str(diretorio_existente)

    # O diretório existe, mas o título está desatualizado.
    diretorio_existente.rename(novo_diretorio)

    return str(novo_diretorio)


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


    # 1. Vídeo local
    arquivo_existente = _resolver_caminho_local(video_url)
    if arquivo_existente:
        print(f"[Download] Utilizando arquivo existente: {arquivo_existente}", flush=True)
        return arquivo_existente


    # 2. Vídeo do YouTube
    yt_dlp, DownloadError = _importar_yt_dlp()

    video_id = _extrair_video_id_youtube(video_url)
    if not video_id:
        raise RuntimeError(f"URL do YouTube inválida: {video_url}")

    diretorio_saida = diretorio_saida or DIRETORIO_SAIDA

    # Localiza o projeto existente pelo ID do vídeo.
    diretorio_projeto = _obter_diretorio_projeto(diretorio_saida, video_id)

    # Se o projeto ainda não existe, cria um diretório provisório utilizando somente o ID até obter as informações do vídeo.
    if not diretorio_projeto:
        diretorio_projeto = os.path.join(diretorio_saida, f"projeto_{video_id}")
        os.makedirs(diretorio_projeto, exist_ok=True)

    # Verifica se o vídeo já foi baixado.
    download_existente = _obter_download_existente(diretorio_projeto, video_id)

    # Vídeo já existe: consulta apenas as informações do vídeo para atualizar o nome do projeto.
    if download_existente:
        print(f"[Download] Reutilizando download existente: {download_existente}", flush=True)

        opcoes_yt_dlp = {
            "quiet": True, 
            "no_warnings": True
            }

        try:
            with yt_dlp.YoutubeDL(opcoes_yt_dlp) as ydl:
                informacoes = ydl.extract_info(video_url, download=False)

            titulo_video = (f"{informacoes['channel']} ~.~ {informacoes['title']}")
            diretorio_projeto = _preparar_diretorio_projeto(diretorio_saida, video_id, titulo_video)

            # O caminho do vídeo muda caso a pasta tenha sido renomeada.
            caminho_video = os.path.join(diretorio_projeto, os.path.basename(download_existente))

        except DownloadError as e:
            print(f"[Download] Falha ao obter informações atuais do vídeo ({e}). Utilizando a pasta existente como está.", flush=True)

    # Caso ainda não exista, faz o download.
    else:
        print(f"[Download] {video_url} @ {resolucao}p → {diretorio_projeto}/", flush=True)

        opcoes_yt_dlp = {
            "format": _obter_seletor_resolucao(resolucao),
            "outtmpl": os.path.join(diretorio_projeto, "video_%(id)s.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "noprogress": False,
        }

        try:
            with yt_dlp.YoutubeDL(opcoes_yt_dlp) as ydl:
                informacoes = ydl.extract_info(video_url, download=True)
                titulo_video = (f"{informacoes['channel']} ~.~ {informacoes['title']}")
                download_atual = ydl.prepare_filename(informacoes)
                # Após a mesclagem, a extensão do arquivo pode ser alterada.
                if not os.path.exists(download_atual):
                    nome_arquivo, _ = os.path.splitext(download_atual)
                    for extensao in (".mp4", ".mkv", ".webm"):
                        if os.path.exists(nome_arquivo + extensao):
                            download_atual = nome_arquivo + extensao
                            break

        except DownloadError as e:
            raise RuntimeError(f"Falha ao baixar o vídeo do YouTube: {e}") from e

        # Atualiza o nome do projeto após obter as informações do vídeo.
        diretorio_projeto_novo = _preparar_diretorio_projeto(diretorio_saida, video_id, titulo_video)

        # O caminho do vídeo muda caso a pasta tenha sido renomeada.
        caminho_video = os.path.join(diretorio_projeto_novo, os.path.basename(download_atual))

        print(f"[Download] Download concluído: {caminho_video}", flush=True)

    # 3. Este ponto só é alcançado para vídeos do YouTube.
    caminho_link = _salvar_link_video(video_url, caminho_video)

    print(f"[Download] Link para o vídeo original salvo em: {caminho_link}", flush=True)

    return caminho_video
