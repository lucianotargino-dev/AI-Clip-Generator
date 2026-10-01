import os

from dotenv import load_dotenv

load_dotenv()

DIRETORIO_SAIDA = "saida"

#WHISPER
DISPOSITIVO_WHISPER = "auto"
MODELO_WHISPER = "base"
FILTRO_WHISPER_VAD = False
PARAMETROS_WHISPER_VAD = {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": float("inf"),
        "min_silence_duration_ms": 2000,
        "speech_pad_ms": 400,
    }

#LLM
PROVEDOR_LLM = str(os.getenv("PROVEDOR_LLM"))
CHAVE_DE_API_OPENAI = os.getenv("CHAVE_DE_API_OPENAI")
MODELO_OPENAI = os.getenv("MODELO_OPENAI", "gpt-4o-mini")
CHAVE_DE_API_GEMINI = os.getenv("CHAVE_DE_API_GEMINI")
MODELO_GEMINI = os.getenv("MODELO_GEMINI", "gemini-3.5-flash-lite")


def obter_chave_openai() -> str:
    if not CHAVE_DE_API_OPENAI:
        raise RuntimeError(
            "A chave de API da OpenAI não foi configurada. "
            "Adicione a variável CHAVE_DE_API_OPENAI ao seu arquivo .env ou defina-a nas variáveis de ambiente."
        )
    return CHAVE_DE_API_OPENAI


def obter_chave_gemini() -> str:
    if not CHAVE_DE_API_GEMINI:
        raise RuntimeError(
            "A chave de API do Gemini não foi configurada. "
            "Adicione a variável CHAVE_DE_API_GEMINI ao seu arquivo .env ou defina-a nas variáveis de ambiente."
        )
    return CHAVE_DE_API_GEMINI