"""Camada de comunicação com provedores de LLM."""

from .configuracao import (
    PROVEDOR_LLM,
    MODELO_GEMINI,
    MODELO_OPENAI,
    obter_chave_gemini,
    obter_chave_openai,
)


def _chamar_openai(prompt: str) -> str:
    """Envia um prompt para a API da OpenAI e retorna a resposta."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "A biblioteca 'openai' é necessária para utilizar a OpenAI. Instale com:\n"
            "    pip install -r requirements.txt"
        ) from e

    cliente = OpenAI(api_key=obter_chave_openai())
    resposta = cliente.chat.completions.create(
        model=MODELO_OPENAI,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return resposta.choices[0].message.content or ""


def _chamar_gemini(prompt: str) -> str:
    """Envia um prompt para a API do Gemini e retorna a resposta."""
    try:
        from google import genai  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "A biblioteca 'google-genai' é necessária para utilizar o Gemini. Instale com:\n"
            "    pip install -r requirements.txt"
        ) from e

    cliente = genai.Client(api_key=obter_chave_gemini())
    resposta = cliente.models.generate_content(
        model=MODELO_GEMINI,
        contents=prompt,
        config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "max_output_tokens": 8192,
        },
    )
    return resposta.text or ""


def chamar_llm(prompt: str) -> str:
    """Envia um prompt para o provedor de LLM configurado e retorna a resposta."""
    provedor = PROVEDOR_LLM.strip().lower()
    if provedor == "openai":
        return _chamar_openai(prompt)
    if provedor == "gemini":
        return _chamar_gemini(prompt)
    raise RuntimeError(
        f"Provedor de LLM desconhecido: {provedor!r}. Utilize 'openai' ou 'gemini'."
    )