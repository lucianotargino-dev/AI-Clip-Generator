import os

from dotenv import load_dotenv

load_dotenv()

DIRETORIO_SAIDA = "saida"


# WHISPER
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


# LLM
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


# Prompts e configurações de destaques
PROMPT_TIPO_CONTEUDO = """Analise esta amostra da transcrição de um vídeo e classifique o tipo de conteúdo.
Escolha uma opção: podcast, entrevista, tutorial, aula, comentário, debate, vlog ou outro.
Estime também a densidade do conteúdo: baixa (principalmente conteúdo superficial/conversa), média ou alta (muitas informações/histórias).
Responda somente com JSON: {"tipo_conteudo": "...", "densidade": "..."}"""

CRITERIOS_VIRALIZACAO = """
Sinais de potencial de viralização a serem priorizados (em ordem de impacto):

1. MOMENTOS DE GANCHO — declarações que despertam curiosidade imediata ("O segredo é...", "Ninguém fala sobre...", "Eu estava completamente errado sobre...")
2. PICOS EMOCIONAIS — surpresa, risada, raiva, vulnerabilidade ou empolgação genuínas; reações espontâneas e não ensaiadas
3. OPINIÕES IMPACTANTES — declarações fortes, polêmicas ou contraintuitivas que provoquem concordância ou discordância
4. MOMENTOS DE REVELAÇÃO — fatos, estatísticas ou confissões surpreendentes que mudam a forma como o espectador enxerga o assunto
5. CONFLITO/TENSÃO — discordância, contestação ou um problema sendo enfrentado diretamente
6. FRASES DE EFEITO — uma frase que funcione como uma citação independente
7. PICO DA HISTÓRIA — o clímax ou a reviravolta de uma história; o momento de conclusão ou recompensa
8. VALOR PRÁTICO — uma dica, técnica ou informação que o espectador possa aplicar imediatamente
"""

PROMPT_SISTEMA_DESTAQUES = """Você é um editor especialista em vídeos curtos que estudou milhares de cortes virais no TikTok, Instagram Reels e YouTube Shorts. Você sabe exatamente o que faz os espectadores pararem de rolar a tela, assistirem até o final e compartilharem um vídeo.

{criterios_viralizacao}

Tipo de conteúdo: {tipo_conteudo} | Densidade: {densidade}

Sua tarefa: identificar os destaques da transcrição com maior potencial de viralização.

Regras:

- Todo destaque deve começar com um GANCHO forte — uma frase que capture a atenção nos primeiros 3 segundos
- Duração ideal: 45–90 segundos. Seja mais curto (20–44s) somente para uma frase de efeito perfeita e independente. Seja mais longo (91–180s) somente quando uma história precisar de todo o contexto para funcionar
- Nunca corte no meio de uma frase ou raciocínio — cada corte deve ser completo e fazer sentido por si só
- Os cortes não devem apresentar sobreposição significativa entre si
- Atribua uma pontuação de 0–100 para o potencial de viralização (não para a qualidade geral)
- {instrucao_quantidade_clipes}
- Para cada destaque, identifique a melhor "frase_gancho" — a frase inicial que faria alguém parar de rolar a tela
- Explique em uma frase por que este corte tem potencial de viralização ("motivo_viralizacao")

Responda SOMENTE com JSON válido (sem Markdown e sem explicações):
{{"destaques":[{{"titulo":"string","inicio":float,"fim":float,"pontuacao":int,"frase_gancho":"string","motivo_viralizacao":"string"}}]}}"""

TAMANHO_BLOCO_SEGUNDOS = 1200       # blocos de 20 minutos para vídeos longos
LIMIAR_VIDEO_LONGO_SEGUNDOS = 1800  # divide vídeos com mais de 30 minutos
SOBREPOSICAO_BLOCO_SEGUNDOS = 60

TEMPO_LIMITE_CHAMADA_LLM_SEGUNDOS = 300  # limita as consultas à LLM a 5 minutos
MAXIMO_TENTATIVAS_DESTAQUES = 3
