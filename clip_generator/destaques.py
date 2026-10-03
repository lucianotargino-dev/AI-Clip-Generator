"""Identifica os destaques com maior potencial de viralização em uma transcri

Lógica adaptada do arquivo transcript_analysis/highlight_generator.py do ViralVadoo:

* detecção do tipo e da densidade do conteúdo;
* divisão de vídeos longos em blocos com sobreposição;
* definição dos critérios de viralização no prompt;
* remoção de destaques duplicados com base na pontuação e supressão de sobreposição.

A chamada à LLM é intercambiável por meio do argumento  funcao_llm , permitindo 
utilizar diferentes provedores de LLM sem alterar a lógica de análise dos 
destaques.
"""

import json
import re
from typing import Callable, Dict, List, Any, Optional

FuncaoLLM = Callable[[str], str]


PROMPT_TIPO_CONTEUDO = """Analise esta amostra da transcrição de um vídeo e classifique o tipo de conteúdo.
Escolha uma opção: podcast, entrevista, tutorial, aula, comentário, debate, vlog ou outro.
Estime também a densidade do conteúdo: baixa (principalmente conteúdo superficial/conversa), média ou alta (muitas informações/histórias).
Responda somente com JSON: {"tipo_conteudo": "...", "densidade": "..."}"""


CRITERIOS_VIRALIZACAO = """
Sinais de potencial de viralização a serem priorizados (em ordem de impacto):
1. MOMENTOS DE GANCHO — declarações que despertam curiosidade imediata ("O segredo é...", "Ninguém fala sobre...", "Eu estava completamente 
errado sobre...")
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


def _interpretar_json_flexivel(texto_bruto: str) -> Dict:
    """Remove formatação Markdown e interpreta uma resposta JSON."""
    texto = texto_bruto.strip()
    texto = re.sub(r"^```(?:json)?\s*", "", texto)
    texto = re.sub(r"\s*```$", "", texto)

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        inicio = texto.find("{")
        fim = texto.rfind("}")

        if inicio != -1 and fim != -1:
            return json.loads(texto[inicio:fim + 1])

        raise


def _converter_para_float(valor: Any, padrao: float = 0.0) -> float:
    try:
        return float(valor)
    except (TypeError, ValueError):
        return padrao


def _converter_para_int(valor: Any, padrao: int = 0) -> int:
    try:
        return int(float(valor))
    except (TypeError, ValueError):
        return padrao


def _validar_destaques(destaques_brutos: object, duracao: float) -> List[Dict]:
    """Normaliza a resposta da LLM para o formato esperado e ignora itens inválidos."""
    if not isinstance(destaques_brutos, list):
        return []

    fim_maximo = duracao if duracao > 0 else float("inf")
    destaques_limpos: List[Dict] = []
    for item in destaques_brutos:
        if not isinstance(item, dict):
            continue

        inicio = _converter_para_float(item.get("inicio"), padrao=-1.0)
        fim = _converter_para_float(item.get("fim"), padrao=-1.0)
        if inicio < 0 or fim <= inicio:
            continue

        if fim_maximo != float("inf"):
            inicio = min(inicio, fim_maximo)
            fim = min(fim, fim_maximo)
            if fim <= inicio:
                continue

        destaques_limpos.append(
            {
                "titulo": str(item.get("titulo") or "Destaque sem título").strip(),
                "inicio": inicio,
                "fim": fim,
                "pontuacao": max(0, min(100, _converter_para_int(item.get("pontuacao"), padrao=0))),
                "frase_gancho": str(item.get("frase_gancho") or "").strip(),
                "motivo_viralizacao": str(item.get("motivo_viralizacao") or "").strip(),
            }
        )

    return destaques_limpos


def _detectar_tipo_conteudo(transcricao: Dict, funcao_llm: FuncaoLLM) -> Dict[str, str]:
    segmentos = transcricao.get("segmentos", [])
    amostra = " ".join(segmento["texto"] for segmento in segmentos[:25])[:3000]
    prompt = f"{PROMPT_TIPO_CONTEUDO}\n\nAmostra de transcrição:\n{amostra}"
    try:
        return _interpretar_json_flexivel(funcao_llm(prompt))
    except Exception:
        return {"tipo_conteudo": "outro", "densidade": "media"}


def _construir_texto_transcricao(transcricao: Dict) -> str:
    segmentos = transcricao.get("segmentos", [])
    return "\n".join(f"[{segmento['inicio']:.1f}s] {segmento['texto'].strip()}" for segmento in segmentos)


def _dividir_transcricao_em_blocos(transcricao: Dict) -> List[Dict]:
    segmentos = transcricao.get("segmentos", [])
    duracao = transcricao.get("duracao", segmentos[-1]["fim"] if segmentos else 0)
    blocos = []
    inicio = 0

    while inicio < duracao:
        fim = min(inicio + TAMANHO_BLOCO_SEGUNDOS, duracao)

        segmentos_bloco = [
            segmento for segmento in segmentos
            if segmento["inicio"] >= inicio and segmento["fim"] <= fim + SOBREPOSICAO_BLOCO_SEGUNDOS
        ]

        if segmentos_bloco:
            bloco = dict(transcricao)
            bloco["segmentos"] = segmentos_bloco
            bloco["duracao"] = fim - inicio
            bloco["_deslocamento"] = inicio
            blocos.append(bloco)

        inicio += TAMANHO_BLOCO_SEGUNDOS - SOBREPOSICAO_BLOCO_SEGUNDOS

    return blocos


def _gerar_destaques_com_llm(
    texto_transcricao: str,
    informacoes_conteudo: Dict,
    duracao: float,
    quantidade_clipes: int,
    funcao_llm: FuncaoLLM,
    eh_bloco: bool = False    
) -> Dict:
    
    alvo = max(quantidade_clipes * 2, 5)
    maximo_natural = max(2 if eh_bloco else 3, int(duracao / 90))
    minimo_clipes = min(alvo, maximo_natural, 8)
    prompt_sistema = PROMPT_SISTEMA_DESTAQUES.format(
        criterios_viralizacao=CRITERIOS_VIRALIZACAO,
        tipo_conteudo=informacoes_conteudo.get("tipo_conteudo", "outro"),
        densidade=informacoes_conteudo.get("densidade", "media"),
        instrucao_quantidade_clipes=f"Gere pelo menos {minimo_clipes} destaques",
    )
    prompt_base = f"{prompt_sistema}\n\nTranscrição:\n{texto_transcricao}"
    prompt = prompt_base
    ultimo_erro = "desconhecido"

    for tentativa in range(1, MAXIMO_TENTATIVAS_DESTAQUES + 1):
        resposta_bruta = funcao_llm(prompt)

        try:
            resposta_interpretada = _interpretar_json_flexivel(resposta_bruta)
            destaques = _validar_destaques(resposta_interpretada.get("destaques"), duracao=duracao)
            if destaques:
                return {"destaques": destaques}
            ultimo_erro = "nenhum destaque válido na resposta"
        except Exception as erro:
            ultimo_erro = str(erro)

        if tentativa < MAXIMO_TENTATIVAS_DESTAQUES:
            print(
                f"[destaques] resposta inválida na tentativa "
                f"{tentativa}/{MAXIMO_TENTATIVAS_DESTAQUES}; tentando novamente",
                flush=True,
            )
            prompt = (
                prompt_base
                + "\n\nIMPORTANTE: Retorne SOMENTE um JSON válido com uma matriz 'destaques' no nível superior."
                + " Cada item deve conter: titulo, inicio, fim, pontuacao, frase_gancho, motivo_viralizacao."
                + " Não utilize blocos Markdown nem comentários."
            )

    raise RuntimeError(
        f"O gerador de destaques produziu uma resposta inválida após {MAXIMO_TENTATIVAS_DESTAQUES} tentativas: {ultimo_erro}"
    )


def _remover_destaques_duplicados(destaques: List[Dict]) -> List[Dict]:
    """Remove destaques com mais de 50% de sobreposição com um destaque melhor avaliado."""
    destaques = sorted(destaques, key=lambda destaque: int(destaque.get("pontuacao", 0)), reverse=True)
    mantidos: List[Dict] = []
    for destaque in destaques:
        inicio_destaque = float(destaque["inicio"])
        fim_destaque = float(destaque["fim"])
        duracao_destaque = fim_destaque - inicio_destaque
        possui_sobreposicao = False
        for mantido in mantidos:
            maior_inicio = max(inicio_destaque, float(mantido["inicio"]))
            menor_fim = min(fim_destaque, float(mantido["fim"]))
            sobreposicao = menor_fim - maior_inicio
            if (sobreposicao > 0 and sobreposicao > 0.5 * duracao_destaque):
                possui_sobreposicao = True
                break
        if not possui_sobreposicao:
            mantidos.append(destaque)
    return mantidos


def obter_destaques(
    transcricao: Dict,
    funcao_llm: FuncaoLLM,
    quantidade_clipes: int = 3
) -> Dict:
    """Função principal que retorna os destaques ordenados por pontuação."""
    duracao = transcricao.get("duracao", 0)
    informacoes_conteudo = _detectar_tipo_conteudo(transcricao, funcao_llm=funcao_llm)
    print(
        f"[destaques] conteúdo={informacoes_conteudo.get('tipo_conteudo')} "
        f"densidade={informacoes_conteudo.get('densidade')} "
        f"duração={duracao:.0f}s",
        flush=True,
    )
    if duracao >= LIMIAR_VIDEO_LONGO_SEGUNDOS:
        blocos = _dividir_transcricao_em_blocos(transcricao)
        print(f"[destaques] vídeo longo — dividido em {len(blocos)} blocos", flush=True)
        todos_destaques: List[Dict] = []
        for indice, bloco in enumerate(blocos):
            deslocamento = bloco.get("_deslocamento", 0)
            texto = _construir_texto_transcricao(bloco)
            print(
                f"[destaques] bloco {indice + 1}/{len(blocos)} (deslocamento {deslocamento:.0f}s)", flush=True)
            resultado = _gerar_destaques_com_llm(texto, informacoes_conteudo, bloco["duracao"], quantidade_clipes, eh_bloco=True, funcao_llm=funcao_llm)
            for destaque in resultado.get("destaques", []):
                destaque["inicio"] = float(destaque["inicio"]) + deslocamento
                destaque["fim"] = float(destaque["fim"]) + deslocamento
                todos_destaques.append(destaque)
        destaques = _remover_destaques_duplicados(todos_destaques)
    else:
        texto = _construir_texto_transcricao(transcricao)
        resultado = _gerar_destaques_com_llm(texto, informacoes_conteudo, duracao, quantidade_clipes, funcao_llm=funcao_llm)
        destaques = _remover_destaques_duplicados(resultado.get("destaques", []))
    return {"destaques": destaques}