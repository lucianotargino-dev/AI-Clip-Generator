"""Identifica os destaques com maior potencial de viralização em uma transcrição.

Lógica adaptada do arquivo transcript_analysis/highlight_generator.py do ViralVadoo:

* detecção do tipo e da densidade do conteúdo;
* divisão de vídeos longos em blocos com sobreposição;
* definição dos critérios de viralização no prompt;
* remoção de destaques duplicados com base na pontuação e supressão de sobreposição.

A chamada à LLM é intercambiável por meio do argumento 'funcao_llm', permitindo
utilizar diferentes provedores de LLM sem alterar a lógica de análise dos
destaques.
"""

import json
import re
from typing import Callable, Dict, List, Any, Optional

from .configuracao import (
        PROMPT_TIPO_CONTEUDO,
        CRITERIOS_VIRALIZACAO,
        PROMPT_SISTEMA_DESTAQUES,

        TAMANHO_BLOCO_SEGUNDOS,
        LIMIAR_VIDEO_LONGO_SEGUNDOS,
        SOBREPOSICAO_BLOCO_SEGUNDOS,
        TEMPO_LIMITE_CHAMADA_LLM_SEGUNDOS,
        MAXIMO_TENTATIVAS_DESTAQUES,
        )

FuncaoLLM = Callable[[str], str]


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
            if segmento["inicio"] >= inicio
            and segmento["fim"] <= fim + SOBREPOSICAO_BLOCO_SEGUNDOS
        ]

        if segmentos_bloco:
            segmentos_bloco_relativos = []

            for segmento in segmentos_bloco:
                segmento_relativo = {
                    **segmento,
                    "inicio": segmento["inicio"] - inicio,
                    "fim": segmento["fim"] - inicio,
                }

                segmentos_bloco_relativos.append(segmento_relativo)

            bloco = dict(transcricao)
            bloco["segmentos"] = segmentos_bloco_relativos
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
    """Gera destaques usando a LLM com tentativas de recuperação em caso de erro."""
    alvo = max(quantidade_clipes * 2, 5)
    maximo_natural = max(2 if eh_bloco else 3, int(duracao / 90))
    minimo_clipes = min(alvo, maximo_natural, 8)
    prompt_sistema = PROMPT_SISTEMA_DESTAQUES.format(
        criterios_viralizacao=CRITERIOS_VIRALIZACAO,
        tipo_conteudo=informacoes_conteudo.get("tipo_conteudo", "outro"),
        densidade=informacoes_conteudo.get("densidade", "média"),
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
            print(f"[Destaques] resposta inválida na tentativa {tentativa}/{MAXIMO_TENTATIVAS_DESTAQUES}; tentando novamente", flush=True)
            prompt = (
                prompt_base
                + "\n\nIMPORTANTE: Retorne SOMENTE um JSON válido com uma matriz 'destaques' no nível superior."
                + " Cada item deve conter: titulo, inicio, fim, pontuacao, frase_gancho, motivo_viralizacao."
                + " Não utilize blocos Markdown nem comentários."
            )

    raise RuntimeError(f"O gerador de destaques produziu uma resposta inválida após {MAXIMO_TENTATIVAS_DESTAQUES} tentativas: {ultimo_erro}")


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
            if sobreposicao > 0 and sobreposicao > 0.5 * duracao_destaque:
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
    print(f"[Destaques] conteúdo={informacoes_conteudo.get('tipo_conteudo')} densidade={informacoes_conteudo.get('densidade')} duração={duracao:.0f}s", flush=True)

    if duracao >= LIMIAR_VIDEO_LONGO_SEGUNDOS:
        blocos = _dividir_transcricao_em_blocos(transcricao)
        print(f"[Destaques] vídeo longo — dividido em {len(blocos)} blocos", flush=True)
        todos_destaques: List[Dict] = []
        for indice, bloco in enumerate(blocos):
            deslocamento = bloco.get("_deslocamento", 0)
            texto = _construir_texto_transcricao(bloco)
            print(f"[Destaques] bloco {indice + 1}/{len(blocos)} (deslocamento {deslocamento:.0f}s)", flush=True)
            resultado = _gerar_destaques_com_llm(texto, informacoes_conteudo, bloco["duracao"], quantidade_clipes, funcao_llm, eh_bloco=True)
            for destaque in resultado.get("destaques", []):
                destaque["inicio"] = float(destaque["inicio"]) + deslocamento
                destaque["fim"] = float(destaque["fim"]) + deslocamento
                todos_destaques.append(destaque)
        destaques = _remover_destaques_duplicados(todos_destaques)
    else:
        texto = _construir_texto_transcricao(transcricao)
        resultado = _gerar_destaques_com_llm(texto, informacoes_conteudo, duracao, quantidade_clipes, funcao_llm)
        destaques = _remover_destaques_duplicados(resultado.get("destaques", []))
    return {"destaques": destaques}