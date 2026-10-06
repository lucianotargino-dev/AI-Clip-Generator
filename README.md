![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/status-in__development-yellow)

# 🎬 AI Clip Generator

Ferramenta em Python para geração automática de cortes verticais (Shorts/Reels/TikTok) a partir de vídeos longos utilizando Inteligência Artificial.

&gt; ⚠️ **Status do Projeto**: Work In Progress (Iniciando desenvolvimento / Envio progressivo de módulos)

---

## 🚀 Sobre o projeto

Este projeto tem como objetivo automatizar a identificação de momentos de destaque em vídeos e realizar o corte automático para o formato vertical (9:16), sem depender de plataformas de terceiros.

A ideia inicial é baseada e inspirada no projeto [AI-Youtube-Shorts-Generator](https://github.com/Anil-matcha/AI-Youtube-Shorts-Generator), servindo como ponto de partida para o estudo de uma pipeline completa de automação, processamento de mídia local e integração com LLMs.

Entre os principais objetivos desta evolução estão:

- **Eliminação do processamento por API** nas etapas de mídia, garantindo que o processamento do vídeo e a transcrição ocorram de forma totalmente local na máquina do usuário;
- **Aportuguesamento do projeto**, reestruturando e traduzindo o código-fonte (funções, variáveis, documentação e comentários) para o português, buscando maior clareza e padronização.
- **Modelo de Dados Próprio**: Isolamento da lógica de negócios em relação aos retornos brutos de bibliotecas e SDKs externos.

Atualmente, o projeto está em sua fase inicial de planejamento e estruturação.

---

## 🗂️ Estrutura do projeto

Neste primeiro momento, o repositório contém apenas os arquivos de configuração inicial:

```text
AI-Clip-Generator/
│
├── clip_generator/
│   ├── configuracao.py       # Módulo de configuração que contém as constantes do projeto
│   ├── destaques.py          # Módulo de escolha de momentos virais
│   ├── download.py           # Módulo de download de vídeos
│   ├── llm.py                # Módulo de conexão com LLMs
│   ├── reenquadramento.py    # Módulo de extração de trechos e reenquadramento
│   └── transcricao.py        # Módulo de transcrição de videos
│
├── .env.example
├── .gitignore                # Regra de ignorados do Git
├── README.md                 # Esta documentação
├── requirements.txt
└── LICENSE
```

---

## 🚧 Próximos Passos (Em Desenvolvimento)

O código-fonte será adicionado e refatorado gradativamente. O pipeline que será construído inclui:

[X] Módulo de Download: Obtenção de vídeos e organização em diretórios padronizados.

[X] Módulo de Transcrição: Processamento local de áudio utilizando faster-whisper.

[X] Módulo de LLM: Camada de conexão desacoplada com suporte inicial para Gemini e OpenAI (ChatGPT), e preparação para modelos locais (Ollama).

[X] Módulo de Destaques: Filtro inteligente de momentos virais por LLM para identificar trechos com potencial viral.

[ ] Módulo de Reenquadramento: Extração de trechos e reenquadramento dinâmico com base nos carimbos de tempo.

[ ] Módulo de Legendas: Geração e sobreposição de legendas animadas sincronizadas palavra por palavra.

---

## ⚠️ Importante

### 1. FFmpeg

O módulo de transcrição requer o **FFmpeg** instalado no sistema.

**No Windows (via Terminal/PowerShell):**

```powershell
winget install FFmpeg
```

**No Linux (Ubuntu/Debian):**

```powershell
sudo apt update && sudo apt install ffmpeg -y
```

**No macOS:**

```powershell
brew install ffmpeg
```

Nota: Após a instalação, feche e reabra o terminal/IDE para atualizar as variáveis de ambiente.

### 2. GPU

O módulo identifica e configura dinamicamente o dispositivo de execução (GPU vs. CPU):

- Aceleração por GPU (NVIDIA CUDA):
  - Requer placa de vídeo NVIDIA com drivers atualizados.

  - Requer as bibliotecas do cuDNN instaladas (especialmente a versão 8 ou superior compatível com **ctranslate2**/**faster-whisper**).

  - No Python, usa precisão **float16** quando em GPU para maior desempenho.

- Fallback Automático para CPU:
  - Caso não haja suporte à CUDA/cuDNN configurado ou ocorra alguma incompatibilidade de hardware, o sistema altera automaticamente a execução para **cpu** utilizando a computação em **int8** ou **float32**

### 3. Whisper

- No primeiro uso de um modelo específico do Whisper (ex: **small**, **medium**, **large-v3**), a biblioteca **faster-whisper** baixará os pesos do modelo diretamente do Hugging Face.

- A primeira execução requer conexão à internet para o download dos pesos e espaço em disco disponível no diretório padrão do Hugging Face (**~/.cache/huggingface/hub/**).

### 4. Modelo de dados da transcrição

Foi decidido na arquitetura do projeto que o mesmo não utilizará diretamente a estrutura nativa do faster-whisper, mas sim um formato proprietário padronizado em português.

```json
{
  "duracao": 120.5,
  "segmentos": [
    {
      "inicio": 0.0,
      "fim": 4.5,
      "texto": "Texto do trecho do vídeo...",
      "palavras": [
        {
          "inicio": 0.0,
          "fim": 1.0,
          "palavra": "Texto"
        },
        {
          "inicio": 1.0,
          "fim": 1.5,
          "palavra": "do"
        },
        {
          "inicio": 1.5,
          "fim": 2.5,
          "palavra": "trecho"
        },
        {
          "inicio": 2.5,
          "fim": 3.0,
          "palavra": "do"
        },
        {
          "inicio": 3.0,
          "fim": 4.0,
          "palavra": "vídeo..."
        }
      ]
    }
  ]
}
```

---

## 👨‍💻 Créditos e Referências

Projeto inspirado na estrutura base de Anil-matcha/AI-Youtube-Shorts-Generator.

---

## 📄 Licença

Este projeto está licenciado sob a Licença MIT.
