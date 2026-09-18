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

Atualmente, o projeto está em sua fase inicial de planejamento e estruturação.

---

## 🗂️ Estrutura do projeto

Neste primeiro momento, o repositório contém apenas os arquivos de configuração inicial:

```text
AI-Clip-Generator/
│
├── .gitignore               # Configurações de ignorar arquivos (herdado do projeto base)
├── README.md                # Esta documentação
└── LICENCE
```

---

## 🚧 Próximos Passos (Em Desenvolvimento)

O código-fonte será adicionado e refatorado gradativamente. O pipeline que será construído inclui:

[ ] Módulo de Download: Obtenção de vídeos e organização em diretórios padronizados.

[ ] Módulo de Transcrição: Processamento local de áudio utilizando faster-whisper.

[ ] Módulo de Destaques: Conexão com LLMs (como Gemini e ChatGPT) para identificar trechos com potencial viral.

[ ] Módulo de Edição: Recorte automático com base nos carimbos de tempo e inserção de legendas.

---

## 👨‍💻 Créditos e Referências

Projeto inspirado na estrutura base de Anil-matcha/AI-Youtube-Shorts-Generator.

---

## 📄 Licença

Este projeto está licenciado sob a Licença MIT.
