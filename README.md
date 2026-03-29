# Agente Analista de Dados
- Author: Prof. Barbosa  
- Contact: infobarbosa@gmail.com  
- Github: [infobarbosa](https://github.com/infobarbosa)


## Introdução

A Inteligência Artificial (IA) Generativa é um ramo da inteligência artificial focado na **criação de novos conteúdos originais**, em vez de apenas analisar ou agir sobre dados existentes. Esses conteúdos podem incluir texto, imagens, música, áudio, vídeos e até mesmo código de software.

O objetivo desse laboratório é implementar um agente simples de análise de dados usando Python e a API do Gemini.

---

## Plano de Construção: Agente Analista de Dados

Vamos dividir o projeto em 5 fases:

### Passo 1: Setup
O objetivo aqui é preparar o ambiente, garantindo que as dependências corretas estejam instaladas e que o código consiga "conversar" com a API do Gemini.
* **1.1.** Criação da estrutura de pastas no diretório `./data-eng-data-analisys-agent`.
* **1.2.** Configuração do ambiente virtual Python (`venv`).
* **1.3.** Instalação das bibliotecas core: `langgraph`, `langchain-google-genai` (para o Gemini), `pandas` e `python-dotenv`.
* **1.4.** Emissão e configuração segura da chave de API do Google Gemini (`.env`).

### Passo 2: O "Hello World" do LangGraph
Antes de lidar com arquivos e dados reais, precisamos garantir que a topologia lógica do LangGraph está funcionando.
* **2.1.** Definição do `AgentState` (memória de curto prazo do nosso agente).
* **2.2.** Criação de um Grafo mínimo com apenas um nó.
* **2.3.** Invocação do modelo Gemini 1.5 (Pro ou Flash) apenas para confirmar que a requisição de rede bate no Google e volta com uma resposta válida.

### Passo 3: As tools
O Gemini não consegue abrir um arquivo zip na sua máquina sozinho. Nesta fase, vamos usar suas habilidades de back-end para escrever as funções Python que farão o trabalho pesado, e então empacotá-las como `Tools` que o agente pode usar.
* **3.1. Tool de Extração:** Uma função para descompactar `.zip` ou ler `.csv` diretamente.
* **3.2. Tool de Análise (Pandas):** Uma função que recebe o caminho de um CSV, carrega no Pandas e retorna um resumo estatístico (tipos de dados, nulos, distribuições básicas).
* **3.3. Tool de Notificação:** Um mock simples que simula o envio de um e-mail com o relatório final (usaremos `print` ou um arquivo `.txt` de log inicialmente para não perder tempo configurando SMTP agora).

### Passo 4: Orquestração do Workflow
Aqui é onde a mágica (que não é mágica) acontece. Vamos unir as Tools construídas na Fase 3 com o Grafo iniciado na Fase 2.
* **4.1.** Criação do nó de Roteamento: Ensinar o LangGraph a entrar em um loop onde ele decide usar uma ferramenta, observa o resultado, e decide se precisa usar outra.
* **4.2.** Definição do "System Prompt" do Agente: As instruções claras de como ele deve se comportar como um analista de dados exploratório (ex: "Sempre verifique se há múltiplas entidades em um arquivo antes de gerar o relatório").

### Passo 5: Teste end-to-end
A validação final do sistema autônomo.
* **5.1.** Criação de um arquivo `.zip` contendo dados "sujos" (mock data) simulando uma extração de banco de dados.
* **5.2.** Execução do script principal apontando para este arquivo.
* **5.3.** Análise dos logs (ver o agente decidindo descompactar, depois decidindo ler as colunas, depois gerando as estatísticas e, por fim, acionando a notificação).

---

## Passo 1: Setup
Abra o seu terminal. O objetivo aqui é isolar completamente o ambiente de execução desse agente para que as dependências não conflitem com outros projetos seus.

Siga os comandos abaixo, passo a passo:

### A. Estrutura de diretórios
Vamos criar o diretório exato que você especificou e entrar nele.

```bash
mkdir -p ./data-eng-data-analisys-agent
cd ./data-eng-data-analisys-agent

```

### B. O Ambiente Virtual (O Sandbox do Agente)
No back-end, isolar as dependências é regra de ouro. Vamos criar um ambiente virtual (venv) e ativá-lo. 

```bash
# Cria o ambiente virtual chamado ".venv"
python3 -m venv .venv

# Ativa o ambiente virtual na sua sessão atual do terminal
source .venv/bin/activate

```

### C. Instalando as dependências
Agora vamos baixar as bibliotecas necessárias:
* `langgraph`: O orquestrador de estado e fluxo lógico.
* `langchain-google-genai`: O cliente HTTP (wrapper) que fará as chamadas REST para a API do Gemini.
* `pandas`: A engine de processamento de dados que usaremos nas ferramentas.
* `python-dotenv`: Para carregar nossa chave de API de forma segura para a memória (variáveis de ambiente).

Execute:
```bash
pip install langgraph langchain-google-genai pandas python-dotenv

```

### D. Criando os scripts

```bash
# Cria os arquivos vazios
touch .env main.py agent.py tools.py

```
* `.env`: Onde a chave da API vai morar.
* `main.py`: O ponto de entrada (entrypoint) que vai inicializar o fluxo.
* `agent.py`: Onde vamos desenhar o Grafo (nós e arestas).
* `tools.py`: Onde ficarão as suas funções puras de Python (suas APIs internas).

### E. Configuração da API Key
O agente precisa de permissão para consultar o cérebro (LLM). 
1. Acesse o **Google AI Studio** ([https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)) com sua conta Google.
2. Clique em **"Create API key"**.
3. Copie a chave gerada.
4. Abra o arquivo `.env` que criamos no seu editor de código favorito (VS Code, Cursor, Vim) e adicione a seguinte linha (substituindo pela sua chave real):

```env
GEMINI_API_KEY=sua_chave_gerada_aqui

```
