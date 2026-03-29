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
Nesta etapa, estabelecemos a orquestração principal do sistema. Vamos unir as Tools construídas na Fase 3 com o Grafo iniciado na Fase 2.
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

### B. O Ambiente Virtual

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
* `.env`: Arquivo destinado ao armazenamento da chave de API.
* `main.py`: O ponto de entrada (entrypoint) que vai inicializar o fluxo.
* `agent.py`: Onde vamos desenhar o Grafo (nós e arestas).
* `tools.py`: Onde ficarão as suas funções puras de Python (suas APIs internas).

### E. Configuração da API Key
O agente precisa de autenticação para realizar requisições ao modelo (LLM). 
1. Acesse o **Google AI Studio** ([https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)) com sua conta Google.
2. Clique em **"Create API key"**.
3. Copie a chave gerada.
4. Abra o arquivo `.env` que criamos no seu editor de código favorito (VS Code, Cursor, Vim) e adicione a seguinte linha (substituindo pela sua chave real):

```env
GEMINI_API_KEY=sua_chave_gerada_aqui

```

---

## Passo 2: O "Hello World" do LangGraph

Nesta etapa, não vamos ler nenhum arquivo ainda. Nosso objetivo é garantir que:
1. O grafo consiga gerenciar o **Estado** (nossa variável de memória).
2. O nosso servidor local consiga fazer a chamada REST (HTTPS) para a API do Google, enviando e recebendo dados corretamente.

---

### A. Construindo o Grafo (Edite o arquivo `agent.py`)

Aqui definimos a lógica de processamento e a topologia. Fisicamente, estamos apenas criando um dicionário tipado (o Estado) e uma função Python comum que recebe esse dicionário, altera uma chave e o devolve.

Copie e cole este código no seu `agent.py`:

```python
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Definindo o Estado (A memória do nosso back-end)
# Tudo o que o agente souber durante a execução viverá dentro deste dicionário.
class AgentState(TypedDict):
    mensagem_entrada: str
    resposta_agente: Optional[str]

# 2. Inicializando o cliente da API do LLM
# Estamos instanciando o wrapper que fará os POSTs para a API do Google.
# Usamos temperature=0 para respostas mais determinísticas (ideal para análise de dados).
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# 3. Criando o Nó (A função de processamento)
def pensar_node(state: AgentState):
    print(">>> [Nó: pensar] Acessando a API do Gemini...")
    
    # Lemos a entrada atual do estado
    mensagem = state["mensagem_entrada"]
    
    # Abre uma conexão HTTPS, envia a string para o Google, aguarda e recebe a resposta.
    resposta = llm.invoke(mensagem)
    
    # Retornamos APENAS a parte do estado que queremos atualizar
    # O LangGraph se encarrega de fazer o "merge" no objeto de estado global.
    return {"resposta_agente": resposta.content}

# 4. Construindo a Topologia do Grafo
workflow = StateGraph(AgentState)

# Adicionamos o nosso único nó
workflow.add_node("pensar", pensar_node)

# Definimos o fluxo: Começo -> Nó Pensar -> Fim
workflow.set_entry_point("pensar")
workflow.add_edge("pensar", END)

# Compilamos o Grafo para transformá-lo em um executável invocável
app = workflow.compile()

```

---

### B. O Ponto de Entrada (Edite o arquivo `main.py`)

Ele carrega a chave de segurança, prepara o payload inicial (o Estado Inicial) e inicia a execução do Grafo.

Copie e cole este código no seu `main.py`:

```python
import os
from dotenv import load_dotenv
from agent import app  # Importamos o grafo compilado que acabamos de criar

# 1. Injeta a GEMINI_API_KEY do arquivo .env nas variáveis de ambiente do sistema operacional
load_dotenv()

def main():
    print("Iniciando o servidor local do Agente...\n")
    
    # 2. Montamos o payload inicial
    estado_inicial = {
        "mensagem_entrada": "Olá! Assuma o papel de um agente de análise de dados. Diga apenas: 'Servidor online. Conexão com o modelo de linguagem estabelecida com sucesso!'."
    }
    
    print("Invocando o Grafo...")
    
    # 3. Executamos o Grafo
    # O .invoke() bloqueia a thread até que o grafo chegue ao nó END.
    estado_final = app.invoke(estado_inicial)
    
    # 4. Consumimos o resultado final
    print("\n=== Resposta do Agente ===")
    print(estado_final["resposta_agente"])
    print("==========================")

if __name__ == "__main__":
    main()
```

---

### C. A Execução (O Teste Físico)

Volte para o seu terminal (certifique-se de que o `(.venv)` ainda está ativo e que você está dentro da pasta do projeto).

Rode o script principal:
```bash
python main.py

```

### O que deve acontecer no seu terminal:
Você verá os `prints` mostrando a ordem exata de execução. Primeiro o `main.py` avisa que iniciou, depois a thread entra no `agent.py` (dentro do `pensar_node`), avisa que está acessando a API, e por fim, imprime a resposta exata que pedimos para o Gemini gerar.

Rode o comando e me diga: **Apareceu a mensagem de "Servidor online" ou o console retornou algum erro de chave de API ou de importação?** Assim que validarmos que a rede está comunicando, vamos implementar as ferramentas de manipulação de dados (Pandas) na Fase 3!