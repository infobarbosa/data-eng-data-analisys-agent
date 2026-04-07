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
* **1.1.** Criação da estrutura de pastas no diretório `./data-eng-data-analysis-agent`.
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
mkdir -p ./data-eng-data-analysis-agent

```

### B. O Ambiente Virtual

```bash
# Cria o ambiente virtual chamado ".venv"
python3 -m venv ./data-eng-data-analysis-agent/.venv

# Ativa o ambiente virtual na sua sessão atual do terminal
source ./data-eng-data-analysis-agent/.venv/bin/activate

```

### C. Instalando as dependências
Agora vamos baixar as bibliotecas necessárias:
* `langgraph`: O orquestrador de estado e fluxo lógico.
* `langchain-google-genai`: O cliente HTTP (wrapper) que fará as chamadas REST para a API do Gemini.
* `pandas`: A engine de processamento de dados que usaremos nas ferramentas.
* `python-dotenv`: Para carregar nossa chave de API de forma segura para a memória (variáveis de ambiente).

1. Crie um arquivo `requirements.txt`:
```sh
touch ./data-eng-data-analysis-agent/requirements.txt

```

2. Adicione as seguintes dependências:
```
langgraph
langchain-google-genai
pandas
python-dotenv

```

3. Instale as dependências:
```bash
pip install -r ./data-eng-data-analysis-agent/requirements.txt

```

### D. Criando os scripts

```bash
# Cria os arquivos vazios
touch ./data-eng-data-analysis-agent/.env 
touch ./data-eng-data-analysis-agent/main.py 
touch ./data-eng-data-analysis-agent/agent.py 
touch ./data-eng-data-analysis-agent/tools.py 
touch ./data-eng-data-analysis-agent/config.py

```
Onde:
* `.env`: Arquivo destinado ao armazenamento da chave de API.
* `main.py`: O ponto de entrada (entrypoint) que vai inicializar o fluxo.
* `agent.py`: Onde vamos desenhar o Grafo (nós e arestas).
* `tools.py`: Onde ficarão as funções puras de Python (suas APIs internas).
* `config.py`: Onde ficarão as configurações do agente.

### E. Configuração da API Key
O agente precisa de autenticação para realizar requisições ao modelo (LLM). 
1. Acesse o **Google AI Studio** ([https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)) com sua conta Google.
2. Clique em **"Create API key"**.
3. Copie a chave gerada.
4. Abra o arquivo `.env` que criamos no seu editor de código favorito (VS Code, Cursor, Vim) e adicione a seguinte linha (substituindo pela sua chave real):

```sh
touch ./data-eng-data-analysis-agent/.env

```

Inclua o seguinte conteúdo no arquivo `.env`:
```env
GEMINI_API_KEY=sua_chave_gerada_aqui
GEMINI_MODEL=gemini-2.5-flash

```

---

## Passo 2: O "Hello World" do LangGraph

Nesta etapa, não vamos ler nenhum arquivo ainda. Nosso objetivo é garantir que:

* O grafo consiga gerenciar o Estado (nossa variável de memória).
* O nosso servidor local consiga fazer a chamada REST (HTTPS) para a API do Google, enviando e recebendo dados corretamente.

Utilizaremos o padrão de **Injeção de Dependência**, onde o ponto de entrada da aplicação (`main.py`) fornece as credenciais necessárias para a construção do agente em `agent.py`.

### A. Configurações (`config.py`)
Em arquiteturas limpas, o acesso a variáveis de ambiente é considerado um detalhe de infraestrutura. Criaremos o arquivo config.py para encapsular essa responsabilidade, permitindo que o restante da aplicação consuma objetos de configuração tipados e validados.
```python
# config.py
import os
from dotenv import load_dotenv

class Settings:
    """
    Camada de Configuração: Responsável por carregar e validar 
    as variáveis de ambiente do projeto.
    """
    def __init__(self):
        load_dotenv()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def validate(self):
        """Valida se as credenciais críticas estão presentes."""
        if not self.gemini_api_key:
            raise ValueError("ERRO: Variável GEMINI_API_KEY não configurada no arquivo .env")

```

### B. O Fluxo (`agent.py`)

Neste arquivo, definiremos a estrutura de memória do agente (Estado) e uma função de fábrica que encapsula a lógica de criação do grafo.

```python
# agent.py
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Definição do Estado (State)
# Objeto de dados que trafega entre os nós do grafo.
class AgentState(TypedDict):
    mensagem_entrada: str
    resposta_agente: Optional[str]

# 2. Função de Fábrica para construção do Workflow
def create_agent_workflow(api_key: str, model: str) -> StateGraph[AgentState]:
    """
    Constrói o grafo do agente injetando a chave de API necessária.
    """
    
    # Inicialização do modelo com a chave injetada
    llm = ChatGoogleGenerativeAI(model=model, api_key=api_key, temperature=0)

    # Definição do Nó de Execução
    def invocar_modelo_node(state: AgentState):
        print(">>> [Nó: invocar_modelo] Solicitando processamento ao modelo...")
        
        # O método .invoke() realiza a chamada HTTPS para o provedor
        resposta = llm.invoke(state["mensagem_entrada"])
        
        # Retorno da atualização do estado
        return {"resposta_agente": resposta.content}

    # Configuração da Topologia
    workflow = StateGraph(AgentState)
    workflow.add_node("invocar_modelo", invocar_modelo_node)
    
    workflow.set_entry_point("invocar_modelo")
    workflow.add_edge("invocar_modelo", END)

    # Compilação e retorno do executável
    return workflow.compile()
```

### C. O Ponto de Entrada (`main.py`)

O arquivo `main.py` atua como a **Raiz de Composição** (Composition Root). Ele é o responsável por carregar as variáveis de ambiente e injetar a dependência (API Key) no construtor do agente.

```python
# main.py
import os
from config import Settings
from dotenv import load_dotenv
from agent import create_agent_workflow

def main():
    print("Iniciando o Composition Root da aplicação...\n")
    
    # 1. Carga e recuperação de configurações de ambiente
    settings = Settings()
    settings.validate()

    # 2. Injeção de dependência e construção do agente
    app = create_agent_workflow(
        api_key=settings.gemini_api_key, 
        model=settings.gemini_model
    )
    
    # 3. Definição do payload inicial
    estado_inicial = {
        "mensagem_entrada": "Olá! Informe o status da conexão com o modelo."
    }
    
    print("Iniciando execução do fluxo...")
    
    # 4. Invocação do fluxo
    estado_final = app.invoke(estado_inicial)
    
    print("\n=== Resposta do Modelo ===")
    print(estado_final["resposta_agente"])
    print("==========================")

if __name__ == "__main__":
    main()

```

---

### D. Execute o teste

Rode o script principal:
```bash
python ./data-eng-data-analysis-agent/main.py

```

### O que deve acontecer no seu terminal:
Você verá os `prints` mostrando a ordem exata de execução. Primeiro o `main.py` avisa que iniciou, depois a thread entra no `agent.py` (dentro do `invocar_modelo_node`), avisa que está acessando a API, e por fim, imprime a resposta exata que pedimos para o Gemini gerar.

---

## Passo 3: Implementação das Ferramentas (Tools)

O modelo de linguagem não executa ações no sistema operacional de forma autônoma. Para que o agente consiga manipular arquivos e dados, precisamos construir funções em Python e expô-las como "Ferramentas" (Tools). O LLM receberá a assinatura dessas funções (nome, parâmetros e descrição) e decidirá quando acioná-las.

Nesta fase, implementaremos três ferramentas fundamentais no arquivo `tools.py`.

### A. Construindo as Funções (`tools.py`)

Abra o arquivo `tools.py` e insira o código abaixo. Observe o uso do decorador `@tool` do LangChain. A *docstring* (texto entre aspas triplas logo abaixo da declaração da função) é crucial, pois é através dela que o modelo entende o propósito da ferramenta e como utilizá-la.

```python
# tools.py
import os
import zipfile
import pandas as pd
from langchain_core.tools import tool

def get_base_path():
    """Helper para garantir que os arquivos sejam salvos na pasta do projeto."""
    return os.path.dirname(os.path.abspath(__file__))

# 1. Tool de Extração (Responsabilidade: I/O de Arquivos)
@tool
def extract_file(file_path: str) -> str:
    """
    Extrai um arquivo .zip e retorna o caminho do .csv contido nele.
    Se já for .csv, valida a existência e retorna o caminho.
    """
    base_path = get_base_path()
    # Resolve o caminho para absoluto se for passado apenas o nome
    full_path = file_path if os.path.isabs(file_path) else os.path.join(base_path, file_path)
    
    print(f"\n[Tool: extract_file] Verificando: {full_path}")
    
    if not os.path.exists(full_path):
        return f"Erro: Ficheiro não encontrado em {full_path}"

    try:
        if full_path.endswith('.csv'):
            return full_path
            
        if full_path.endswith('.zip'):
            with zipfile.ZipFile(full_path, 'r') as zip_ref:
                zip_ref.extractall(base_path)
                for f in zip_ref.namelist():
                    if f.endswith('.csv'):
                        return os.path.join(base_path, f)
            return "Erro: Nenhum CSV encontrado no ZIP."
    except Exception as e:
        return f"Erro na extração: {str(e)}"
    
    return "Erro: Formato não suportado."

# 2. Tool de Análise (Responsabilidade: Processamento de Dados)
@tool
def analyze_data(csv_path: str) -> str:
    """
    Realiza análise estatística quantitativa em um arquivo CSV.
    Retorna dimensões, tipos de dados e estatísticas descritivas.
    """
    print(f"\n[Tool: analyze_data] Analisando: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        stats = {
            "linhas": df.shape[0],
            "colunas": df.shape[1],
            "nulos": df.isnull().sum().to_dict(),
            "resumo": df.describe().to_string()
        }
        return f"Análise concluída: {stats}"
    except Exception as e:
        return f"Erro na análise: {str(e)}"

# 3. Tool de Notificação (Responsabilidade: Comunicação/Output)
@tool
def notify_user(report: str) -> str:
    """
    Gera o artefato final da análise (relatório de log).
    Deve ser chamada apenas para encerrar o ciclo de análise.
    """
    log_path = os.path.join(get_base_path(), "relatorio_final_log.txt")
    print(f"\n[Tool: notify_user] Gravando relatório em: {log_path}")
    
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"=== RELATÓRIO FINAL ===\n\n{report}")
        return f"Sucesso: Relatório gravado em {log_path}"
    except Exception as e:
        return f"Erro na notificação: {str(e)}"

```

---

#### O que acontece fisicamente nesta etapa?
As funções acima são código Python nativo. A biblioteca `langchain_core` utiliza o decorador `@tool` para abstrair os metadados da função (nome, tipo de entrada e docstring) e transformá-los em um esquema JSON (especificamente, no padrão de *Function Calling*). Na próxima fase, ensinaremos o LangGraph a injetar esse JSON na requisição para o modelo, permitindo que a IA solicite a execução dessas funções.

---

## Passo 4: Orquestração do Workflow

Nesta etapa, estabeleceremos a orquestração principal do sistema. Modificaremos o arquivo `agent.py` para implementar um padrão cíclico conhecido como **Tool Calling**. O fluxo consistirá em: o modelo avalia a solicitação, decide acionar uma ferramenta, o LangGraph executa a função localmente, injeta o resultado de volta no estado e devolve ao modelo para uma nova avaliação.

### A. Atualizando a Topologia do Grafo (`agent.py`)

Abra o arquivo `agent.py` e substitua o código anterior por esta nova estrutura. Introduziremos o `ToolNode` (um nó predefinido do LangGraph que executa as funções) e o `tools_condition` (uma aresta condicional que realiza o roteamento automático).

```python
# agent.py
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage

# Importação das ferramentas (nossos adaptadores de saída)
from tools import extract_file, analyze_data, notify_user

# 1. Definição do Estado (Memória do Agente)
class AgentState(TypedDict):
    # O add_messages garante que o histórico seja acumulado e não sobrescrito
    messages: Annotated[list, add_messages]

# 2. Configuração do Comportamento (System Prompt)
AGENT_SYSTEM_PROMPT = """Você é um agente autônomo de análise de dados.
Seu objetivo é processar arquivos solicitados, analisá-los e notificar o usuário.
Siga estas etapas rigorosamente:
1. Extraia o arquivo fornecido utilizando a ferramenta apropriada.
2. Analise os dados extraídos para identificar colunas, nulos e distribuições estatísticas.
3. Elabore um relatório consolidado e envie a notificação ao usuário.
Sempre utilize as ferramentas disponíveis. Não invente ou simule dados.
"""

# 3. Função de Fábrica do Orquestrador
def create_agent_workflow(api_key: str, model: str):
    """
    Constrói e compila o grafo do agente, injetando as credenciais.
    """
    llm = ChatGoogleGenerativeAI(model=model, temperature=0, api_key=api_key)
    
    # Vinculamos as ferramentas ao modelo
    tools = [extract_file, analyze_data, notify_user]
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        print(">>> [Nó: Agente] Analisando histórico e decidindo próxima ação...")
        # Injetamos o prompt de sistema no início do histórico de mensagens
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Montagem da Topologia do Grafo
    workflow = StateGraph(AgentState)
    
    # Adicionamos os nós (unidades de execução)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    
    # Definimos as arestas (fluxo de decisão)
    workflow.set_entry_point("agent")
    
    # Aresta Condicional: Decide se vai para as ferramentas ou encerra (END)
    workflow.add_conditional_edges("agent", tools_condition)
    
    # Após usar uma ferramenta, o fluxo volta obrigatoriamente para o agente
    workflow.add_edge("tools", "agent")

    return workflow.compile()

```

### B. Atualizando o Ponto de Entrada (`main.py`)

Agora que o agente possui ferramentas e um fluxo de decisão, precisamos atualizar o `main.py` para enviar a solicitação real de análise de dados, abandonando a mensagem de "Hello World".

Substitua o conteúdo de `main.py` pelo código abaixo:

```python
# main.py
import os
from langchain_core.messages import HumanMessage
from config import Settings
from agent import create_agent_workflow

def main():
    print("Iniciando a orquestração do Agente Analista de Dados...\n")
    
    # 1. Recuperação e Validação de Configurações
    settings = Settings()
    settings.validate()
    
    # 2. Construção do Agente via Injeção de Dependência
    app = create_agent_workflow(
        api_key=settings.gemini_api_key, 
        model=settings.gemini_model
    )
    
    # 3. Definição do Problema (Input)
    base_path = os.path.dirname(os.path.abspath(__file__))
    arquivo_alvo = os.path.join(base_path, "dados_teste.zip")
    
    mensagem_usuario = HumanMessage(
        content=f"Inicie a rotina de análise para o arquivo '{arquivo_alvo}'."
    )
    
    # 4. Execução do Fluxo Autônomo
    print("Invocando a execução do grafo...\n")
    estado_final = app.invoke({"messages": [mensagem_usuario]})
    
    print("\n=== Execução Concluída ===")
    print(estado_final["messages"][-1].content)

if __name__ == "__main__":
    main()

```

---

#### O que acontece fisicamente nesta etapa?
O método `bind_tools` converte as funções Python (extraídas via decorador `@tool`) em um esquema JSON e as anexa ao payload da requisição HTTP. Quando o modelo de linguagem recebe esse payload e identifica que precisa ler um arquivo, sua resposta à API não é um texto convencional, mas sim um objeto estruturado solicitando a chamada da função `extract_file`. 

A função `tools_condition` intercepta essa resposta, roteia a execução para o nó `tools` no seu ambiente local (seu MacBook), executa o código Python da ferramenta, e anexa o resultado (o caminho do CSV, por exemplo) de volta no histórico para que o modelo possa prosseguir com o Passo 2 (análise).

---

Excelente. Chegamos à etapa final do laboratório, onde consolidamos todo o conhecimento aplicado.

Neste **Passo 5**, vamos gerar uma massa de dados controlada (mock data) para simular um cenário real de engenharia de dados. Em seguida, executaremos o pipeline de ponta a ponta e analisaremos o comportamento autônomo do agente através dos logs gerados no terminal.

Adicione este conteúdo ao seu material didático:

---

## Passo 5: Teste End-to-End

A validação de um sistema autônomo exige um cenário com variáveis conhecidas para que possamos auditar as decisões tomadas pelo modelo. Para isso, criaremos um arquivo compactado contendo uma base de dados simulada.

### A. Geração da Massa de Dados

Para garantir que todos os alunos tenham exatamente a mesma estrutura de dados para o teste, criaremos um script utilitário. Este script irá gerar um arquivo `.csv` com colunas numéricas, categorias e alguns valores nulos intencionais (para testar a ferramenta de análise estatística), e então irá compactá-lo em um arquivo `.zip`.

Crie um arquivo chamado `create_mock_data.py` na raiz do seu projeto e insira o seguinte código:

```sh
touch ./data-eng-data-analysis-agent/create_mock_data.py

```

```python
# create_mock_data.py
import pandas as pd
import zipfile
import os
import numpy as np

def generate_mock_data():
    # Resolve o caminho absoluto do diretório deste script
    base_path = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(base_path, 'dados_teste.zip')
    csv_temp_path = os.path.join(base_path, 'vendas_mock.csv')

    print(f"Gerando massa de dados em: {base_path}")
    
    # Criando dados simulados com nulos para testar a robustez da análise
    data = {
        'id_transacao': range(1, 101),
        'valor_compra': np.random.uniform(10.0, 500.0, 100),
        'categoria_produto': np.random.choice(['Eletrônicos', 'Roupas', 'Alimentos'], 100),
        'avaliacao_cliente': np.random.choice([1.0, 5.0, np.nan], 100)
    }

    df = pd.DataFrame(data)
    df.to_csv(csv_temp_path, index=False)

    # Compactação garantindo que o CSV interno não tenha caminhos relativos
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(csv_temp_path, arcname='vendas_mock.csv')

    os.remove(csv_temp_path) 
    print(f"Sucesso: Arquivo '{zip_path}' gerado.")

if __name__ == "__main__":
    generate_mock_data()

```

Execute este script no seu terminal para preparar o ambiente:
```bash
python ./data-eng-data-analysis-agent/create_mock_data.py

```

### B. Execução do Pipeline Autônomo

Com o arquivo `dados_teste.zip` presente no diretório, nosso cenário está pronto. Agora, iniciaremos a execução do agente através do nosso ponto de entrada principal.

No terminal, execute:
```bash
python ./data-eng-data-analysis-agent/main.py

```

### C. Auditoria de Execução (Análise de Logs)

Durante a execução, observe atentamente a saída no terminal. O comportamento esperado reflete o padrão de orquestração que configuramos no LangGraph. A sequência lógica deve ser:

1.  **Início da Orquestração:** O `main.py` envia a instrução para o nó do agente.
2.  **Roteamento para Extração:** O modelo avalia a solicitação, percebe que o alvo é um `.zip` e invoca a ferramenta correspondente. Você verá o log impresso pela sua função Python:
    `[Tool: extract_file] Processando o arquivo: dados_teste.zip`
    `[Tool: extract_file] Arquivo extraído com sucesso: vendas_mock.csv`
3.  **Roteamento para Análise:** O LangGraph devolve o caminho do arquivo extraído para o modelo. O modelo, seguindo o prompt de sistema, invoca a ferramenta de leitura do Pandas. Você verá:
    `[Tool: analyze_data] Iniciando análise do arquivo: vendas_mock.csv`
    `[Tool: analyze_data] Análise concluída.`
4.  **Roteamento para Notificação:** Com os dados estatísticos em mãos (estado), o modelo formata o relatório e aciona a última ferramenta.
    `[Tool: notify_user] Preparando o envio da notificação...`
    `[Tool: notify_user] Notificação salva no arquivo 'relatorio_final_log.txt'.`
5.  **Encerramento:** O agente conclui que o fluxo está completo, gera a resposta final em texto para o usuário e o programa é finalizado.

Após a execução, um novo arquivo chamado `relatorio_final_log.txt` deverá aparecer no seu diretório. Abra-o para verificar o relatório redigido pelo modelo, contendo as dimensões do dataset, os tipos de dados e as estatísticas geradas pelo Pandas.

---

## Passo 6: Dedução Ontológica e Inferência Semântica

Até o momento, o agente executa uma análise quantitativa (estatística descritiva). Neste passo, incrementaremos o sistema com a capacidade de realizar uma **análise qualitativa**. O objetivo é que o agente não apenas leia os números, mas deduza o significado real dos dados (Ontologia) e o contexto de negócio do arquivo.

### Por que realizar este incremento?
Nomes de colunas em bancos de dados costumam ser abreviados ou ambíguos (ex: `CAT`, `VAL`, `USR`). Enquanto a ferramenta de estatística nos dá a "forma" do dado, a inspeção de uma amostra real permite que o modelo de linguagem (LLM) identifique padrões semânticos e infira a identidade do dataset.

### A. Nova Ferramenta de Inspeção (`tools.py`)

Adicionaremos a função `get_sample_data`. Ela fornecerá ao agente uma visão direta de alguns registros do arquivo, servindo como a "evidência qualitativa" para a dedução.

```python
# Adicione ao final do arquivo tools.py

@tool
def get_sample_data(csv_path: str) -> str:
    """
    Retorna as primeiras 5 linhas de um ficheiro .csv para inspeção qualitativa.
    Essencial para inferir o significado semântico das colunas.
    """
    base_path = get_base_path()
    full_path = csv_path if os.path.isabs(csv_path) else os.path.join(base_path, csv_path)
    
    print(f"\n[Tool: get_sample_data] Obtendo amostra de: {full_path}")
    
    try:
        df = pd.read_csv(full_path)
        # Retorna o cabeçalho como string para o histórico de mensagens
        return df.head(5).to_string()
    except Exception as e:
        return f"Erro ao obter amostra: {str(e)}"
```

### B. Configuração da Lógica de Inferência (agent.py)

Precisamos agora registrar a nova ferramenta e, mais importante, atualizar as instruções de comportamento do agente (*System Prompt*) para que ele saiba como correlacionar os dados estatísticos com a amostra coletada.

#### B.1. Importação da nova ferramenta
```python
# 1. Atualize a importação no topo do arquivo agent.py
from tools import extract_file, analyze_data, notify_user, get_sample_data
```

#### B.2. Atualização do System Prompt
```python

# 2. Refine o AGENT_SYSTEM_PROMPT para incluir a diretriz de inferência
AGENT_SYSTEM_PROMPT = """Você é um agente autônomo de análise de dados.
Sua missão é realizar uma análise exploratória profunda e semântica.

PROTOCOLO OBRIGATÓRIO:
1. EXTRAÇÃO: Use 'extract_file' para acessar os dados.
2. ESTATÍSTICA: Use 'analyze_data' para entender as distribuições.
3. AMOSTRAGEM: Use 'get_sample_data' para visualizar exemplos reais.
4. INFERÊNCIA SEMÂNTICA: Cruze as estatísticas com a amostra para deduzir:
   - O significado de cada coluna no mundo real.
   - A natureza do ficheiro (ex: Log de Vendas, Cadastro de RH).
5. NOTIFICAÇÃO: Consolide o relatório técnico e a dedução ontológica.
"""

```
#### B.3. Na função create_agent_workflow, inclua a nova ferramenta
```python
# 3. Na função create_agent_workflow, inclua a nova ferramenta
    tools = [extract_file, analyze_data, get_sample_data, notify_user]
```

### C. Execute o teste

Rode o script principal:
```bash
python ./data-eng-data-analysis-agent/main.py

```

---

### O que observamos nesta etapa?

Ao executar o `main.py` novamente, observamos que o agente agora realiza uma iteração adicional no grafo. Além do resumo do Pandas, ele solicita a amostra para confirmar suas inferências.

**A Lógica da Correlação:**
Fisicamente, o modelo de linguagem recebe no seu histórico de mensagens:
1.  **Evidência A (Estatística):** "A coluna `VAL` tem valores entre 10 e 500".
2.  **Evidência B (Amostra):** "A coluna `VAL` contém dados como `125.50`, `45.00`".
3.  **Dedução Cognitiva:** O modelo correlaciona essas informações com o contexto do arquivo e conclui: *"A coluna VAL refere-se ao Valor de Venda Unitário da transação"*.

Este passo demonstra como o LangGraph permite que o agente acumule evidências em seu Estado até que possua informações suficientes para realizar uma inferência de alto nível, simulando o processo de análise de um Engenheiro de Dados humano.