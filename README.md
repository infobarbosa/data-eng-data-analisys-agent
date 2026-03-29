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


---

## Passo 3: Implementação das Ferramentas (Tools)

O modelo de linguagem não executa ações no sistema operacional de forma autônoma. Para que o agente consiga manipular arquivos e dados, precisamos construir funções em Python e expô-las como "Ferramentas" (Tools). O LLM receberá a assinatura dessas funções (nome, parâmetros e descrição) e decidirá quando acioná-las.

Nesta fase, implementaremos três ferramentas fundamentais no arquivo `tools.py`.

### A. Construindo as Funções (Edite o arquivo `tools.py`)

Abra o arquivo `tools.py` criado no Passo 1 e insira o código abaixo. Observe o uso do decorador `@tool` do LangChain. A *docstring* (texto entre aspas triplas logo abaixo da declaração da função) é crucial, pois é através dela que o modelo entende o propósito da ferramenta e como utilizá-la.

```python
import os
import zipfile
import pandas as pd
from langchain_core.tools import tool

# 1. Tool de Extração de Arquivos
@tool
def extract_file(file_path: str) -> str:
    """
    Extrai um arquivo .zip e retorna o caminho do arquivo .csv contido nele.
    Se o arquivo de entrada já for um .csv, retorna o próprio caminho.
    """
    print(f"\n[Tool: extract_file] Processando o arquivo: {file_path}")
    
    if not os.path.exists(file_path):
        return f"Erro: O arquivo {file_path} não foi encontrado."

    if file_path.endswith('.csv'):
        return file_path
        
    if file_path.endswith('.zip'):
        extract_dir = os.path.dirname(file_path)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            extracted_files = zip_ref.namelist()
            
            # Busca o primeiro CSV dentro do ZIP
            for file in extracted_files:
                if file.endswith('.csv'):
                    csv_path = os.path.join(extract_dir, file)
                    print(f"[Tool: extract_file] Arquivo extraído com sucesso: {csv_path}")
                    return csv_path
                    
        return "Erro: Nenhum arquivo .csv encontrado dentro do .zip."
    
    return "Erro: Formato de arquivo não suportado."

# 2. Tool de Análise Estatística (Pandas)
@tool
def analyze_data(csv_path: str) -> str:
    """
    Lê um arquivo .csv e retorna um resumo estatístico contendo:
    - O nome das colunas e seus tipos de dados.
    - A contagem de valores nulos.
    - As estatísticas descritivas (média, mínimo, máximo, etc.) para colunas numéricas.
    """
    print(f"\n[Tool: analyze_data] Iniciando análise do arquivo: {csv_path}")
    
    try:
        # Carrega os dados usando a biblioteca Pandas
        df = pd.read_csv(csv_path)
        
        # Coleta metadados essenciais
        info_buffer = []
        info_buffer.append(f"Dimensões do dataset: {df.shape[0]} linhas e {df.shape[1]} colunas.\n")
        
        info_buffer.append("Tipos de Dados e Nulos:")
        for col in df.columns:
            null_count = df[col].isnull().sum()
            dtype = str(df[col].dtype)
            info_buffer.append(f"- Coluna '{col}' ({dtype}): {null_count} valores nulos.")
            
        info_buffer.append("\nEstatísticas Descritivas (Colunas Numéricas):")
        # O método describe() gera estatísticas como média, desvio padrão, min e max
        desc = df.describe().to_string()
        info_buffer.append(desc)
        
        print("[Tool: analyze_data] Análise concluída.")
        return "\n".join(info_buffer)
        
    except Exception as e:
        return f"Erro ao analisar o arquivo: {str(e)}"

# 3. Tool de Notificação
@tool
def notify_user(report: str) -> str:
    """
    Simula o envio de um e-mail com o relatório final da análise.
    Deve ser chamada apenas quando a análise exploratória estiver concluída.
    """
    print("\n[Tool: notify_user] Preparando o envio da notificação...")
    
    # Em um ambiente de produção real, aqui entraria a lógica de SMTP/API (ex: SendGrid, AWS SES)
    log_file = "relatorio_final_log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== RELATÓRIO DE ANÁLISE EXPLORATÓRIA ===\n\n")
        f.write(report)
        f.write("\n==========================================\n")
        
    print(f"[Tool: notify_user] Notificação salva no arquivo '{log_file}'.")
    return "Notificação enviada e registrada com sucesso."
```

---

#### O que acontece fisicamente nesta etapa?
As funções acima são código Python nativo. A biblioteca `langchain_core` utiliza o decorador `@tool` para abstrair os metadados da função (nome, tipo de entrada e docstring) e transformá-los em um esquema JSON (especificamente, no padrão de *Function Calling*). Na próxima fase, ensinaremos o LangGraph a injetar esse JSON na requisição para o modelo, permitindo que a IA solicite a execução dessas funções.

---

## Passo 4: Orquestração do Workflow

Nesta etapa, estabeleceremos a orquestração principal do sistema. Modificaremos o arquivo `agent.py` para implementar um padrão cíclico conhecido como **Tool Calling**. O fluxo consistirá em: o modelo avalia a solicitação, decide acionar uma ferramenta, o LangGraph executa a função localmente, injeta o resultado de volta no estado e devolve ao modelo para uma nova avaliação.

### A. Atualizando a Topologia do Grafo (Edite o arquivo `agent.py`)

Abra o arquivo `agent.py` e substitua o código anterior por esta nova estrutura. Introduziremos o `ToolNode` (um nó predefinido do LangGraph que executa as funções) e o `tools_condition` (uma aresta condicional que realiza o roteamento automático).

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage

# Importando as ferramentas
from tools import extract_file, analyze_data, notify_user

# 1. Definição do Estado
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Função de Fábrica para injeção de dependência
def create_agent_workflow(api_key: str):
    """
    Constrói e compila o grafo do agente, injetando as credenciais necessárias.
    """
    # Instanciação do modelo utilizando a chave injetada explicitamente
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0, 
        api_key=api_key
    )
    
    tools = [extract_file, analyze_data, notify_user]
    llm_with_tools = llm.bind_tools(tools)

    sys_msg = SystemMessage(content="""Você é um agente autônomo de análise de dados.
    Seu objetivo é processar arquivos solicitados, analisá-los e notificar o usuário.
    Siga estas etapas rigorosamente:
    1. Extraia o arquivo fornecido utilizando a ferramenta apropriada.
    2. Analise os dados extraídos para identificar colunas, nulos e distribuições estatísticas.
    3. Elabore um relatório consolidado e envie a notificação ao usuário.
    Sempre utilize as ferramentas disponíveis. Não invente ou simule dados.
    """)

    def agent_node(state: AgentState):
        print(">>> [Nó: Agente] Avaliando o estado atual e determinando a próxima ação...")
        messages = [sys_msg] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Construção da Topologia do Grafo
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    # Retorna o aplicativo compilado, pronto para invocação
    return workflow.compile()

```

### B. Atualizando o Ponto de Entrada (Edite o arquivo `main.py`)

Agora que o agente possui ferramentas e um fluxo de decisão, precisamos atualizar o `main.py` para enviar a solicitação real de análise de dados, abandonando a mensagem de "Hello World".

Substitua o conteúdo de `main.py` pelo código abaixo:

```python
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Importamos a função de fábrica em vez do objeto estático
from agent import create_agent_workflow

def main():
    print("Iniciando a orquestração do Agente Analista de Dados...\n")
    
    # 1. Recuperação de Configurações (Composition Root)
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    # Validação de segurança em nível de orquestrador
    if not api_key:
        raise ValueError("Erro Crítico: A variável de ambiente GEMINI_API_KEY não foi encontrada.")
        
    # 2. Injeção de Dependência
    # O main.py entrega a credencial para o construtor do agente
    app = create_agent_workflow(api_key=api_key)
    
    # 3. Definição do arquivo alvo (que criaremos no Passo 5)
    arquivo_alvo = "dados_teste.zip"
    
    mensagem_usuario = HumanMessage(
        content=f"Inicie a rotina de análise exploratória para o arquivo '{arquivo_alvo}' e notifique-me ao concluir."
    )
    
    estado_inicial = {"messages": [mensagem_usuario]}
    
    print("Invocando a execução do fluxo autônomo...\n")
    
    # 4. Execução do Grafo
    estado_final = app.invoke(estado_inicial)
    
    print("\n=== Execução Concluída ===")
    
    resposta_final = estado_final["messages"][-1].content
    print(resposta_final)

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

```python
import pandas as pd
import zipfile
import os
import numpy as np

def generate_mock_data():
    print("Gerando massa de dados simulada...")
    
    # 1. Definição do conjunto de dados
    # Incluímos valores nulos (np.nan) intencionalmente na coluna de avaliação
    data = {
        'id_transacao': range(1, 101),
        'valor_compra': np.random.uniform(10.0, 500.0, 100),
        'idade_cliente': np.random.randint(18, 70, 100),
        'categoria_produto': np.random.choice(['Eletrônicos', 'Roupas', 'Alimentos'], 100),
        'avaliacao_cliente': np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, np.nan], 100)
    }

    df = pd.DataFrame(data)
    csv_filename = 'vendas_mock.csv'
    
    # 2. Exportação para CSV
    df.to_csv(csv_filename, index=False)

    # 3. Compactação para ZIP
    zip_filename = 'dados_teste.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(csv_filename)

    # 4. Limpeza do arquivo original
    # Removemos o CSV solto para forçar o agente a utilizar a ferramenta de extração
    os.remove(csv_filename) 
    
    print(f"Arquivo '{zip_filename}' criado com sucesso no diretório atual.")

if __name__ == "__main__":
    generate_mock_data()
```

Execute este script no seu terminal para preparar o ambiente:
```bash
python create_mock_data.py

```

### B. Execução do Pipeline Autônomo

Com o arquivo `dados_teste.zip` presente no diretório, nosso cenário está pronto. Agora, iniciaremos a execução do agente através do nosso ponto de entrada principal.

No terminal, execute:
```bash
python main.py
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
