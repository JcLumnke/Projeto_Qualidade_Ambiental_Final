# Projeto de Classificação de Qualidade Ambiental

Projeto criado a partir do notebook `Desafio_Final.ipynb` para classificar a qualidade ambiental com base em dados de sensores. A estrutura foi desenvolvida para facilitar a refatoração do código, o reuso de componentes e o deploy da solução via API.

## Estrutura do Projeto

Projeto_Qualidade_Ambiental_Final/
├─ data/                   # Datasets de entrada
├─ models/                 # Modelos treinados (serializados)
├─ mlruns/                 # Experimentos e modelos versionados pelo MLflow
├─ notebooks/              # Notebooks de exploração e prototipagem
│  └─ Desafio_Final.ipynb
├─ outputs/                # Resultados (gráficos, matrizes de confusão, etc.)
│  ├─ check_models.py      # Script para listar e verificar modelos salvos
│  └─ clean_mlruns.py      # Script para limpar logs e modelos antigos do MLflow
├─ src/                    # Código fonte da aplicação
│  ├─ app.py               # API FastAPI para servir o modelo
│  ├─ data_processing.py   # Carregamento e preparação inicial do dataset
│  ├─ data_cleaning.py     # Funções de limpeza e tratamento dos dados
│  ├─ feature_engineering.py # Criação de features derivadas
│  ├─ eda.py               # Análise exploratória dos dados
│  ├─ train.py             # Treinamento e salvamento de modelos com MLflow
│  └─ utils.py             # Funções auxiliares
├─ tests/                  # Testes automatizados
├─ requirements.txt        # Dependências do projeto
└─ Procfile                # Configuração para deploy (Heroku, etc.)

## Pipeline de Machine Learning

1.  **Carregamento de Dados**: Leitura do dataset a partir da pasta `data/`.
2.  **Análise Exploratória (EDA)**: Análise inicial de colunas, tipos e amostras de dados (`eda.py`).
3.  **Limpeza e Pré-processamento**: Tratamento de valores nulos e normalização de colunas (`data_cleaning.py`).
4.  **Codificação de Variáveis**: Conversão de variáveis categóricas para formato numérico.
5.  **Balanceamento de Classe**: Aplicação de técnicas para balancear a variável alvo (`Qualidade_Ambiental`).
6.  **Engenharia de Features**: Criação de variáveis de risco (Risco de Chuva Ácida, Smog Fotoquímico e Efeito Estufa).
7.  **Treinamento**: Treinamento de modelos de classificação (`RandomForestClassifier` e `SVM`).
8.  **Versionamento**: Salvamento e registro dos modelos utilizando **MLflow**.
9.  **API com FastAPI**: Disponibilização do modelo treinado através de endpoints para predição e pré-processamento.

## Pipeline de Machine Learning

1.  **Carregamento de Dados**: Leitura do dataset a partir da pasta `data/`.
2.  **Análise Exploratória (EDA)**: Análise inicial de colunas, tipos e amostras de dados (`eda.py`).
3.  **Limpeza e Pré-processamento**: Tratamento de valores nulos e normalização de colunas (`data_cleaning.py`).
4.  **Codificação de Variáveis**: Conversão de variáveis categóricas para formato numérico.
5.  **Balanceamento de Classe**: Aplicação de técnicas para balancear a variável alvo (`Qualidade_Ambiental`).
6.  **Engenharia de Features**: Criação de variáveis de risco (Risco de Chuva Ácida, Smog Fotoquímico e Efeito Estufa).
7.  **Treinamento**: Treinamento de modelos de classificação (`RandomForestClassifier` e `SVM`).
8.  **Versionamento**: Salvamento e registro dos modelos utilizando **MLflow**.
9.  **API com FastAPI**: Disponibilização do modelo treinado através de endpoints para predição e pré-processamento.

## Como Executar

### 1. Pré-requisitos

Clone o repositório e instale as dependências listadas no `requirements.txt`:

```bash
pip install -r requirements.txt

# Executar a análise exploratória de dados
python -m src.eda

# Executar a limpeza dos dados
python -m src.data_cleaning

# Executar o treinamento do modelo
python .\src\train.py


# Verificar os modelos salvos pelo MLflow (ID, tamanho, etc.)
python .\src\check_models.py

# Limpar execuções e modelos antigos da pasta mlruns
python .\src\clean_miruns.py

uvicorn src.app:app --reload --port 8000

{
  "features": {
    "Temperatura": 31.95,
    "Umidade": 32.33,
    "CO2": 19998.31,
    "CO": 48.22,
    "Pressao_Atm": 962.63,
    "NO2": 23.57,
    "SO2": 23.69,
    "O3": 37.42
  }
}

{
  "prediction": {
    "Qualidade_Ambiental": "Boa",
    "Risco_Chuva_Acida": "Não",
    "Risco_Smog_Fotoquimico": "Não",
    "Risco_Efeito_Estufa": "Sim"
  }
}

Sobre o Projeto
Este estudo foi desenvolvido como parte do programa de Residência em Inteligência Artificial do UniSenai - Campus Florianópolis (SC), com o objetivo de aprimorar os conhecimentos em Machine Learning e MLOps.

Aviso: O dataset utilizado é o dataset_ambiental.csv. As previsões geradas pelo modelo são baseadas nos dados de treinamento e devem ser consideradas uma simulação para fins acadêmicos. Para uma aplicação no mundo real, seria necessária uma base de dados mais robusta e a validação de especialistas da área ambiental.