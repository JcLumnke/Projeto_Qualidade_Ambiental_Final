import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
from dotenv import load_dotenv
import requests

# -----------------------------
# Imports relativos do seu projeto original
# (Mantidos para garantir que o endpoint /preprocess funcione)
# -----------------------------
try:
    from .data_processing import load_data
    from .eda import exploracao_df
    from .data_cleaning import (
        tratar_pressao_atm,
        codificar_qualidade_ambiental,
        balancear_qualidade_ambiental
    )
    from .feature_engineering import (
        criar_risco_chuva_acida,
        criar_risco_smog,
        criar_risco_efeito_estufa,
        mostrar_amostra
    )
    # Renomeando a função importada para evitar conflito
    from .fetch_weather import get_weather as get_weather_from_city
except ImportError:
    print("Aviso: Módulos de pré-processamento não encontrados. O endpoint /preprocess pode falhar.")


# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# --- Configuração da Aplicação FastAPI ---
app = FastAPI(
    title="API de Qualidade Ambiental",
    description="Prevê a qualidade do ar e riscos de fenômenos naturais com base em dados atmosféricos.",
    version="1.1.0"
)

# --- Carregamento do Modelo com Joblib ---
MODEL = None
MODEL_URI = os.path.join(
    os.path.dirname(__file__),
    "outputs",
    "predictions",
    "RandomForest_MultiOutput_Robusto_compressed.pkl"
)

@app.on_event('startup')
def load_model():
    """Carrega o modelo .pkl ao iniciar o FastAPI"""
    global MODEL
    if not os.path.exists(MODEL_URI):
        print(f"ERRO CRÍTICO: Arquivo do modelo não encontrado em '{MODEL_URI}'")
        return

    try:
        MODEL = joblib.load(MODEL_URI)
        print(f"Modelo carregado com sucesso de '{MODEL_URI}'")
    except Exception as e:
        print(f"ERRO CRÍTICO: Não foi possível carregar o modelo no startup: {e}")
        MODEL = None

# --- Modelos de Dados (Pydantic) ---

class FeaturesInput(BaseModel):
    Temperatura: float = Field(..., example=25.5)
    Umidade: float = Field(..., example=60.2)
    CO2: float = Field(..., example=415.5)
    CO: float = Field(..., example=5.0)
    Pressao_Atm: float = Field(..., example=1012.5)
    NO2: float = Field(..., example=20.1)
    SO2: float = Field(..., example=15.3)
    O3: float = Field(..., example=30.8)

class ManualPredictionPayload(BaseModel):
    features: FeaturesInput

class CityInput(BaseModel):
    city: str = Field(..., example="São Paulo")

class PredictionResult(BaseModel):
    Qualidade_Ambiental: str
    Risco_Chuva_Acida: str
    Risco_Smog_Fotoquimico: str
    Risco_Efeito_Estufa: str

class PredictionResponse(BaseModel):
    prediction: PredictionResult

# --- Funções Auxiliares ---

def get_openweather_data(city: str, api_key: str) -> Dict[str, Any]:
    # (Esta é a função da minha sugestão anterior, que é mais robusta)
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={api_key}"
    geo_response = requests.get(geo_url)
    if geo_response.status_code != 200 or not geo_response.json():
        raise HTTPException(status_code=404, detail=f"Cidade '{city}' não encontrada.")
    
    geo_data = geo_response.json()[0]
    lat, lon = geo_data['lat'], geo_data['lon']

    air_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    air_response = requests.get(air_url)
    if air_response.status_code != 200:
        raise HTTPException(status_code=500, detail="Erro ao buscar dados de poluição.")
    air_data = air_response.json()['list'][0]['components']

    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    weather_response = requests.get(weather_url)
    if weather_response.status_code != 200:
        raise HTTPException(status_code=500, detail="Erro ao buscar dados de clima.")
    weather_data = weather_response.json()['main']

    return {
        "Temperatura": weather_data.get('temp', 25.0),
        "Umidade": weather_data.get('humidity', 50.0),
        "CO2": 418.0,
        "CO": air_data.get('co', 0.0),
        "Pressao_Atm": weather_data.get('pressure', 1013.0),
        "NO2": air_data.get('no2', 0.0),
        "SO2": air_data.get('so2', 0.0),
        "O3": air_data.get('o3', 0.0)
    }

def format_prediction(pred: list) -> dict:
    # Usando os mapeamentos do seu código original
    qualidade_mapping = {0: 'Muito Ruim', 1: 'Ruim', 2: 'Moderada', 3: 'Boa', 4: 'Excelente'}
    risco_mapping = {0: 'Não', 1: 'Sim'}
    
    # pred[0] contém o array com as 4 saídas
    raw_preds = pred[0]

    return {
        "Qualidade_Ambiental": qualidade_mapping.get(raw_preds[0], 'Desconhecido'),
        "Risco_Chuva_Acida": risco_mapping.get(raw_preds[1], 'Desconhecido'),
        "Risco_Smog_Fotoquimico": risco_mapping.get(raw_preds[2], 'Desconhecido'),
        "Risco_Efeito_Estufa": risco_mapping.get(raw_preds[3], 'Desconhecido')
    }

# --- Endpoints da API ---

@app.get("/", summary="Endpoint raiz da API")
def read_root():
    model_status = "Carregado" if MODEL is not None else "Não Carregado"
    return {"status": "API de Qualidade Ambiental está online", "model_status": model_status}

@app.post("/predict-features", response_model=PredictionResponse, summary="Prevê com dados manuais")
async def predict_with_features(payload: ManualPredictionPayload):
    if MODEL is None:
        raise HTTPException(status_code=503, detail=f"Modelo não carregado. Verifique o caminho: {MODEL_URI}")
        
    try:
        features_dict = payload.features.dict()
        
        # O seu modelo original parece esperar uma lista de listas, sem cabeçalho.
        # A ordem das features importa! Vamos garantir a mesma ordem sempre.
        # Seu código original usava `sorted(keys)`, vamos replicar isso.
        required_keys = sorted(features_dict.keys())
        X = [[features_dict[k] for k in required_keys]]
        
        pred = MODEL.predict(X)
        
        # Formata a resposta com seus mapeamentos e estrutura
        formatted_response = format_prediction(pred.tolist())
        return {"prediction": formatted_response}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro durante a predição: {str(e)}")

@app.post("/predict-city", response_model=PredictionResponse, summary="Prevê com nome da cidade")
async def predict_with_city(payload: CityInput):
    if MODEL is None:
        raise HTTPException(status_code=503, detail=f"Modelo não carregado. Verifique o caminho: {MODEL_URI}")

    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Chave da API OpenWeather não configurada no servidor.")

    try:
        features_dict = get_openweather_data(payload.city, api_key)
        
        required_keys = sorted(features_dict.keys())
        X = [[features_dict[k] for k in required_keys]]
        
        pred = MODEL.predict(X)
        
        formatted_response = format_prediction(pred.tolist())
        return {"prediction": formatted_response}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro inesperado: {str(e)}")

# -----------------------------
# Endpoint para testar pipeline de dados (do seu código original)
# -----------------------------
@app.get('/preprocess', summary="Testa o pipeline de processamento de dados")
def preprocess_data():
    try:
        df = load_data()
        print("\n=== Exploração inicial ===")
        exploracao_df(df)
        df = codificar_qualidade_ambiental(df)        
        df = tratar_pressao_atm(df)
        df = balancear_qualidade_ambiental(df)
        df = criar_risco_chuva_acida(df)
        df = criar_risco_smog(df)
        df = criar_risco_efeito_estufa(df) # uvicorn src.app:app --reload --port 8000 --app-dir .

        mostrar_amostra(df, n=10)
        return {"status": "pipeline executado com sucesso", "linhas": len(df)}
    except NameError:
         raise HTTPException(status_code=501, detail="Módulos de pré-processamento não foram carregados. Endpoint desabilitado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))