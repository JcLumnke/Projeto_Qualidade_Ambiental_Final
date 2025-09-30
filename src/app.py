# src/app.py
"""FastAPI app para servir o modelo salvo com MLflow e testar pipeline de dados"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
import joblib

# -----------------------------
# Imports relativos corretos
# -----------------------------
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
from .fetch_weather import get_weather  # Novo import para OpenWeather

# -----------------------------
# Modelo MLflow
# -----------------------------
class PredictRequest(BaseModel):
    features: dict = None   # Para envio manual
    city: str = None        # Para OpenWeather

app = FastAPI(title='API de predição')

# Atualize para o modelo compactado
MODEL_URI = "outputs/predictions/RandomForest_MultiOutput_Robusto_compressed.pkl"
MODEL = None

@app.on_event('startup')
def load_model():
    """Carrega modelo MLflow ao iniciar o FastAPI"""
    global MODEL
    try:
        MODEL = joblib.load(MODEL_URI)
        print('Modelo carregado de', MODEL_URI)
    except Exception as e:
        print('Aviso: não foi possível carregar o modelo no startup:', e)
        MODEL = None

@app.get('/')
def root():
    return {'status': 'ok', 'model_uri': MODEL_URI}

# -----------------------------
# Endpoint de predição atualizado
# -----------------------------
@app.post('/predict')
def predict(req: PredictRequest):
    """
    Recebe features manualmente ou uma cidade para puxar via OpenWeather.
    Retorna a predição do modelo com labels legíveis.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail='Modelo não carregado')
    try:
        if req.city:
            feat_dict = get_weather(req.city)
        elif req.features:
            feat_dict = req.features
        else:
            raise HTTPException(status_code=400, detail='Envie "features" ou "city"')

        keys = sorted(feat_dict.keys())
        X = [[feat_dict[k] for k in keys]]
        pred = MODEL.predict(X)

        if hasattr(pred, 'tolist'):
            pred = pred.tolist()

        # Mapear valores numéricos para labels
        qualidade_mapping = {0: 'Muito Ruim', 1: 'Ruim', 2: 'Moderada', 3: 'Boa', 4: 'Excelente'}
        risco_mapping = {0: 'Não', 1: 'Sim'}

        response = {
            "Qualidade_Ambiental": qualidade_mapping[pred[0][0]],
            "Risco_Chuva_Acida": risco_mapping[pred[0][1]],
            "Risco_Smog_Fotoquimico": risco_mapping[pred[0][2]],
            "Risco_Efeito_Estufa": risco_mapping[pred[0][3]]
        }

        return {"prediction": response}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -----------------------------
# Endpoint para testar pipeline de dados
# -----------------------------
@app.get('/preprocess')
def preprocess_data():
    """
    Executa pipeline completo de exploração, tratamento e criação de features.
    Retorna quantidade de linhas processadas.
    """
    try:
        df = load_data()
        print("\n=== Exploração inicial ===")
        exploracao_df(df)

        # Limpeza e transformação
        df = codificar_qualidade_ambiental(df)
        df = tratar_pressao_atm(df)
        df = balancear_qualidade_ambiental(df)

        # Engenharia de features
        df = criar_risco_chuva_acida(df)
        df = criar_risco_smog(df)
        df = criar_risco_efeito_estufa(df)

        # Mostrar amostra final. Para iniciar a FastAPI: uvicorn src.app:app --reload
        mostrar_amostra(df, n=10)

        return {"status": "pipeline executado com sucesso", "linhas": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
