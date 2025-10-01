import os
import requests
from dotenv import load_dotenv

# Carrega a chave do .env
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

def get_weather(city_name: str):
    """
    Retorna os dados de clima necessários para o modelo a partir do OpenWeather.
    Alguns poluentes são valores fixos se não houver dados reais.
    """
    params = {
        "q": city_name,
        "appid": API_KEY,
        "units": "metric"
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()

    features = {
        "Temperatura": data["main"]["temp"],
        "Umidade": data["main"]["humidity"],
        "CO2": 19998.31,
        "CO": 48.22,
        "Pressao_Atm": data["main"]["pressure"],
        "NO2": 23.57,
        "SO2": 23.69,
        "O3": 37.42
    }
    return features