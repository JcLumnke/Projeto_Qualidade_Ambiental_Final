# src/data_processing.py

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

# Caminho relativo para o arquivo CSV
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset_ambiental.csv')


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Carrega o dataset do CSV e retorna um DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"O arquivo CSV não foi encontrado no caminho: {path}")
    
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica etapas de limpeza (ajuste conforme seu notebook)."""
    # Exemplo: remover valores nulos
    df = df.dropna()
    return df


def split_data(
    df: pd.DataFrame, target: str = "Qualidade_Ambiental", test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa features e target, divide em treino e teste."""
    if target not in df.columns:
        raise ValueError(f"A coluna alvo '{target}' não existe no DataFrame.")
    
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


if __name__ == "__main__":
    # Teste rápido no terminal
    print(f"Caminho do CSV usado: {DATA_PATH}")
    df = load_data()
    print("Pré-visualização dos dados:")
    print(df.head())
    print(f"Número de linhas e colunas: {df.shape}")
    print(f"Número de valores nulos por coluna:\n{df.isnull().sum()}")
    # Para rodar noterminal: python .\src\data_processing.py
