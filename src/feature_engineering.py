# src/feature_engineering.py
import pandas as pd
import numpy as np

def criar_risco_chuva_acida(df: pd.DataFrame) -> pd.DataFrame:
    # exemplo de implementação
    df['Risco_Chuva_Acida'] = ((df['NO2'] > df['NO2'].quantile(0.65)) &
                               (df['SO2'] > df['SO2'].quantile(0.65)) &
                               (df['Umidade'] > df['Umidade'].quantile(0.75))).astype(int)
    return df

def criar_risco_smog(df: pd.DataFrame) -> pd.DataFrame:
    df['Risco_Smog_Fotoquimico'] = ((df['NO2'] > df['NO2'].quantile(0.7)) &
                                    (df['O3'] > df['O3'].quantile(0.7)) &
                                    (df['Temperatura'] > df['Temperatura'].quantile(0.7))).astype(int)
    return df

def criar_risco_efeito_estufa(df: pd.DataFrame) -> pd.DataFrame:
    df['Risco_Efeito_Estufa'] = (((df['CO2'] > df['CO2'].quantile(0.7)).astype(int) +
                                  (df['Temperatura'] > df['Temperatura'].quantile(0.7)).astype(int) +
                                  (df['Umidade'] > df['Umidade'].quantile(0.7)).astype(int)) >= 2).astype(int)
    return df

def mostrar_amostra(df: pd.DataFrame, n=5):
    print(df.head(n))
