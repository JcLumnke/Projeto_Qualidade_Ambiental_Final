# src/data_preparation.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_data(df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42):
    """
    Divide o dataset em treino e teste com múltiplos targets.
    Estratifica pelo target 'Qualidade_Ambiental' para manter equilíbrio de classes.

    Retorna:
    - X_train, X_test, y_train, y_test
    """
    target_cols = [
        'Qualidade_Ambiental',
        'Risco_Chuva_Acida',
        'Risco_Smog_Fotoquimico',
        'Risco_Efeito_Estufa'
    ]

    # Features e targets
    X = df.drop(columns=target_cols)
    y = df[target_cols]

    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=df['Qualidade_Ambiental']
    )

    print("▶️ Tamanhos dos conjuntos:")
    print("   Treino:", X_train.shape, "Teste:", X_test.shape)
    print("   y_train:", y_train.shape, "y_test:", y_test.shape)
    print("="*70)

    return X_train, X_test, y_train, y_test


def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Aplica padronização (StandardScaler) apenas nas colunas numéricas.
    
    Retorna:
    - X_train_scaled, X_test_scaled, scaler
    """
    numeric_cols = X_train.select_dtypes(include="number").columns
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    print("▶️ Padronização concluída (apenas variáveis numéricas).")
    print("="*70)

    return X_train_scaled, X_test_scaled, scaler


# Execução direta pelo terminal (opcional)
if __name__ == "__main__":
    from data_processing import load_data
    from data_cleaning import (
        codificar_qualidade_ambiental,
        tratar_pressao_atm,
        balancear_qualidade_ambiental,
        criar_risco_chuva_acida,
        criar_risco_smog,
        criar_risco_efeito_estufa
    )

    # 1. Carregar dados
    df = load_data()

    # 2. Aplicar transformações
    df = codificar_qualidade_ambiental(df)
    df = tratar_pressao_atm(df)
    df = balancear_qualidade_ambiental(df)
    df = criar_risco_chuva_acida(df)
    df = criar_risco_smog(df)
    df = criar_risco_efeito_estufa(df)

    # 3. Separar dados
    X_train, X_test, y_train, y_test = split_data(df)

    # 4. Padronizar
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # Mensagem de confirmação Para rodar no terminal: python .\src\data_preparation.py
    print("✅ Pré-processamento concluído. Variáveis prontas para treino.")

