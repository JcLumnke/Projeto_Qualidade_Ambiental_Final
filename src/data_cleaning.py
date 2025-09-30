# src/data_cleaning.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow


# -----------------------------
# FUNÇÕES DE TRANSFORMAÇÃO
# -----------------------------

def explorar_qualidade_ambiental(df: pd.DataFrame):
    """Exibe contagem das categorias da coluna 'Qualidade_Ambiental'."""
    print("▶️ CONTAGEM DE VALORES NA COLUNA 'Qualidade_Ambiental':")
    print(df['Qualidade_Ambiental'].value_counts())
    print("="*70)


def codificar_qualidade_ambiental(df: pd.DataFrame) -> pd.DataFrame:
    """Converte categorias da coluna 'Qualidade_Ambiental' em inteiros."""
    qualidade_mapping = {
        'Muito Ruim': 0,
        'Ruim': 1,
        'Moderada': 2,
        'Boa': 3,
        'Excelente': 4
    }
    df['Qualidade_Ambiental'] = df['Qualidade_Ambiental'].map(qualidade_mapping)
    print("▶️ Coluna 'Qualidade_Ambiental' codificada:")
    print(df[['Qualidade_Ambiental']].head())
    print("="*70)
    return df


def tratar_pressao_atm(df: pd.DataFrame) -> pd.DataFrame:
    """Converte 'Pressao_Atm' para float e remove linhas inválidas."""
    df['Pressao_Atm'] = pd.to_numeric(df['Pressao_Atm'], errors='coerce')
    df = df.dropna(subset=['Pressao_Atm'])
    print("▶️ Dimensões após remover nulos em 'Pressao_Atm':", df.shape)
    print("="*70)
    return df

# -----------------------------
# REMOVER NANS EM TEMPERATURA E UMIDADE
# -----------------------------
def remover_nans_temperatura_umidade(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas com NaNs nas colunas 'Temperatura' e 'Umidade'."""
    df = df.dropna(subset=['Temperatura', 'Umidade'])
    print("▶️ Dimensões após remover NaNs em 'Temperatura' e 'Umidade':", df.shape)
    print("="*70)
    return df

def balancear_qualidade_ambiental(df: pd.DataFrame) -> pd.DataFrame:
    """Redistribui amostras da classe 2 para classes 0 e 4."""
    df_0 = df[df['Qualidade_Ambiental'] == 0]
    df_1 = df[df['Qualidade_Ambiental'] == 1]
    df_2 = df[df['Qualidade_Ambiental'] == 2]
    df_3 = df[df['Qualidade_Ambiental'] == 3]
    df_4 = df[df['Qualidade_Ambiental'] == 4]

    df_2_extra = df_2.sample(3000, random_state=42)
    df_2_rest = df_2.drop(df_2_extra.index)

    df_0_aug = pd.concat([df_0, df_2_extra.sample(1000, random_state=42).assign(Qualidade_Ambiental=0)])
    df_4_aug = pd.concat([df_4, df_2_extra.drop(df_2_extra.sample(1500, random_state=42).index).assign(Qualidade_Ambiental=4)])

    df_balanced = pd.concat([df_0_aug, df_1, df_2_rest, df_3, df_4_aug])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Distribuição após redistribuição 'Qualidade_Ambiental':")
    print(df_balanced['Qualidade_Ambiental'].value_counts())
    print("="*70)
    return df_balanced


def criar_risco_chuva_acida(df: pd.DataFrame) -> pd.DataFrame:
    """Cria coluna 'Risco_Chuva_Acida' e realiza balanceamento."""
    limiar_no2 = np.percentile(df["NO2"], 65)
    limiar_so2 = np.percentile(df["SO2"], 65)
    limiar_umidade = np.percentile(df["Umidade"], 75)

    df["Risco_Chuva_Acida"] = np.where(
        (df["NO2"] > limiar_no2) & (df["SO2"] > limiar_so2) & (df["Umidade"] > limiar_umidade),
        1, 0
    )

    # Balanceamento
    df_0 = df[df['Risco_Chuva_Acida'] == 0]
    df_1 = df[df['Risco_Chuva_Acida'] == 1]
    n_add_1 = 1310 - len(df_1)
    df.loc[df_0.sample(n=n_add_1, random_state=42).index, 'Risco_Chuva_Acida'] = 1

    print("Distribuição 'Risco_Chuva_Acida':")
    print(df['Risco_Chuva_Acida'].value_counts())
    print("="*70)
    return df


def criar_risco_smog(df: pd.DataFrame) -> pd.DataFrame:
    """Cria coluna 'Risco_Smog_Fotoquimico' e realiza balanceamento."""
    NO2_lim = df['NO2'].quantile(0.7)
    O3_lim = df['O3'].quantile(0.7)
    Temp_lim = df['Temperatura'].quantile(0.7)

    df['Risco_Smog_Fotoquimico'] = ((df['NO2'] > NO2_lim) &
                                    (df['O3'] > O3_lim) &
                                    (df['Temperatura'] > Temp_lim)).astype(int)

    df_0 = df[df['Risco_Smog_Fotoquimico'] == 0]
    df_1 = df[df['Risco_Smog_Fotoquimico'] == 1]
    n_to_add = 1260 - len(df_1)
    df_1_extra = df_0.sample(n_to_add, random_state=42).copy()
    df_1_extra['Risco_Smog_Fotoquimico'] = 1
    df = pd.concat([df.drop(df_1_extra.index), df_1, df_1_extra], ignore_index=True)

    print("Distribuição 'Risco_Smog_Fotoquimico':")
    print(df['Risco_Smog_Fotoquimico'].value_counts())
    print("="*70)
    return df


def criar_risco_efeito_estufa(df: pd.DataFrame) -> pd.DataFrame:
    """Cria coluna 'Risco_Efeito_Estufa' combinando CO2, Temperatura e Umidade."""
    CO2_lim = df['CO2'].quantile(0.7)
    Temp_lim = df['Temperatura'].quantile(0.7)
    Umid_lim = df['Umidade'].quantile(0.7)

    df['Risco_Efeito_Estufa'] = (
        ((df['CO2'] > CO2_lim).astype(int) +
         (df['Temperatura'] > Temp_lim).astype(int) +
         (df['Umidade'] > Umid_lim).astype(int)) >= 2
    ).astype(int)

    print("Distribuição 'Risco_Efeito_Estufa':")
    print(df['Risco_Efeito_Estufa'].value_counts())
    print("="*70)
    return df


# -----------------------------
# FUNÇÃO DE GRÁFICOS PÓS-TRATAMENTO
# -----------------------------

def gerar_graficos_pos_tratamento(df: pd.DataFrame, salvar_pasta: str = "miruns"):
    """Gera gráficos pós-tratamento das colunas de risco e matriz de correlação."""
    os.makedirs(salvar_pasta, exist_ok=True)
    colunas_risco = ['Risco_Chuva_Acida', 'Risco_Smog_Fotoquimico', 'Risco_Efeito_Estufa']

    with mlflow.start_run(run_name="Distribuicoes_e_Correlacoes"):
        # Distribuição das variáveis
        for col in colunas_risco:
            plt.figure(figsize=(6,4))
            sns.countplot(data=df, x=col, palette='pastel')
            plt.title(f"Distribuição de {col}")
            plt.tight_layout()
            path = os.path.join(salvar_pasta, f"{col}_distribuicao.png")
            plt.savefig(path)
            plt.show()
            plt.close()
            mlflow.log_artifact(path)

        # Matriz de correlação
        plt.figure(figsize=(10,8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Matriz de Correlação")
        plt.tight_layout()
        corr_path = os.path.join(salvar_pasta, "matriz_correlacao.png")
        plt.savefig(corr_path)
        plt.show()
        plt.close()
        mlflow.log_artifact(corr_path)

    print("▶️ Gráficos de distribuição e matriz de correlação gerados e salvos no MLflow.")


# -----------------------------
# EXECUÇÃO PRINCIPAL (TESTE)
# -----------------------------

if __name__ == "__main__":
    from .data_processing import load_data

    df = load_data()

    # Remover NaNs em Temperatura e Umidade
    df = remover_nans_temperatura_umidade(df)

    # Encadeamento das transformações
    df = (
        df
        .pipe(codificar_qualidade_ambiental)
        .pipe(tratar_pressao_atm)
        .pipe(balancear_qualidade_ambiental)
        .pipe(criar_risco_chuva_acida)
        .pipe(criar_risco_smog)
        .pipe(criar_risco_efeito_estufa)
    )

    # Gerar gráficos pós-tratamento
    gerar_graficos_pos_tratamento(df) # Para rodar no terminal: python -m src.data_cleaning
