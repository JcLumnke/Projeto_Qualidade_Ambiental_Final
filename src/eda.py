# src/eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

def exploracao_df(df: pd.DataFrame):
    """
    Realiza uma exploração inicial de um DataFrame do Pandas,
    exibindo informações essenciais com explicações.
    """
    print("====================================================================")
    print("               INICIANDO ANÁLISE EXPLORATÓRIA DO DATAFRAME")
    print("====================================================================\n")

    # --- df.head() ---
    print("▶️ AMOSTRA DOS DADOS (PRIMEIRAS 5 LINHAS):")
    print("   Mostra as primeiras linhas do DataFrame para uma inspeção visual rápida dos dados.\n")
    print(df.head(), "\n")
    print("="*70, "\n")

    # --- df.columns ---
    print("▶️ NOMES DAS COLUNAS:")
    print("   Exibe todos os rótulos (nomes) das colunas presentes no DataFrame.\n")
    print(list(df.columns), "\n")
    print("="*70, "\n")

    # --- df.dtypes ---
    print("▶️ TIPOS DE DADOS POR COLUNA:")
    print("   Informa o tipo de dado de cada coluna (ex: int64, float64, object para texto).\n")
    print(df.dtypes, "\n")
    print("="*70, "\n")

    # --- df.shape ---
    print("▶️ DIMENSÕES DO DATAFRAME (LINHAS E COLUNAS):")
    print("   Retorna uma tupla representando as dimensões do DataFrame (número_de_linhas, número_de_colunas).\n")
    print(df.shape, "\n")
    print("="*70, "\n")

    # --- df.info() ---
    print("▶️ INFORMAÇÕES GERAIS DO DATAFRAME:")
    print("   Fornece um resumo conciso, incluindo o tipo de índice, colunas, contagem de valores não-nulos e uso de memória.\n")
    df.info()
    print("\n" + "="*70 + "\n")

    # --- df.isnull().sum() ---
    print("▶️ CONTAGEM DE VALORES NULOS (AUSENTES) POR COLUNA:")
    print("   Soma a quantidade de valores nulos (NaN) em cada coluna. Essencial para limpeza de dados.\n")
    print(df.isnull().sum(), "\n")
    print("="*70, "\n")

    # --- VERIFICAÇÃO DE NEGATIVOS ---
    print("▶️ VERIFICAÇÃO DE VALORES NEGATIVOS:")
    print("   Verifica as colunas numéricas para identificar a presença de valores negativos.\n")
    df_numerico = df.select_dtypes(include=np.number)
    negativos_encontrados = False
    if df_numerico.empty:
        print("   - Não há colunas numéricas para verificar.\n")
    else:
        for coluna in df_numerico.columns:
            contagem_negativos = (df_numerico[coluna] < 0).sum()
            if contagem_negativos > 0:
                print(f"   - Alerta! Coluna '{coluna}': Encontrado(s) {contagem_negativos} valor(es) negativo(s).")
                negativos_encontrados = True
        if not negativos_encontrados:
            print("   - Nenhuma coluna numérica com valores negativos foi encontrada.\n")
    print("="*70, "\n")

    # --- df.describe() ---
    print("▶️ RESUMO ESTATÍSTICO DAS COLUNAS NUMÉRICAS:")
    print("   Gera estatísticas descritivas como contagem, média, desvio padrão, mínimo, máximo e quartis.\n")
    print(df.describe(), "\n")
    print("="*70, "\n")

    print("====================================================================")
    print("                          FIM DA ANÁLISE")
    print("====================================================================\n")


# ==========================
# Funções para geração de gráficos
# ==========================
def gerar_graficos(df):
    num_cols = ['Temperatura', 'Umidade', 'CO2', 'CO', 'Pressao_Atm', 'NO2', 'SO2', 'O3']

    mlflow.set_experiment("analise_exploratoria_ambiental")

    # 🔹 Histogramas
    with mlflow.start_run(run_name="distribuicao_variaveis"):
        for col in num_cols:
            plt.figure(figsize=(6,4))
            sns.histplot(df[col], kde=True, bins=30, color='skyblue')
            plt.title(f"Distribuição de {col}")
            plt.xlabel(col)
            plt.ylabel("Frequência")
            plt.tight_layout()

            filename = f"{col}_distribuicao.png"
            plt.savefig(filename, dpi=100)
            mlflow.log_artifact(filename)
            plt.show()
            plt.close()

        print("▶️ Distribuições geradas e salvas no MLflow.")

    # 🔹 Boxplot
    with mlflow.start_run(run_name="boxplots_variaveis"):
        plt.figure(figsize=(10,5))
        sns.boxplot(data=df[num_cols])
        plt.xticks(rotation=45)
        plt.title("Boxplot das variáveis numéricas")
        plt.tight_layout()

        boxplot_path = "boxplot_variaveis.png"
        plt.savefig(boxplot_path, dpi=100)
        mlflow.log_artifact(boxplot_path)
        plt.show()
        plt.close()

        print("▶️ Boxplot gerado e salvo no MLflow.")

    # 🔹 Heatmap de correlação
    with mlflow.start_run(run_name="correlacao_variaveis"):
        corr = df[num_cols].corr()
        plt.figure(figsize=(8,6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
        plt.title("Mapa de Correlação")
        plt.tight_layout()

        corr_path = "correlacao_variaveis.png"
        plt.savefig(corr_path, dpi=100)
        mlflow.log_artifact(corr_path)
        plt.show()
        plt.close()

        print("▶️ Heatmap de correlação gerado e salvo no MLflow.")


# ==========================
# Bloco principal
# ==========================
if __name__ == "__main__":
    from .data_processing import load_data
    df = load_data()  # Para rodar no terminal: python -m src.eda

    exploracao_df(df)
    gerar_graficos(df)
