# src/train.py

import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend não interativo, gera imagens sem abrir janela
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# Importar funções de preparação de dados
from data_preparation import split_data, scale_data
from data_processing import load_data
from data_cleaning import (
    codificar_qualidade_ambiental,
    tratar_pressao_atm,
    balancear_qualidade_ambiental,
    criar_risco_chuva_acida,
    criar_risco_smog,
    criar_risco_efeito_estufa
)

# ==================================================
# Preparação de pastas
# ==================================================
os.makedirs("outputs/confusion_matrices", exist_ok=True)
os.makedirs("outputs/predictions", exist_ok=True)

# ==================================================
# Labels e targets
# ==================================================
target_cols = ['Qualidade_Ambiental', 'Risco_Chuva_Acida',
               'Risco_Smog_Fotoquimico', 'Risco_Efeito_Estufa']

inv_mapping_qa = {0: 'Muito Ruim', 1: 'Ruim', 2: 'Moderada',
                  3: 'Boa', 4: 'Excelente'}

# ==================================================
# Carregar e preparar dados
# ==================================================
df = load_data()
df = codificar_qualidade_ambiental(df)
df = tratar_pressao_atm(df)
df = balancear_qualidade_ambiental(df)
df = criar_risco_chuva_acida(df)
df = criar_risco_smog(df)
df = criar_risco_efeito_estufa(df)

# ⚠️ Checar NaNs
if df.isna().sum().sum() > 0:
    print("⚠️ Atenção: Existem valores NaN no dataset após limpeza!")
    print(df.isna().sum())
    # Opcional: preencher NaNs com média para colunas numéricas
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    print("✅ NaNs preenchidos com a média nas colunas numéricas.")

# Separar treino/teste
X_train, X_test, y_train, y_test = split_data(df)
# Padronizar
X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

# ==================================================
# Treino - SVM Multilabel
# ==================================================
mlflow.set_experiment("miruns")

for col in target_cols:
    print(f"\n=== Treinando modelo SVM para {col} ===")

    X_train_col = X_train_scaled
    X_test_col = X_test_scaled
    y_train_col = y_train[col].values
    y_test_col = y_test[col].values

    model_name = f"SVM_{col}"
    with mlflow.start_run(run_name=model_name):

        model = SVC(kernel='rbf', probability=True, random_state=42)
        model.fit(X_train_col, y_train_col)

        y_pred = model.predict(X_test_col)

        # Métricas
        acc = accuracy_score(y_test_col, y_pred)
        f1 = f1_score(y_test_col, y_pred, average='weighted', zero_division=1)
        precision = precision_score(y_test_col, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test_col, y_pred, average='weighted', zero_division=1)

        print(f"Acurácia: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.sklearn.log_model(model, "model")

        # Matriz de confusão
        cm = confusion_matrix(y_test_col, y_pred)
        print("Matriz de Confusão:")
        print(cm)

        labels = list(inv_mapping_qa.values()) if col == "Qualidade_Ambiental" else [0, 1]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Matriz de Confusão - {col}")
        plot_path = os.path.join("outputs/confusion_matrices", f"svm_confusion_matrix_{col}.png")
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)

        # Salvar previsões
        resultado = pd.DataFrame({"Real": y_test_col, "Predito": y_pred})
        if col == 'Qualidade_Ambiental':
            resultado["Real_Label"] = resultado["Real"].map(inv_mapping_qa)
            resultado["Predito_Label"] = resultado["Predito"].map(inv_mapping_qa)
        resultado.to_csv(os.path.join("outputs/predictions", f"svm_preds_{col}.csv"), index=False)

# ==================================================
# Treino - Random Forest MultiOutput
# ==================================================
model_name = "RandomForest_MultiOutput_Robusto"
mlflow.set_experiment("miruns")

with mlflow.start_run(run_name=model_name):

    base_rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    multi_rf = MultiOutputClassifier(base_rf, n_jobs=-1)
    multi_rf.fit(X_train, y_train)

    y_pred = multi_rf.predict(X_test)

    # Loop para métricas de cada target
    for idx, col in enumerate(y_train.columns):
        y_true_col = y_test[col].values
        y_pred_col = y_pred[:, idx]

        acc = accuracy_score(y_true_col, y_pred_col)
        f1 = f1_score(y_true_col, y_pred_col, average='weighted', zero_division=1)
        precision = precision_score(y_true_col, y_pred_col, average='weighted', zero_division=1)
        recall = recall_score(y_true_col, y_pred_col, average='weighted', zero_division=1)

        print(f"\n=== Métricas RF para {col} ===")
        print(f"Acurácia: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        mlflow.log_metric(f"{col}_accuracy", acc)
        mlflow.log_metric(f"{col}_f1", f1)
        mlflow.log_metric(f"{col}_precision", precision)
        mlflow.log_metric(f"{col}_recall", recall)

        # Matriz de confusão
        cm = confusion_matrix(y_true_col, y_pred_col)
        print(f"Matriz de Confusão - {col}:")
        print(cm)

        labels = list(inv_mapping_qa.values()) if col == "Qualidade_Ambiental" else np.unique(y_true_col)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Matriz de Confusão - {col}")
        plot_path = os.path.join("outputs/confusion_matrices", f"rf_confusion_matrix_{col}.png")
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)

        # Salvar previsões individuais
        resultado = pd.DataFrame({"Real": y_true_col, "Predito": y_pred_col})
        if col == 'Qualidade_Ambiental':
            resultado["Real_Label"] = resultado["Real"].map(inv_mapping_qa)
            resultado["Predito_Label"] = resultado["Predito"].map(inv_mapping_qa)
        resultado.to_csv(os.path.join("outputs/predictions", f"rf_preds_{col}.csv"), index=False)

    # Log modelo no MLflow
    mlflow.sklearn.log_model(multi_rf, "model")

    # Salvar modelo RF localmente
    import joblib
    rf_path = os.path.join("outputs/predictions", "RandomForest_MultiOutput_Robusto.pkl")
    joblib.dump(multi_rf, rf_path)
    print(f"✅ Random Forest salvo em {rf_path}") # Para rodar no terminal: python .\src\train.py
