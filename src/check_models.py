# src/check_models.py
import os

def listar_modelos(base_dir="mlruns"):
    """
    Lista todos os modelos salvos dentro da pasta mlruns
    e mostra o tamanho em MB de cada arquivo model.pkl.
    """
    if not os.path.exists(base_dir):
        print(f"Pasta '{base_dir}' não encontrada.")
        return

    print(f"\n=== Verificando modelos em '{base_dir}' ===\n")

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("model.pkl"):  # Arquivo real do modelo Para rodar no terminal: python .\src\check_models.py
                caminho = os.path.join(root, file)
                tamanho_mb = os.path.getsize(caminho) / (1024 * 1024)
                print(f"📂 Modelo: {caminho} | 💾 Tamanho: {tamanho_mb:.2f} MB")

if __name__ == "__main__":
    listar_modelos()
