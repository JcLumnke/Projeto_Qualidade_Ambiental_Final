# src/clean_miruns.py
import shutil
import os

def limpar_mlruns(base_dir="mlruns"):
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            caminho = os.path.join(base_dir, item)
            if os.path.isdir(caminho):
                shutil.rmtree(caminho)
            else:
                os.remove(caminho) # Para rodar no terminal: python .\src\clean_miruns.py
        print(f"✅ Pasta '{base_dir}' limpa com sucesso.")
    else:
        print(f"⚠️ Pasta '{base_dir}' não encontrada.")

if __name__ == "__main__":
    limpar_mlruns()
