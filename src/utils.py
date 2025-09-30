"""utils.py
Helpers úteis, transformações e funções reutilizáveis.
"""
import json
def save_json(obj, path):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
