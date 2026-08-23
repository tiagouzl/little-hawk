# scripts/download_weights.py
"""
Script utilitário para baixar pesos de modelos HuggingFace.
"""
from huggingface_hub import hf_hub_download
import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python download_weights.py <repo_id> <filename>")
        sys.exit(1)
    repo_id, filename = sys.argv[1], sys.argv[2]
    path = hf_hub_download(repo_id, filename)
    print(f"Baixado para: {path}")
