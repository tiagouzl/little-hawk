"""
utils/helpers.py — Funções auxiliares (sem I/O de rede).
"""

import os
from pathlib import Path


def ensure_dir(path: str) -> None:
    """Garante que o diretório existe"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def format_bytes(size: int) -> str:
    """Formata bytes em unidade legível"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TB"


def validate_weights_file(path: str) -> tuple[bool, str]:
    """Valida se arquivo de pesos existe e é válido"""
    if not os.path.exists(path):
        return False, f"Arquivo não encontrado: {path}"

    if not path.endswith(".npz"):
        return False, "Arquivo deve ter extensão .npz"

    meta_path = path.replace(".npz", "_meta.json")
    if not os.path.exists(meta_path):
        return False, f"Arquivo meta não encontrado: {meta_path}"

    return True, "OK"
