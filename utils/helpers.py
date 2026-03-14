#!/usr/bin/env python3
"""
utils/helpers.py — Funções auxiliares
"""
import os
import json
from pathlib import Path
from typing import Optional, Tuple

def ensure_dir(path: str) -> None:
    """Garante que o diretório existe"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def find_file_in_cache(model_id: str, filename: str) -> Optional[str]:
    """Procura arquivo no cache do HuggingFace"""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_files(model_id)
        if filename in files:
            return api.hf_hub_download(model_id, filename)
    except ImportError:
        pass
    return None

def load_json_safe(path: str) -> Optional[dict]:
    """Carrega JSON com tratamento de erro"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def save_json_safe(path: str, data: dict) -> bool:
    """Salva JSON com tratamento de erro"""
    try:
        ensure_dir(path)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def format_bytes(size: int) -> str:
    """Formata bytes em unidade legível"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TB"

def validate_weights_file(path: str) -> Tuple[bool, str]:
    """Valida se arquivo de pesos existe e é válido"""
    if not os.path.exists(path):
        return False, f"Arquivo não encontrado: {path}"

    if not path.endswith('.npz'):
        return False, "Arquivo deve ter extensão .npz"

    meta_path = path.replace('.npz', '_meta.json')
    if not os.path.exists(meta_path):
        return False, f"Arquivo meta não encontrado: {meta_path}"

    return True, "OK"