#!/usr/bin/env python3
"""
utils/config.py — Configurações padrão do Little Hawk
"""
import os
from typing import Dict, Any

# Configurações de modelo
DEFAULT_MODEL_CONFIG = {
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 2,
    "sink_size": 4,
    "window_size": 28,
    "vocab_size": 512,
    "rope_base": 10000.0,
    "seed": 42
}

# Configurações de inferência
DEFAULT_INFERENCE_CONFIG = {
    "max_tokens": 80,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.92,
    "rep_penalty": 1.15
}

# Configurações de API
DEFAULT_API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "weights_env": "LITTLE_HAWK_WEIGHTS"
}

# Modelos suportados
SUPPORTED_MODELS = {
    "smollm-135m": {
        "id": "HuggingFaceTB/SmolLM-135M",
        "d_model": 576,
        "n_heads": 9,
        "n_layers": 30,
        "vocab_size": 49152,
        "rope_base": 10000.0
    },
    "qwen2.5-0.5b": {
        "id": "Qwen/Qwen2.5-0.5B",
        "d_model": 896,
        "n_heads": 14,
        "n_layers": 24,
        "vocab_size": 151936,
        "rope_base": 1000000.0
    }
}

def load_config_from_env() -> Dict[str, Any]:
    """Carrega configurações do ambiente"""
    config = {
        "weights_path": os.getenv("LITTLE_HAWK_WEIGHTS"),
        "api_host": os.getenv("LITTLE_HAWK_API_HOST", DEFAULT_API_CONFIG["host"]),
        "api_port": int(os.getenv("LITTLE_HAWK_API_PORT", DEFAULT_API_CONFIG["port"]))
    }
    return {k: v for k, v in config.items() if v is not None}