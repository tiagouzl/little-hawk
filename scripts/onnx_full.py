#!/usr/bin/env python3
"""
scripts/onnx_full.py — ONNX completo com KV cache + RoPE (P3)

Exporta o passo autoregressivo completo (30L, cache, RoPE) para ONNX.
A POC anterior (onnx_export.py) só media FFN; esta estende para attention.

Estado v0.5.0: POC FFN já provou teto 6.6× (30L 294→44ms). Full com cache
requer torch.export + past_key_values dinâmicos (onnx 1.18+).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def export_full(weights="little_hawk_weights.npz", bench=False):
    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch não instalado")
        return 1

    from engine.engine import MultiLayerEngine

    meta = json.loads(Path(weights.replace(".npz", "_meta.json")).read_text())
    eng = MultiLayerEngine(
        d_model=meta["d_model"], n_heads=meta["n_heads"], n_layers=meta["n_layers"],
        sink_size=4, window_size=508, vocab_size=meta["vocab_size"],
    )
    eng.load_weights(weights)
    print(f"Pesos: {meta['n_layers']}L d={meta['d_model']} h={meta['n_heads']} d_k={eng.d_k}")
    print("POC full: export com cache dinâmico requer torch.export + onnx 1.18+")
    print("Teto já medido via onnx_export.py: 30L 294→44ms (6.62×) — FFN fusion")
    print("Próximo: torch.export.export() com `past_key_values` como pytree + RoPE")
    print("  - k_cache/v_cache como inputs dinâmicos [1,H,512,d_k]")
    print("  - position freeze via `torch.where(n_ctx <= max_cap, n_ctx-1, max_cap-1)`")
    print("  - validar logits top-5 vs NumPy (tolerância 1e-3)")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="little_hawk_weights.npz")
    p.add_argument("--bench", action="store_true")
    sys.exit(export_full(p.parse_args().weights, p.parse_args().bench))
