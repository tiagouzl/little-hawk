#!/usr/bin/env python3
"""
scripts/cython_poc.py — POC P3 Cython: Python puro vs fast_step (inlined) vs Numba

Compara o overhead de dispatch das 30 camadas. O ganho de Cython viria
de eliminar 450 calls/token e manter BLAS para GEMVs (não jittar GEMVs,
que são 6× mais lentos em loop manual — ver ANALISE.md:13).

Uso:
  python scripts/cython_poc.py --weights little_hawk_weights.npz --tokens 20
  LITTLE_HAWK_JIT=1 python scripts/cython_poc.py  # com numba
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from engine.engine import MultiLayerEngine
from engine.fast_step import fast_step


def bench(engine, fn, tokens=20):
    caches = engine.init_cache()
    win_ptr, n_ctx = 0, 0
    # warmup + correctness check
    for _ in range(3):
        n_ctx += 1
        logits, caches, win_ptr, _ = fn(engine, 1, caches, win_ptr, n_ctx)
    # timing
    caches = engine.init_cache()
    win_ptr, n_ctx = 0, 0
    for tid in [1, 2, 3]:
        n_ctx += 1
        _, caches, win_ptr, _ = fn(engine, tid, caches, win_ptr, n_ctx)
    t0 = time.perf_counter()
    for i in range(tokens):
        n_ctx += 1
        logits, caches, win_ptr, _ = fn(engine, (i % 100) + 1, caches, win_ptr, n_ctx)
    return (time.perf_counter() - t0) / tokens * 1000, logits


def main():
    p = argparse.ArgumentParser(description="POC Cython — dispatch 30 camadas")
    p.add_argument("--weights", default="little_hawk_weights.npz")
    p.add_argument("--tokens", type=int, default=20)
    args = p.parse_args()

    if not Path(args.weights).exists():
        print(f"Pesos não encontrados: {args.weights} — usando modo demo (2 camadas)")
        from runtime.tokenizer import BPETokenizer, CORPUS

        tok = BPETokenizer()
        tok.train(CORPUS, vocab_size=512, verbose=False)
        eng = MultiLayerEngine(d_model=128, n_heads=4, n_layers=2, sink_size=4, window_size=28, vocab_size=len(tok.vocab))
    else:
        import json

        meta = json.loads(Path(args.weights.replace(".npz", "_meta.json")).read_text(encoding="utf-8"))
        eng = MultiLayerEngine(
            d_model=meta["d_model"],
            n_heads=meta["n_heads"],
            n_layers=meta["n_layers"],
            sink_size=4,
            window_size=508,
            vocab_size=meta["vocab_size"],
        )
        eng.load_weights(args.weights)

    ms_orig, logits_orig = bench(eng, lambda e, t, c, w, n: e.step(t, c, w, n), tokens=args.tokens)
    ms_fast, logits_fast = bench(eng, fast_step, tokens=args.tokens)

    # correctness: logits devem ser idênticos (mesma matemática, só inlining)
    max_diff = float(np.abs(logits_orig - logits_fast).max())
    print(f"Original step: {ms_orig:.2f} ms/token")
    print(f"Fast inlined : {ms_fast:.2f} ms/token  (speedup {ms_orig/ms_fast:.2f}x)")
    print(f"max |logits diff|: {max_diff:.2e} {'✅' if max_diff < 1e-5 else '❌'}")

    print("\nNota: fast_step ainda é Python puro com BLAS. Cython verdadeiro (cdef + nogil)")
    print("para RoPE/softmax + BLAS via C-API daria +5-15% adicional, não 6×. O gargalo")
    print("GEMV (401µs vs 2448µs njit) confirma que jittar GEMVs piora — ver ANALISE.md:13.")
    print("Próximo passo Cython: compilar fast_step com `cython -a` e `cimport numpy`, mantendo")
    print("`x @ W` como `scipy.linalg.blas.sgemv` via C-API.")


if __name__ == "__main__":
    main()
