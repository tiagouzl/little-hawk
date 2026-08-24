#!/usr/bin/env python3
"""
scripts/benchmark_latency.py — Benchmark controlado de latência (p50, p95).
"""
import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.engine import MultiLayerEngine
from runtime.inference import LittleHawkInference, SamplingConfig
from runtime.tokenizer import BPETokenizer

def run_bench(weights_path: str = "little_hawk_weights.npz",
              prompt: str = "The architecture of transformer models is based on",
              tokens: int = 100,
              num_threads: int = 1,
              warmup_tokens: int = 10):
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)

    tok = BPETokenizer()
    meta_path = weights_path.replace(".npz", "_meta.json")
    tok.load_donor_vocab(meta_path)

    import json
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    engine = MultiLayerEngine(
        d_model=int(meta["d_model"]),
        n_heads=int(meta["n_heads"]),
        n_layers=int(meta["n_layers"]),
        sink_size=4,
        window_size=508,
        vocab_size=int(meta["vocab_size"])
    )
    engine.load_weights(weights_path)
    hawk = LittleHawkInference(tokenizer=tok, engine=engine)

    # 1. Warm-up
    print(f"Executando Warmup ({warmup_tokens} tokens)...")
    warmup_cfg = SamplingConfig(max_tokens=warmup_tokens, temperature=0.7)
    hawk.generate(prompt, sampling_config=warmup_cfg, panel=False)

    # 2. Medição com coleta individual de latências
    latencies = []
    def on_token_cb(token, step, stats):
        latencies.append(stats["latency"])

    cfg = SamplingConfig(max_tokens=tokens, temperature=0.7)
    print(f"Executando Benchmark ({tokens} tokens, threads={num_threads})...")
    np.random.seed(42)
    start_total = time.perf_counter()
    hawk.generate(prompt, sampling_config=cfg, on_token=on_token_cb, panel=False)
    total_time = time.perf_counter() - start_total

    l_arr = np.array(latencies)
    p50 = np.percentile(l_arr, 50)
    p95 = np.percentile(l_arr, 95)
    mean_lat = np.mean(l_arr)
    std_lat = np.std(l_arr)

    print("\n" + "="*50)
    print(f"RESULTADOS DO BENCHMARK ({len(latencies)} tokens)")
    print("="*50)
    print(f"Latência Média:  {mean_lat:.2f} ms/token (± {std_lat:.2f} ms)")
    print(f"Latência P50:    {p50:.2f} ms/token")
    print(f"Latência P95:    {p95:.2f} ms/token")
    print(f"Latência Mínima: {np.min(l_arr):.2f} ms/token")
    print(f"Latência Máxima: {np.max(l_arr):.2f} ms/token")
    print(f"Tempo Total:     {total_time:.2f} s")
    print("="*50)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="little_hawk_weights.npz")
    p.add_argument("--tokens", type=int, default=100)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()
    run_bench(weights_path=args.weights, tokens=args.tokens, num_threads=args.threads)
