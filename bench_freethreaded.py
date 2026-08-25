#!/usr/bin/env python3
"""
bench_freethreaded.py — Little Hawk

Isola o efeito do GIL na concorrência do engine, sem depender de pesos
carregados: simula a carga de trabalho dominante do decode (GEMV/GEMM
em NumPy) sob N threads concorrentes e mede throughput + latência p50/p99.

Uso:
    # Baseline (CPython normal, GIL ligado)
    python3.11 bench_freethreaded.py --threads 1 2 4 --iters 200

    # Free-threaded (requer build 3.13t)
    python3.13t bench_freethreaded.py --threads 1 2 4 --iters 200

    # Forçar GIL ligado mesmo em 3.13t (para comparação isolada)
    PYTHON_GIL=1 python3.13t bench_freethreaded.py --threads 1 2 4 --iters 200

Interpretação:
    - Se throughput escalar quase linear com threads em 3.13t mas ficar
      achatado em 3.11/3.13-GIL, o free-threading está realmente
      ajudando na parte NumPy do seu forward pass.
    - Se NÃO escalar nem em 3.13t, o gargalo não é o GIL (pode ser
      BLAS já usando todos os cores, ou memory bandwidth) — nesse
      caso o Semaphore(2) atual provavelmente já está perto do ótimo
      e migrar pra 3.13t não compensaria o esforço.

Requer: numpy, psutil
"""

import argparse
import json
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import psutil


def gil_status() -> str:
    """Reporta se o GIL está ativo (Python 3.13+ free-threaded builds)."""
    if hasattr(sys, "_is_gil_enabled"):
        return "enabled" if sys._is_gil_enabled() else "disabled"
    return "n/a (build normal, GIL sempre ligado)"


def make_workload(d_model: int, intermediate: int, n_layers_sim: int):
    """
    Simula o custo de um passo de decode (GEMV batch-1) equivalente ao
    SmolLM-135M: d=576, intermediate=1536, ~30 camadas, projeções
    Q/K/V/O + gate/up/down por camada.
    """
    # Escala pequena + normalização por camada evita explosão numérica
    # em 30 camadas encadeadas (o objetivo aqui é custo de FLOPs, não
    # fidelidade numérica).
    scale = 1.0 / np.sqrt(d_model)
    rng = np.random.default_rng(42)
    w_qkvo = [(rng.standard_normal((d_model, d_model)) * scale).astype(np.float32) for _ in range(4)]
    w_gate = (rng.standard_normal((d_model, intermediate)) * scale).astype(np.float32)
    w_up = (rng.standard_normal((d_model, intermediate)) * scale).astype(np.float32)
    w_down = (rng.standard_normal((intermediate, d_model)) * scale).astype(np.float32)

    def rms_norm(x: np.ndarray) -> np.ndarray:
        return x / (np.sqrt(np.mean(x * x, axis=-1, keepdims=True)) + 1e-6)

    def step(x: np.ndarray) -> np.ndarray:
        for _ in range(n_layers_sim):
            x = rms_norm(x)
            for w in w_qkvo:
                x = x @ w
            x = rms_norm(x)
            gate = np.clip(x @ w_gate, -30, 30)
            up = x @ w_up
            silu = gate / (1.0 + np.exp(-gate))
            x = (silu * up) @ w_down
        return x

    return step


@dataclass
class ThreadResult:
    latencies_ms: list = field(default_factory=list)


def worker(step_fn, d_model: int, iters: int, result: ThreadResult, barrier: threading.Barrier):
    x = np.random.default_rng().standard_normal((1, d_model)).astype(np.float32)
    barrier.wait()  # alinha o start de todas as threads
    for _ in range(iters):
        t0 = time.perf_counter()
        x = step_fn(x)
        result.latencies_ms.append((time.perf_counter() - t0) * 1000)


def run_trial(n_threads: int, iters: int, d_model: int, intermediate: int, n_layers_sim: int):
    step_fn = make_workload(d_model, intermediate, n_layers_sim)
    results = [ThreadResult() for _ in range(n_threads)]
    barrier = threading.Barrier(n_threads)
    threads = [
        threading.Thread(target=worker, args=(step_fn, d_model, iters, results[i], barrier)) for i in range(n_threads)
    ]

    proc = psutil.Process()
    rss_samples = []
    stop_flag = threading.Event()

    def sample_rss():
        while not stop_flag.is_set():
            rss_samples.append(proc.memory_info().rss / (1024 * 1024))
            time.sleep(0.05)

    sampler = threading.Thread(target=sample_rss, daemon=True)

    t_start = time.perf_counter()
    sampler.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop_flag.set()
    sampler.join(timeout=1)
    wall_s = time.perf_counter() - t_start

    all_lat = [lat for r in results for lat in r.latencies_ms]
    total_steps = n_threads * iters
    throughput = total_steps / wall_s

    return {
        "n_threads": n_threads,
        "wall_s": round(wall_s, 3),
        "total_steps": total_steps,
        "throughput_steps_per_s": round(throughput, 2),
        "latency_p50_ms": round(statistics.median(all_lat), 2),
        "latency_p99_ms": round(statistics.quantiles(all_lat, n=100)[98] if len(all_lat) >= 100 else max(all_lat), 2),
        "peak_rss_mb": round(max(rss_samples), 1) if rss_samples else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4], help="Contagens de threads a testar")
    ap.add_argument("--iters", type=int, default=200, help="Iterações (steps) por thread")
    ap.add_argument("--d-model", type=int, default=576, help="Dimensão do modelo (default: SmolLM-135M)")
    ap.add_argument("--intermediate", type=int, default=1536, help="Dimensão intermediária SwiGLU")
    ap.add_argument("--n-layers-sim", type=int, default=30, help="Camadas simuladas por step")
    ap.add_argument("--json", type=str, default=None, help="Caminho para salvar resultados em JSON")
    args = ap.parse_args()

    print(f"Python: {sys.version.split()[0]}  |  GIL: {gil_status()}")
    print(f"BLAS threads padrão do NumPy: verifique OMP_NUM_THREADS/OPENBLAS_NUM_THREADS no ambiente")
    print(f"Workload: d_model={args.d_model} intermediate={args.intermediate} layers_sim={args.n_layers_sim}")
    print("-" * 72)

    all_results = []
    baseline_throughput = None
    for n in args.threads:
        r = run_trial(n, args.iters, args.d_model, args.intermediate, args.n_layers_sim)
        if baseline_throughput is None:
            baseline_throughput = r["throughput_steps_per_s"]
        scaling = r["throughput_steps_per_s"] / baseline_throughput
        r["scaling_vs_1thread"] = round(scaling, 2)
        all_results.append(r)
        print(
            f"threads={n:2d}  throughput={r['throughput_steps_per_s']:8.2f} steps/s  "
            f"(escala {scaling:.2f}x)  p50={r['latency_p50_ms']:6.2f}ms  "
            f"p99={r['latency_p99_ms']:6.2f}ms  RSS={r['peak_rss_mb']}MB"
        )

    print("-" * 72)
    ideal = args.threads[-1] / args.threads[0] if len(args.threads) > 1 else 1
    actual = all_results[-1]["scaling_vs_1thread"]
    print(f"Escala ideal ({args.threads[0]}->{args.threads[-1]} threads): {ideal:.2f}x  |  Observada: {actual:.2f}x")
    if actual < ideal * 0.6:
        print("=> Pouca escala: gargalo provavelmente NÃO é o GIL (BLAS já paraleliza, ou memory-bound).")
    else:
        print("=> Boa escala: concorrência está ajudando — free-threading pode valer a pena aqui.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {"python_version": sys.version, "gil_status": gil_status(), "results": all_results},
                f,
                indent=2,
            )
        print(f"\nResultados salvos em {args.json}")


if __name__ == "__main__":
    main()
