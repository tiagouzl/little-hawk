#!/usr/bin/env python3
"""
scripts/bench_prefill_ttft.py — TTFT pareado do prefill: NumPy vs ONNX vs Mojo.

Critério de aprovação já definido para o port do kernel de GEMM do prefill
para Mojo: TTFT precisa ficar >= 1.5x mais rápido que o baseline NumPy
batelado (MultiLayerEngine.prefill), que é o caminho rápido real. O ONNX
é medido só como referência informativa: OnnxEngine.prefill() hoje é
sequencial (loop de step() via grafo single-token) e já é ~7x mais lento
que o NumPy batelado no prefill — usá-lo como baseline tornaria o
critério vácuo (qualquer kernel batelado passaria sem fazer nada).
Se não bater 1.5x vs NumPy, arquivar como testado e rejeitado
(ver engine/mojo_kernels.py).

Atenção: OnnxEngine.prefill() hoje é sequencial (loop de step() via grafo
single-token), enquanto MultiLayerEngine.prefill() é batelado (T<=max_cap
em um único forward GEMM causal). O "speedup" onnx/numpy que sai daqui
é esperado < 1 no prefill (~0.15 medido); o 1.45x citado no docstring de
onnx_engine.py é por step, não por prefill. O que importa para o veredito
é onde o Mojo fica em relação ao NumPy batelado.

Uso:
  python scripts/bench_prefill_ttft.py --weights little_hawk_weights.npz --reps 10
  LITTLE_HAWK_MOJO_PREFILL=1 python scripts/bench_prefill_ttft.py --reps 10  # quando o kernel existir
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.engine import MultiLayerEngine
from runtime.tokenizer import BPETokenizer

# Prompt de referência (dá ~64 tokens no vocab real; o nome antigo PROMPT_129
# valia para tokens sintéticos de tamanho fixo — contagem real sai no print).
PROMPT_REF = (
    "The development of modern science began when thinkers decided to test ideas "
    "against observation rather than authority, a shift that took centuries to "
    "complete across different fields of inquiry and required new institutions, "
    "new instruments, and new standards of evidence that could be shared and "
    "checked by independent observers working in different countries and languages "
    "who rarely agreed on anything else"
)


def load_engine_and_tokens(weights_path: str, prompt: str):
    meta_path = weights_path.replace(".npz", "_meta.json")
    m = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    tok = BPETokenizer()
    tok.load_donor_vocab(meta_path)
    eng = MultiLayerEngine(
        d_model=m["d_model"], n_heads=m["n_heads"], n_layers=m["n_layers"],
        sink_size=4, window_size=508, vocab_size=m["vocab_size"],
    )
    eng.load_weights(weights_path)
    ids = tok.encode(prompt, add_bos=True)
    return eng, ids


def load_onnx_engine(weights_path: str):
    from engine.onnx_engine import OnnxEngine, HAS_ORT, HAS_TORCH
    if not (HAS_ORT and HAS_TORCH):
        return None
    return OnnxEngine(npz_path=weights_path)


def time_prefill(engine, ids: list[int]) -> float:
    """TTFT isolado: só o prefill, cache fresco a cada rep."""
    t0 = time.perf_counter()
    engine.prefill(ids, caches=None)
    return time.perf_counter() - t0


def run(weights_path: str, reps: int, prompt: str, mojo_enabled: bool) -> dict:
    eng, ids = load_engine_and_tokens(weights_path, prompt)
    T = len(ids)
    print(f"Prompt: {T} tokens (T <= max_cap={eng.max_cap} -> caminho batelado)" if T <= eng.max_cap
          else f"Prompt: {T} tokens (T > max_cap={eng.max_cap} -> chunked, primeiro chunk batelado + loop)")

    onnx_eng = load_onnx_engine(weights_path)
    if onnx_eng is None:
        print("AVISO: onnxruntime/torch indisponíveis -- pulando baseline ONNX "
              "(instale com `pip install -e '.[onnx]'`)", file=sys.stderr)
    else:
        assert eng.max_cap == onnx_eng.max_cap, \
            f"max_cap divergente: numpy={eng.max_cap} onnx={onnx_eng.max_cap}"

    # warm-up: primeira chamada paga custo de import/JIT/alocação de buffers,
    # não deve entrar na medição
    _ = time_prefill(eng, ids)
    if onnx_eng is not None:
        _ = time_prefill(onnx_eng, ids)

    numpy_times, onnx_times = [], []
    for i in range(reps):
        # alterna ordem pra não deixar efeito térmico/cache sistematicamente
        # a favor de um backend
        if i % 2 == 0:
            nt = time_prefill(eng, ids)
            ot = time_prefill(onnx_eng, ids) if onnx_eng is not None else float("nan")
        else:
            ot = time_prefill(onnx_eng, ids) if onnx_eng is not None else float("nan")
            nt = time_prefill(eng, ids)
        numpy_times.append(nt)
        onnx_times.append(ot)
        print(f"[{i+1}/{reps}] numpy={nt*1000:.2f}ms onnx={ot*1000:.2f}ms")

    result = {
        "prompt_tokens": T,
        "reps": reps,
        "numpy_ms": [round(t * 1000, 3) for t in numpy_times],
        "onnx_ms": [round(t * 1000, 3) for t in onnx_times] if onnx_eng is not None else None,
    }

    if onnx_eng is not None:
        speedup_onnx_vs_numpy = [n / o for n, o in zip(numpy_times, onnx_times)]
        result["summary_onnx_vs_numpy"] = {
            "median_numpy_ms": statistics.median(result["numpy_ms"]),
            "median_onnx_ms": statistics.median(result["onnx_ms"]),
            "median_speedup": statistics.median(speedup_onnx_vs_numpy),
            "note": "só referência informativa; ONNX sequencial é esperado "
                    "mais lento que NumPy batelado no prefill (o 1.45x de "
                    "onnx_engine.py é por step, não por prefill)",
        }

    if mojo_enabled:
        from engine.mojo_kernels import HAS_MOJO
        if not HAS_MOJO:
            print("LITTLE_HAWK_MOJO_PREFILL=1 setado mas o binding little_hawk_mojo "
                  "não foi encontrado -- implemente engine/mojo_kernels.py primeiro.",
                  file=sys.stderr)
        else:
            # Mesmo engine NumPy, mas prefill() já despacha para o kernel Mojo
            # via engine/mojo_kernels.py::prefill_layer_forward quando HAS_MOJO=True
            mojo_times = []
            _ = time_prefill(eng, ids)  # warm-up do kernel mojo
            for i in range(reps):
                mojo_times.append(time_prefill(eng, ids))
            result["mojo_ms"] = [round(t * 1000, 3) for t in mojo_times]
            speedup_mojo_vs_numpy = [n / m for n, m in zip(numpy_times, mojo_times)]
            median = statistics.median(speedup_mojo_vs_numpy)
            result["summary_mojo_vs_numpy"] = {
                "median_numpy_ms": statistics.median(result["numpy_ms"]),
                "median_mojo_ms": statistics.median(result["mojo_ms"]),
                "median_speedup": median,
                "criterio": "mediana >= 1.5x vs NumPy batelado",
                "veredito": "APROVADO" if median >= 1.5 else "REJEITADO",
            }

    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="little_hawk_weights.npz")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--prompt", default=PROMPT_REF)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    mojo_enabled = os.getenv("LITTLE_HAWK_MOJO_PREFILL") == "1"
    result = run(args.weights, args.reps, args.prompt, mojo_enabled)

    print("\n--- Resumo ---")
    print(json.dumps(
        {k: v for k, v in result.items() if k.startswith("summary")},
        indent=2, ensure_ascii=False,
    ))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[salvo em {args.json_out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
