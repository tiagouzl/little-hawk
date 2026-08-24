#!/usr/bin/env python3
"""
scripts/onnx_export.py — POC P3 ONNX: exporta Little Hawk para ONNX + bench onnxruntime

Exporta 1 LlamaLayer (ou N layers) via torch.onnx para ONNX Runtime, que usa
kernels otimizados em C++ (MKL, etc) — potencial 2-4× vs NumPy puro para 30L.

Dependências opcionais:
  pip install -e '.[onnx]'  # torch + onnx + onnxruntime
  python scripts/onnx_export.py --weights little_hawk_weights.npz --layers 1 --bench

Sem torch/onnxruntime: script explica o caminho e sai graceful.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def try_export(weights="little_hawk_weights.npz", layers=1, bench=False):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("torch não instalado — POC ONNX requer `pip install torch --index-url https://download.pytorch.org/whl/cpu`")
        print("Caminho: definir `class LlamaLayerTorch(nn.Module)` espelhando `engine/transformer.py:9`,")
        print("carregar pesos de `transplants/qwen.py:98` e exportar via `torch.onnx.export(..., dynamo=True)`.")
        return 1

    try:
        import onnx  # noqa: F401
        import onnxruntime as ort
    except ImportError:
        print("onnx/onnxruntime não instalado — instale com `pip install onnx onnxruntime`")
        print("Sem eles, o export ainda pode ser feito via `torch.onnx.export` mas sem bench.")
        has_ort = False
    else:
        has_ort = True

    # Carrega pesos NumPy
    if not Path(weights).exists():
        print(f"Pesos não encontrados: {weights}")
        return 1
    meta = json.loads(Path(weights.replace(".npz", "_meta.json")).read_text(encoding="utf-8"))
    import numpy as np

    from engine.engine import MultiLayerEngine

    eng = MultiLayerEngine(
        d_model=meta["d_model"], n_heads=meta["n_heads"], n_layers=meta["n_layers"],
        sink_size=4, window_size=508, vocab_size=meta["vocab_size"],
    )
    eng.load_weights(weights)
    print(f"Pesos carregados: {meta['n_layers']}L, d={meta['d_model']}, h={meta['n_heads']}")

    # Define camada Torch espelhada (RMSNorm + RoPE simplificado sem cache para POC)
    class LlamaLayerTorch(nn.Module):
        def __init__(self, layer, d_model, n_heads, d_k):
            super().__init__()
            self.n_heads, self.d_k = n_heads, d_k
            self.W_q = nn.Parameter(torch.from_numpy(layer.W_q.T.copy()))
            self.W_k = nn.Parameter(torch.from_numpy(layer.W_k.T.copy()))
            self.W_v = nn.Parameter(torch.from_numpy(layer.W_v.T.copy()))
            self.W_o = nn.Parameter(torch.from_numpy(layer.W_o.T.copy()))
            self.rms_w = nn.Parameter(torch.from_numpy(layer.rms_attn.copy()))
            self.gate = nn.Parameter(torch.from_numpy(layer.gate.T.copy()))
            self.up = nn.Parameter(torch.from_numpy(layer.up.T.copy()))
            self.down = nn.Parameter(torch.from_numpy(layer.down.T.copy()))

        def forward(self, x):
            # x: [1, d]
            # RMSNorm
            var = x.pow(2).mean(-1, keepdim=True)
            x_n = x * torch.rsqrt(var + 1e-6) * self.rms_w
            q = x_n @ self.W_q.T
            k = x_n @ self.W_k.T
            v = x_n @ self.W_v.T
            # attention simplificada (sem RoPE/cache) — só para bench GEMV
            # FFN
            x_n2 = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.rms_w
            g = x_n2 @ self.gate.T
            u = x_n2 @ self.up.T
            h = (g / (1 + torch.exp(-g))) * u
            out = h @ self.down.T
            return out

    # Exporta N camadas (stack) como POC
    n_export = min(layers, eng.n_layers)
    layers_torch = [LlamaLayerTorch(eng.layers[i], eng.d_model, eng.n_heads, eng.d_k) for i in range(n_export)]

    class StackTorch(nn.Module):
        def __init__(self, lst):
            super().__init__()
            self.layers = nn.ModuleList(lst)
        def forward(self, x):
            for l in self.layers:
                x = x + l(x)  # simplificado: attn+ffn residual
            return x

    model = StackTorch(layers_torch)
    model.eval()
    dummy = torch.randn(1, eng.d_model)
    layer0 = eng.layers[0]

    onnx_path = f"/tmp/little_hawk_L{layers}.onnx"
    try:
        torch.onnx.export(model, dummy, onnx_path, input_names=["x"], output_names=["out"], dynamo=False)
        print(f"ONNX exportado: {onnx_path} ({Path(onnx_path).stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        print(f"Export falhou: {e}")
        return 1

    if bench and has_ort:
        import os

        os.environ["OMP_NUM_THREADS"] = "1"
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        for _ in range(5):
            _ = sess.run(None, {"x": dummy.numpy()})
        # NumPy bench para N camadas (stack simplificado)
        x_np = dummy.numpy()
        def numpy_stack(x):
            for i in range(n_export):
                l = eng.layers[i]
                # RMSNorm simplificado + ffn gate
                var = (x * x).mean(axis=-1, keepdims=True)
                x_n = x / np.sqrt(var + 1e-6) * l.rms_attn
                g = x_n @ l.gate
                u = x_n @ l.up
                h = (g / (1 + np.exp(-g))) * u @ l.down
                x = x + h
            return x
        t0 = time.perf_counter()
        for _ in range(30):
            _ = numpy_stack(x_np)
        ms_np = (time.perf_counter() - t0) / 30 * 1000
        t0 = time.perf_counter()
        for _ in range(30):
            _ = sess.run(None, {"x": dummy.numpy()})
        ms_ort = (time.perf_counter() - t0) / 30 * 1000
        print(f"NumPy {n_export}L stack: {ms_np:.2f} ms")
        print(f"ONNX Runtime {n_export}L: {ms_ort:.2f} ms  (speedup {ms_np/ms_ort:.2f}x)")
    elif bench:
        print("Bench pulado — onnxruntime não instalado.")

    print("\nPróximo passo ONNX: estender para 30L com cache (KV como inputs/outputs),")
    print("usar `torch.onnx.export` com `past_key_values` dinâmicos e validar logits vs NumPy.")
    return 0


def main():
    p = argparse.ArgumentParser(description="POC ONNX export")
    p.add_argument("--weights", default="little_hawk_weights.npz")
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--bench", action="store_true")
    args = p.parse_args()
    sys.exit(try_export(args.weights, args.layers, args.bench))


if __name__ == "__main__":
    main()
