"""
engine/onnx_engine.py — Backend ONNX Runtime para Little Hawk (opt-in)

Implementa mesma interface que MultiLayerEngine.step mas via ONNX Runtime
com cache linear dinâmico (S cresce). Ativado via LITTLE_HAWK_ONNX=1.

Export: `python scripts/onnx_export.py --weights little_hawk_weights.npz --layers 30 --bench`
gera /tmp/test_30L_stack.onnx (478 MB). Este módulo carrega esse ONNX
ou exporta sob demanda se não existir.

Uso:
  LITTLE_HAWK_ONNX=1 python little_hawk_cli.py infer --weights little_hawk_weights.npz --prompt "..."
  # fallback para NumPy se ONNX não disponível

Benchmark 30L 1 thread: NumPy 294 ms → ONNX 31.5 ms (9×) / Torch 70 ms (4×)
"""
import os
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class OnnxEngine:
    """Wrapper com mesma API que MultiLayerEngine para uso em runtime/inference.py"""

    def __init__(self, npz_path="little_hawk_weights.npz", onnx_path="/tmp/little_hawk_30L_stack.onnx"):
        if not HAS_ORT or not HAS_TORCH:
            raise RuntimeError("onnxruntime/torch não instalados — `pip install -e '.[onnx]'`")
        from engine.torch_model import LittleHawkTorch

        # Carrega para obter meta, mas ONNX já tem pesos embutidos
        self.npz_path = npz_path
        self.onnx_path = onnx_path
        # Exporta se não existir
        if not Path(onnx_path).exists():
            self._export_onnx(npz_path, onnx_path)
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        # Meta para compatibilidade com MultiLayerEngine
        import json

        meta = json.loads(Path(str(npz_path).replace(".npz", "_meta.json")).read_text())
        self.d_model = int(meta["d_model"])
        self.n_heads = int(meta["n_heads"])
        self.n_layers = int(meta["n_layers"])
        self.d_k = self.d_model // self.n_heads
        self.V = int(meta["vocab_size"])
        self.bos_id = int(meta.get("bos_id", 1))
        self.eos_id = int(meta.get("eos_id", 2))
        self.S, self.W, self.max_cap = 4, 508, 512
        # Para compatibilidade, mantém inv_freq etc
        self.inv_freq = 1.0 / (1000000.0 ** (np.arange(0, self.d_k, 2, dtype=np.float32) / self.d_k)) if self.d_k else np.array([])
        self.wbi = np.arange(self.W, dtype=np.int64)
        self.si = np.arange(self.S, dtype=np.int64)
        # ONNX usa cache linear, não circular — win_ptr é dummy
        self._init_onnx_caches()

    def _export_onnx(self, npz_path, onnx_path):
        """Exporta 30L stack via torch.onnx (dynamo=False)"""
        from engine.torch_model import LlamaLayerTorch
        import json
        import torch
        import torch.nn as nn

        data = np.load(npz_path, allow_pickle=False)
        meta = json.loads(Path(str(npz_path).replace(".npz", "_meta.json")).read_text())
        d_model = int(data["_meta_d_model"]); n_heads = int(data["_meta_n_heads"]); d_k = d_model // n_heads
        rope_base = float(data["_meta_rope_base"])
        inv_freq = 1.0 / (rope_base ** (np.arange(0, d_k, 2, dtype=np.float32) / d_k))
        n_layers = int(data["_meta_n_layers"])
        layers = [
            LlamaLayerTorch(
                data[f"L{i}_W_q"], data[f"L{i}_W_k"], data[f"L{i}_W_v"], data[f"L{i}_W_o"],
                data[f"L{i}_rms_attn"], data[f"L{i}_gate"], data[f"L{i}_up"], data[f"L{i}_down"], data[f"L{i}_rms_ffn"],
                n_heads, d_k, inv_freq=inv_freq,
            )
            for i in range(n_layers)
        ]

        class Stack(nn.Module):
            def __init__(self, lst):
                super().__init__()
                self.layers = nn.ModuleList(lst)
            def forward(self, x, k_stack, v_stack):
                new_k, new_v = [], []
                for i, l in enumerate(self.layers):
                    x, k_out, v_out = l(x, k_stack[i], v_stack[i])
                    new_k.append(k_out)
                    new_v.append(v_out)
                return x, torch.stack(new_k), torch.stack(new_v)

        m = Stack(layers)
        m.eval()
        x = torch.randn(1, 1, d_model)
        k_stack = torch.zeros(n_layers, 1, n_heads, 4, d_k)
        v_stack = torch.zeros(n_layers, 1, n_heads, 4, d_k)
        torch.onnx.export(
            m, (x, k_stack, v_stack), onnx_path,
            input_names=["x", "k_stack", "v_stack"],
            output_names=["out", "k_out", "v_out"],
            dynamic_axes={"x": {1: "seq"}, "k_stack": {3: "S"}, "v_stack": {3: "S"}},
            opset_version=17, dynamo=False,
        )
        print(f"ONNX exportado: {onnx_path} ({Path(onnx_path).stat().st_size/1e6:.1f} MB)")

    def _init_onnx_caches(self):
        # cache linear inicial com S=4 (sink)
        self.k_stack = np.zeros((self.n_layers, 1, self.n_heads, 4, self.d_k), dtype=np.float32)
        self.v_stack = np.zeros((self.n_layers, 1, self.n_heads, 4, self.d_k), dtype=np.float32)

    def init_cache(self):
        # Para compatibilidade com API NumPy, retorna lista de tuples mas ONNX usa stack
        # Retorna stack como cache para uso interno; win_ptr/n_ctx são gerenciados externamente
        return [(np.zeros((1, self.n_heads, self.max_cap, self.d_k), np.float32),
                 np.zeros((1, self.n_heads, self.max_cap, self.d_k), np.float32)) for _ in range(self.n_layers)]

    def load_weights(self, path):
        # Pesos já embutidos no ONNX — apenas valida
        import json

        meta_path = str(path).replace(".npz", "_meta.json")
        meta = json.loads(Path(meta_path).read_text())
        assert int(meta["n_layers"]) == self.n_layers

    def step(self, token_id, caches, win_ptr, n_ctx):
        # Usa cache linear interno (ignora caches/win_ptr externos para POC)
        # x from embed
        import json

        # Carrega embed e norm/lm_head para final (NumPy, como fallback)
        # Para v0.6 POC, embed ainda NumPy, layers via ONNX
        data = np.load(self.npz_path, allow_pickle=False)
        embed = data["embed"]
        x_np = embed[token_id][None, None, :]  # [1,1,576] float32
        x_t = x_np.astype(np.float32)

        # ONNX run: x [1,1,576], k/v_stack [30,1,9,S,64] S cresce
        out, k_out, v_out = self.sess.run(
            None,
            {
                "x": x_t,
                "k_stack": self.k_stack,
                "v_stack": self.v_stack,
            },
        )
        self.k_stack, self.v_stack = k_out, v_out
        # final norm + lm_head (NumPy, como no engine original)
        # out is [1,1,576] from last layer
        W_lm_t = data["lm_head"]  # [V,d]
        norm_w = data["norm_w"]
        # RMSNorm
        xn = out[:, 0, :]  # [1,576]
        xn = xn / np.sqrt(np.mean(xn * xn, axis=-1, keepdims=True) + 1e-6) * norm_w
        logits = (W_lm_t @ xn[0].reshape(-1, 1)).T  # [1,V]
        new_win_ptr = (win_ptr + 1) % self.W if n_ctx > self.S else win_ptr
        # Retorna caches dummy para compatibilidade (lista de 512)
        # Para manter API, retorna caches originais com k/v atualizados no slot 0
        new_caches = caches  # mantém compat, mas ONNX cache é linear
        return logits, new_caches, new_win_ptr, 0.0


def is_onnx_enabled():
    return os.getenv("LITTLE_HAWK_ONNX") == "1" and HAS_ORT and HAS_TORCH
