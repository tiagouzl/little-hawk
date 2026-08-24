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

    def __init__(self, npz_path="little_hawk_weights.npz", onnx_path="/tmp/test_30L_stack.onnx"):
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
        """Exporta LittleHawkTorch 30L completo com position freeze via torch.onnx (dynamo=False)"""
        from engine.torch_model import LittleHawkTorch
        import torch

        m = LittleHawkTorch(npz_path)
        m.eval()
        input_ids = torch.tensor([[1]], dtype=torch.long)
        k_stack = torch.zeros(m.n_layers, 1, m.n_heads, 512, 64)
        v_stack = torch.zeros(m.n_layers, 1, m.n_heads, 512, 64)
        win_ptr = torch.tensor(0, dtype=torch.int64)
        n_ctx = torch.tensor(1, dtype=torch.int64)
        torch.onnx.export(
            m, (input_ids, k_stack, v_stack, win_ptr, n_ctx), onnx_path,
            input_names=["input_ids", "k_stack", "v_stack", "win_ptr", "n_ctx"],
            output_names=["logits", "k_out", "v_out", "new_win_ptr"],
            dynamic_axes={"input_ids": {1: "seq"}},
            opset_version=17, dynamo=False,
        )
        print(f"ONNX exportado: {onnx_path} ({Path(onnx_path).stat().st_size/1e6:.1f} MB)")

    def _init_onnx_caches(self):
        # cache circular 512 para ONNX (position freeze)
        self.k_stack = np.zeros((self.n_layers, 1, self.n_heads, 512, self.d_k), dtype=np.float32)
        self.v_stack = np.zeros((self.n_layers, 1, self.n_heads, 512, self.d_k), dtype=np.float32)
        self.win_ptr_onnx = 0
        self.n_ctx_onnx = 0

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
        # ONNX full com position freeze — usa cache interno empilhado
        # token_id -> input_ids [[token_id]], k/v_stack [30,1,9,512,64], win_ptr/n_ctx escalares
        # Para manter API compatível, caches externos são ignorados (ONNX tem cache interno)
        # n_ctx é incrementado fora (como em runtime/inference.py), aqui apenas repassa para ONNX
        input_ids = np.array([[token_id]], dtype=np.int64)
        # ONNX run
        logits, k_out, v_out, new_win_ptr = self.sess.run(
            None,
            {
                "input_ids": input_ids,
                "k_stack": self.k_stack,
                "v_stack": self.v_stack,
                "win_ptr": np.array(win_ptr, dtype=np.int64),
                "n_ctx": np.array(n_ctx, dtype=np.int64),
            },
        )
        self.k_stack, self.v_stack = k_out, v_out
        # logits já vem do ONNX [1,1,V] -> [1,V]
        logits = logits[:, 0, :] if logits.ndim == 3 else logits
        # new_win_ptr é array 0-d
        new_win_ptr = int(new_win_ptr) if isinstance(new_win_ptr, np.ndarray) else int(new_win_ptr)
        # Retorna caches dummy para compatibilidade com runtime/inference.py (que espera lista)
        # O cache real está em self.k_stack/v_stack
        return logits, caches, new_win_ptr, 0.0


def is_onnx_enabled():
    return os.getenv("LITTLE_HAWK_ONNX") == "1" and HAS_ORT and HAS_TORCH
