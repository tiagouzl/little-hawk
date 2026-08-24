"""
engine/jit_kernels.py — Kernels Numba JIT para aceleração matemática do Little Hawk.

Estado (benchmarks internos): no decode batch-1 os ganhos são NULOS ou negativos
(RMSNorm/SwiGLU são ~2% do passo; o GEMV do lm_head domina com ~75%). Por isso o
JIT é OPT-IN: exige numba instalado ([jit]) E a variável LITTLE_HAWK_JIT=1.
Sem numba: fallback VETORIZADO em NumPy puro (mesma assinatura, mesma semântica).
"""
import math
import os

import numpy as np

HAS_NUMBA = False
if os.getenv("LITTLE_HAWK_JIT") == "1":
    try:
        from numba import njit

        HAS_NUMBA = True
    except ImportError:
        pass

if HAS_NUMBA:

    @njit(fastmath=True, nogil=True)
    def _jit_rms_norm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """RMSNorm otimizado sem alocações intermediárias de arrays."""
        orig_shape = x.shape
        d = orig_shape[-1]
        flat_x = x.reshape(-1, d)
        n_rows = flat_x.shape[0]
        out = np.empty_like(flat_x)

        for i in range(n_rows):
            sum_sq = 0.0
            for j in range(d):
                val = flat_x[i, j]
                sum_sq += val * val
            scale = 1.0 / math.sqrt((sum_sq / d) + eps)
            for j in range(d):
                out[i, j] = flat_x[i, j] * scale * w[j]

        return out.reshape(orig_shape)

    @njit(fastmath=True, nogil=True)
    def _jit_silu_mul(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
        """SwiGLU activation fusion: SiLU(gate) * up."""
        orig_shape = gate.shape
        flat_g = gate.ravel()
        flat_u = up.ravel()
        n = flat_g.size
        out = np.empty(n, dtype=np.float32)

        for i in range(n):
            g = flat_g[i]
            silu = g / (1.0 + math.exp(-g))
            out[i] = silu * flat_u[i]

        return out.reshape(orig_shape)

else:

    def _jit_rms_norm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Fallback NumPy vetorizado (sem numba)."""
        return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * w

    def _jit_silu_mul(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Fallback NumPy vetorizado (sem numba): SiLU(gate) * up."""
        return (gate / (1.0 + np.exp(-gate))) * up


def _rope_numpy(x: np.ndarray, pos: np.ndarray, inv_freq: np.ndarray) -> np.ndarray:
    """
    RoPE rotate_half via NumPy vetorizado (convenção HuggingFace).
    x: shape (B, n_heads, seq_len, d_k)
    """
    ang = np.outer(pos.astype(np.float32), inv_freq)
    emb = np.concatenate([ang, ang], axis=-1)[np.newaxis, np.newaxis]
    s, c = np.sin(emb), np.cos(emb)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * c + np.concatenate([-x2, x1], axis=-1) * s
