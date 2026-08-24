# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
engine/cython_fast.pyx — Cython extension para P3

Compila fast_step com cdef + nogil para partes não-GEMV (RoPE/softmax loops),
mantendo GEMVs em BLAS via typed memoryviews + `cimport numpy`.

Build:
  pip install -e '.[cython]'
  python setup.py build_ext --inplace  # ou `pip install -e .` já compila via pyproject
Uso:
  from engine.cython_fast import cython_fast_step  # se compilado
  fallback para engine.fast_step.fast_step se não compilado
"""
import cython
import numpy as np
cimport numpy as cnp
import math

from engine.jit_kernels import _jit_rms_norm, _jit_silu_mul, _rope_numpy

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_fast_step(object engine, int token_id, list caches, int win_ptr, int n_ctx):
    """Mesma semântica de engine.fast_step, mas loop em Cython."""
    cdef list layers = engine.layers
    cdef object inv_freq = engine.inv_freq
    cdef int S = engine.S
    cdef int W = engine.W
    cdef int max_cap = engine.max_cap
    cdef object wbi = engine.wbi
    cdef object si = engine.si
    cdef object embed = engine.embed
    cdef object norm_w = engine.norm_w
    cdef object W_lm_t = engine.W_lm_t

    cdef object x = embed[token_id][None, None, :]
    cdef double sm0 = 0.0
    cdef list new_caches = []
    cdef int li
    cdef object layer, kc, vc, x_n, _q, _k, _v, q, k, v
    cdef object qr, kr, sc, at, out, x_n2, h, xn, logits
    cdef int B, n_heads, d_k, slot, n_sink, n_win, new_win_ptr
    cdef object ctx, pos_sink, pos_win, pos_q, pos_ctx, kc_sel, vc_sel, win_ctx

    for li in range(len(layers)):
        layer = layers[li]
        kc, vc = caches[li]
        x_n = _jit_rms_norm(x, layer.rms_attn)
        _q = x_n @ layer.W_q
        _k = x_n @ layer.W_k
        _v = x_n @ layer.W_v
        if layer.b_q is not None:
            _q = _q + layer.b_q
        if layer.b_k is not None:
            _k = _k + layer.b_k
        if layer.b_v is not None:
            _v = _v + layer.b_v
        B = 1
        n_heads = layer.n_heads
        d_k = layer.d_k
        q = _q.reshape(B, 1, n_heads, d_k).transpose(0, 2, 1, 3)
        k = _k.reshape(B, 1, n_heads, d_k).transpose(0, 2, 1, 3)
        v = _v.reshape(B, 1, n_heads, d_k).transpose(0, 2, 1, 3)

        if n_ctx <= S:
            slot = n_ctx - 1
        else:
            slot = S + win_ptr
        kc[:, :, slot:slot+1, :] = k
        vc[:, :, slot:slot+1, :] = v

        n_sink = S if n_ctx >= S else n_ctx
        n_win = n_ctx - S
        if n_win < 0:
            n_win = 0
        elif n_win > W:
            n_win = W
        if n_win < W:
            win_ctx = np.arange(S, S + n_win, dtype=np.int64)
        else:
            win_ctx = (wbi + win_ptr + 1) % W + S
        ctx = np.concatenate([si[:n_sink], win_ctx])

        if n_ctx <= max_cap:
            pos_sink = np.arange(n_sink, dtype=np.int64)
            pos_win = np.arange(S, S + len(win_ctx), dtype=np.int64)
            pos_q = np.array([n_ctx - 1], dtype=np.int64)
        else:
            pos_sink = np.arange(n_sink, dtype=np.int64)
            pos_win = np.arange(S, S + len(win_ctx), dtype=np.int64)
            pos_q = np.array([max_cap - 1], dtype=np.int64)
        pos_ctx = np.concatenate([pos_sink, pos_win])

        kc_sel = kc[:, :, ctx, :]
        vc_sel = vc[:, :, ctx, :]
        qr = _rope_numpy(q, pos_q, inv_freq)
        kr = _rope_numpy(kc_sel, pos_ctx, inv_freq)
        sc = (qr @ kr.transpose(0, 1, 3, 2)) / math.sqrt(d_k)
        sc = sc - sc.max(axis=-1, keepdims=True)
        at = np.exp(sc)
        at = at / at.sum(axis=-1, keepdims=True)
        out = (at @ vc_sel).transpose(0, 2, 1, 3).reshape(B, 1, layer.d_model) @ layer.W_o
        x = x + out
        x_n2 = _jit_rms_norm(x, layer.rms_ffn)
        h = _jit_silu_mul(x_n2 @ layer.gate, x_n2 @ layer.up) @ layer.down
        x = x + h
        new_caches.append((kc, vc))
        if li == 0:
            sm0 = float(at[:, :, 0, 0].mean() * 100)

    xn = _jit_rms_norm(x[:, 0, :], norm_w)
    logits = (W_lm_t @ xn[0].reshape(-1, 1)).T
    new_win_ptr = (win_ptr + 1) % W if n_ctx > S else win_ptr
    return logits, new_caches, new_win_ptr, sm0
