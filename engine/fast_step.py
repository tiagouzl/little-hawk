"""
engine/fast_step.py — POC de fusão Python pura para P3 Cython

Objetivo: reduzir overhead de dispatch das 30 camadas sem perder BLAS.
Em vez de 30× chamadas a `layer.attn_step` + `layer.ffn` (450 calls/token),
inlines o loop em função única com locals binding — protótipo do que Cython
faria em C (cdef + nogil para partes não-GEMV, BLAS via C-API).

Ganho esperado: 10-20% (não 6×), pois GEMVs continuam em OpenBLAS.
Medido em `scripts/cython_poc.py`.
"""
import math

import numpy as np

from engine.jit_kernels import _jit_rms_norm, _jit_silu_mul, _rope_numpy


def fast_step(engine, token_id, caches, win_ptr, n_ctx):
    """Versão inlined de MultiLayerEngine.step — sem chamadas por camada."""
    # locals binding para evitar attr lookup no loop
    layers = engine.layers
    inv_freq = engine.inv_freq
    S, W, max_cap = engine.S, engine.W, engine.max_cap
    wbi, si = engine.wbi, engine.si
    embed = engine.embed
    norm_w = engine.norm_w
    W_lm_t = engine.W_lm_t

    x = embed[token_id][np.newaxis, np.newaxis, :]
    sm0 = 0.0
    new_caches = []

    for li, layer in enumerate(layers):
        kc, vc = caches[li]
        # ---- attn_step inlined ----
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
        n_heads, d_k = layer.n_heads, layer.d_k
        q = _q.reshape(B, 1, n_heads, d_k).transpose(0, 2, 1, 3)
        k = _k.reshape(B, 1, n_heads, d_k).transpose(0, 2, 1, 3)
        v = _v.reshape(B, 1, n_heads, d_k).transpose(0, 2, 1, 3)

        if n_ctx <= S:
            slot = n_ctx - 1
        else:
            slot = S + win_ptr
        kc[:, :, slot : slot + 1, :] = k
        vc[:, :, slot : slot + 1, :] = v

        n_sink = min(S, n_ctx)
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
        # ---- ffn inlined ----
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
