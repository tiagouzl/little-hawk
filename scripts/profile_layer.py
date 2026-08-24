#!/usr/bin/env python3
"""scripts/profile_layer.py — Breakdown por componente de um passo da camada.

Mede, com pesos reais e BLAS em 1 thread, o custo de cada estágio de
attn_step + ffn, intercalado e aquecido. Identifica onde os ~250 ms/token
das 30 camadas são gastos.
"""
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

eng = MultiLayerEngine(d_model=576, n_heads=9, n_layers=30, sink_size=4,
                       window_size=508, vocab_size=49152)
eng.load_weights("little_hawk_weights.npz")
layer = eng.layers[0]
S, W, MAX_CAP = eng.S, eng.W, eng.max_cap
inv_freq = eng.inv_freq
wbi, si = eng.wbi, eng.si

# estado típico estacionário
n_ctx = MAX_CAP + 100
win_ptr = (n_ctx - S) % W
x_t = np.random.default_rng(0).normal(0, 0.5, (1, 1, 576)).astype(np.float32)
caches = eng.init_cache()
kc_arr, vc_arr = caches[0]
# preenche cache com valores típicos
for i in range(MAX_CAP):
    kc_arr[:, :, i, :] = np.random.default_rng(i).normal(0, .1, (kc_arr.shape[1], kc_arr.shape[3])).astype(np.float32)
    vc_arr[:] = kc_arr

acc: dict[str, float] = {}
def tick(name, t0):
    acc[name] = acc.get(name, 0.0) + time.perf_counter() - t0

def profiled_attn(x_t, k_cache, v_cache, win_ptr):
    t = time.perf_counter()
    x_n = layer._rms_norm(x_t, layer.rms_attn)
    tick("1 rms_norm", t)

    t = time.perf_counter()
    _q = x_n @ layer.W_q; _k = x_n @ layer.W_k; _v = x_n @ layer.W_v
    tick("2 proj QKV (gemv)", t)

    t = time.perf_counter()
    if layer.b_q is not None: _q = _q + layer.b_q
    if layer.b_k is not None: _k = _k + layer.b_k
    if layer.b_v is not None: _v = _v + layer.b_v
    tick("3 bias", t)

    t = time.perf_counter()
    B = 1
    q = _q.reshape(B, 1, layer.n_heads, layer.d_k).transpose(0, 2, 1, 3)
    k = _k.reshape(B, 1, layer.n_heads, layer.d_k).transpose(0, 2, 1, 3)
    v = _v.reshape(B, 1, layer.n_heads, layer.d_k).transpose(0, 2, 1, 3)
    tick("4 reshape/transp qkv", t)

    t = time.perf_counter()
    if n_ctx <= S: slot = n_ctx - 1
    else: slot = S + win_ptr
    k_cache[:, :, slot:slot + 1, :] = k
    v_cache[:, :, slot:slot + 1, :] = v
    tick("5 escrita cache", t)

    t = time.perf_counter()
    n_sink = min(n_ctx, S); n_win = max(0, min(n_ctx - S, W))
    if n_win < W: win_ctx = np.arange(S, S + n_win, dtype=np.int64)
    else: win_ctx = (wbi + win_ptr + 1) % W + S
    ctx = np.concatenate([si[:n_sink], win_ctx])
    tick("6 idx contexto", t)

    t = time.perf_counter()
    if n_ctx <= MAX_CAP:
        pos_sink = np.arange(n_sink, dtype=np.int64)
        pos_win = np.arange(S, S + len(win_ctx), dtype=np.int64)
        pos_q = np.array([n_ctx - 1], dtype=np.int64)
    else:
        pos_sink = np.arange(n_sink, dtype=np.int64)
        pos_win = np.arange(S, S + len(win_ctx), dtype=np.int64)
        pos_q = np.array([MAX_CAP - 1], dtype=np.int64)
    pos_ctx = np.concatenate([pos_sink, pos_win])
    tick("7 posições", t)

    t = time.perf_counter()
    kc = k_cache[:, :, ctx, :]; vc = v_cache[:, :, ctx, :]
    tick("8 gather kc/vc", t)

    t = time.perf_counter()
    ang = np.outer(pos_q.astype(np.float32), inv_freq)
    emb = np.concatenate([ang, ang], axis=-1)[np.newaxis, np.newaxis]
    s_, c_ = np.sin(emb), np.cos(emb)
    half = q.shape[-1] // 2
    x1, x2 = q[..., :half], q[..., half:]
    qr = q * c_ + np.concatenate([-x2, x1], axis=-1) * s_
    angc = np.outer(pos_ctx.astype(np.float32), inv_freq)
    embc = np.concatenate([angc, angc], axis=-1)[np.newaxis, np.newaxis]
    sc_, cc_ = np.sin(embc), np.cos(embc)
    y1, y2 = kc[..., :half], kc[..., half:]
    kr = kc * cc_ + np.concatenate([-y2, y1], axis=-1) * sc_
    tick("9 rope (sin/cos+mult)", t)

    t = time.perf_counter()
    sc = (qr @ kr.transpose(0, 1, 3, 2)) / np.sqrt(layer.d_k)
    tick("10 scores qk^T", t)

    t = time.perf_counter()
    sc = sc - sc.max(axis=-1, keepdims=True)
    at = np.exp(sc); at /= at.sum(axis=-1, keepdims=True)
    tick("11 softmax", t)

    t = time.perf_counter()
    out = (at @ vc).transpose(0, 2, 1, 3).reshape(B, 1, layer.d_model) @ layer.W_o
    tick("12 @vc + W_o", t)
    return out

def profiled_ffn(x):
    t = time.perf_counter(); x_n = layer._rms_norm(x, layer.rms_ffn); tick("13 ffn rms", t)
    t = time.perf_counter(); g = x_n @ layer.gate; u = x_n @ layer.up; tick("14 ffn gemv gate/up", t)
    t = time.perf_counter(); h = (g / (1.0 + np.exp(-g))) * u; tick("15 silu*mul", t)
    t = time.perf_counter(); r = h @ layer.down; tick("16 ffn gemv down", t)
    return r

N, WARM = 200, 30
for i in range(WARM): profiled_attn(x_t, kc_arr, vc_arr, win_ptr)
for k in list(acc): acc[k] = 0.0
t_all = time.perf_counter()
for i in range(N):
    out_a = profiled_attn(x_t, kc_arr, vc_arr, win_ptr)
    out_f = profiled_ffn(out_a)
total_ms = (time.perf_counter() - t_all) / N * 1000

print(f"{'estágio':<26}{'µs/passo':>10}{'% camada':>10}")
order = sorted(acc.items(), key=lambda kv: -kv[1])
for name, secs in order:
    us = secs / N * 1e6
    print(f"{name:<26}{us:>10.1f}{us / (total_ms * 1000) * 100:>9.1f}%")
print(f"\nTOTAL 1 camada: {total_ms:.2f} ms  → 30 camadas ≈ {total_ms * 30:.0f} ms/token")
