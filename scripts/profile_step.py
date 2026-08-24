import time
import numpy as np
from engine.engine import MultiLayerEngine
from runtime.tokenizer import BPETokenizer

meta_path = "little_hawk_weights_meta.json"
tok = BPETokenizer()
tok.load_donor_vocab(meta_path)

import json
with open(meta_path) as f:
    meta = json.load(f)

eng = MultiLayerEngine(
    d_model=int(meta["d_model"]),
    n_heads=int(meta["n_heads"]),
    n_layers=int(meta["n_layers"]),
    sink_size=4,
    window_size=508,
    vocab_size=int(meta["vocab_size"])
)
eng.load_weights("little_hawk_weights.npz")
caches = eng.init_cache()

# Warmup JIT
for _ in range(5):
    eng.step(1, caches, 0, 1)

# Timings individuais
timings = {
    "embed": 0.0,
    "qkv_gemv": 0.0,
    "rope": 0.0,
    "attn_core": 0.0,
    "wo_gemv": 0.0,
    "ffn_gate_up": 0.0,
    "silu_mul": 0.0,
    "ffn_down": 0.0,
    "lm_head": 0.0,
}

N = 100
for step in range(N):
    tid = step % 500 + 1
    t0 = time.perf_counter()
    x = eng.embed[tid][np.newaxis, np.newaxis, :]
    timings["embed"] += time.perf_counter() - t0

    for li, layer in enumerate(eng.layers):
        kc, vc = caches[li]
        x_n = layer._rms_norm(x, layer.rms_attn)

        t_gemv = time.perf_counter()
        _q = x_n @ layer.W_q
        _k = x_n @ layer.W_k
        _v = x_n @ layer.W_v
        timings["qkv_gemv"] += time.perf_counter() - t_gemv

        B = 1
        q = _q.reshape(B, 1, layer.n_heads, layer.d_k).transpose(0, 2, 1, 3)
        k = _k.reshape(B, 1, layer.n_heads, layer.d_k).transpose(0, 2, 1, 3)
        v = _v.reshape(B, 1, layer.n_heads, layer.d_k).transpose(0, 2, 1, 3)

        kc[:, :, step:step+1, :] = k
        vc[:, :, step:step+1, :] = v

        pos_sink = np.arange(min(step+1, 4), dtype=np.int64)
        pos_win = np.arange(4, 4 + max(0, min(step+1-4, 508)), dtype=np.int64)
        pos_ctx = np.concatenate([pos_sink, pos_win])
        pos_q = np.array([step], dtype=np.int64)

        t_rope = time.perf_counter()
        from engine.jit_kernels import _jit_rope_rotate
        qr = _jit_rope_rotate(q, pos_q, eng.inv_freq)
        kr = _jit_rope_rotate(kc[:, :, :step+1, :], pos_ctx, eng.inv_freq)
        timings["rope"] += time.perf_counter() - t_rope

        t_attn = time.perf_counter()
        sc = (qr @ kr.transpose(0, 1, 3, 2)) / np.sqrt(layer.d_k)
        sc = sc - sc.max(axis=-1, keepdims=True)
        at = np.exp(sc)
        at /= at.sum(axis=-1, keepdims=True)
        out = (at @ vc[:, :, :step+1, :]).transpose(0, 2, 1, 3).reshape(B, 1, layer.d_model)
        timings["attn_core"] += time.perf_counter() - t_attn

        t_wo = time.perf_counter()
        ao = out @ layer.W_o
        x = x + ao
        timings["wo_gemv"] += time.perf_counter() - t_wo

        t_ffn1 = time.perf_counter()
        x_ffn_n = layer._rms_norm(x, layer.rms_ffn)
        g_proj = x_ffn_n @ layer.gate
        u_proj = x_ffn_n @ layer.up
        timings["ffn_gate_up"] += time.perf_counter() - t_ffn1

        t_silu = time.perf_counter()
        from engine.jit_kernels import _jit_silu_mul
        act = _jit_silu_mul(g_proj, u_proj)
        timings["silu_mul"] += time.perf_counter() - t_silu

        t_ffn2 = time.perf_counter()
        x = x + act @ layer.down
        timings["ffn_down"] += time.perf_counter() - t_ffn2

    t_lm = time.perf_counter()
    xn = eng._rms_norm(x[:, 0, :], eng.norm_w)
    logits = xn @ eng.W_lm
    timings["lm_head"] += time.perf_counter() - t_lm

print("\n--- PERFIL DE TEMPO POR ETAPA (Total por token) ---")
total = sum(timings.values())
for k, v in timings.items():
    ms_tok = (v / N) * 1000
    pct = (v / total) * 100
    print(f"{k:<15}: {ms_tok:6.2f} ms/tok ({pct:5.1f}%)")
print(f"TOTAL MEDIDO   : {(total / N) * 1000:6.2f} ms/tok")
