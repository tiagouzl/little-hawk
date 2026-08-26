#!/usr/bin/env python3
"""
B0 — Microbenchmark instrumentado de verify_chunk (stationary branch).

Mede, POR COMPONENTE e POR CAMADA, o custo de verify_chunk para k={1,2,4,8}.
Reproduz exatamente as condições do benchmark que produziu o 1.90×:
  - 135M real (little_hawk_weights.npz)
  - FIFO (eviction=None no engine)
  - stationary (n_ctx > max_cap = 512)
  - R=20 repetições, mediana + IQR

SAÍDA: tabela no terminal + JSON opcional.
NENHUMA alteração em código de produção.

Protocolo B0 (PARECER_ROADMAP): medir primeiro, depois decidir.
"""
import json
import math
import os
import sys
import time
import copy

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.engine import MultiLayerEngine
from engine.jit_kernels import _jit_rms_norm, _rope_numpy
from runtime.tokenizer import BPETokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(ROOT, "little_hawk_weights.npz")
META = os.path.join(ROOT, "little_hawk_weights_meta.json")
R = 20  # repetições
KS = [1, 2, 4, 8]


def setup_engine_and_cache(n_ctx_target=600):
    """Carrega engine real, preenche cache até n_ctx_target em modo stationary."""
    with open(META, encoding="utf-8") as f:
        meta = json.load(f)
    eng = MultiLayerEngine(
        d_model=int(meta["d_model"]),
        n_heads=int(meta["n_heads"]),
        n_layers=int(meta["n_layers"]),
        sink_size=4,
        window_size=508,
        vocab_size=int(meta["vocab_size"]),
    )
    eng.load_weights(NPZ)
    caches = eng.init_cache()
    rng = np.random.default_rng(42)
    win_ptr = 0
    # Preenche cache com n_ctx_target tokens reais via step()
    for n_ctx in range(1, n_ctx_target + 1):
        tid = int(rng.integers(3, meta["vocab_size"]))
        _, caches, win_ptr, _ = eng.step(tid, caches, win_ptr, n_ctx)
    return eng, caches, win_ptr, n_ctx_target


def bench_component_stationary(eng, layer, caches_li, cand_slots, kk, vv,
                                pos_q, pos_sink, rankings, k, inv_freq):
    """
    Mede os componentes DA CAMADA no branch stationary, exatamente como
    o loop j=0..k-1 em verify_chunk faz.
    Retorna dict {componente: tempo_ns} para esta camada/iteração.
    """
    kc, vc = caches_li
    S, W, max_cap = eng.S, eng.W, eng.max_cap
    d_k = layer.d_k

    times = {c: 0 for c in ["rms_norm", "qkv_proj", "rope_q", "rope_k",
                              "qk_score", "softmax", "av_out", "proj_out", "write_cache"]}

    # ── rms_norm ──
    t0 = time.perf_counter_ns()
    x_n = _jit_rms_norm(kk[:, :, :k, :], layer.rms_attn)  # kkk é o input embed; aqui usamos dummy
    times["rms_norm"] += time.perf_counter_ns() - t0

    # Na verdade, para medir corretamente, precisamos do x_n real.
    # Vamos medir os componentes isolados do loop j=0..k-1:
    # Preparar snapshots
    v_sink = vc[:, :, :S, :].copy()
    v_win = vc[:, :, S:, :].copy()

    sc_rows = []
    v_rows = []
    for j in range(k):
        si = cand_slots[j] - S

        # ── write_cache (kc write) ──
        t0 = time.perf_counter_ns()
        kc[:, :, cand_slots[j]:cand_slots[j]+1, :] = kk[:, :, j:j+1, :]
        times["write_cache"] += time.perf_counter_ns() - t0

        # ── write valor ──
        v_win_copy = v_win.copy()
        v_win_copy[:, :, si:si+1, :] = vv[:, :, j:j+1, :]

        # ── rope_q (1 query) ──
        qr_j_input = kk[:, :, j:j+1, :]  # dummy: shape [1,1,H,d_k] — na verdade q[:,j:j+1]
        t0 = time.perf_counter_ns()
        qr_j = _rope_numpy(qr_j_input, pos_q[j:j+1], inv_freq)
        times["rope_q"] += time.perf_counter_ns() - t0

        # ── rope_k (S+W keys) ──
        kc_win = kc[:, :, S:, :].copy()
        t0 = time.perf_counter_ns()
        kr_j = _rope_numpy(kc_win, rankings[j], inv_freq)
        times["rope_k"] += time.perf_counter_ns() - t0

        # ── qk_score (sink + window) ──
        ks_rope_input = kc[:, :, :S, :].copy()
        t0 = time.perf_counter_ns()
        ks_rope = _rope_numpy(ks_rope_input, pos_sink, inv_freq)
        sc_win = (qr_j @ kr_j.transpose(0, 1, 3, 2)) / math.sqrt(d_k)
        sk = (qr_j @ ks_rope.transpose(0, 1, 3, 2)) / math.sqrt(d_k)
        sc_j = np.concatenate([sk, sc_win], axis=-1)
        times["qk_score"] += time.perf_counter_ns() - t0

        sc_rows.append(sc_j)
        v_rows.append(np.concatenate([v_sink, v_win_copy], axis=2))

    # ── softmax (todos os k de uma vez) ──
    sc_all = np.concatenate(sc_rows, axis=2)  # [1,H,k,S+W]
    t0 = time.perf_counter_ns()
    at = np.exp(sc_all - sc_all.max(axis=-1, keepdims=True))
    at /= at.sum(axis=-1, keepdims=True)
    times["softmax"] += time.perf_counter_ns() - t0

    # ── av_out (k queries) ──
    t0 = time.perf_counter_ns()
    out_rows = [at[:, :, j:j+1, :] @ v_rows[j] for j in range(k)]
    times["av_out"] += time.perf_counter_ns() - t0

    return times


def bench_verify_chunk_instrumented(eng, tokens, caches, win_ptr, n_ctx, k):
    """
    Executa verify_chunk com medição granular de cada componente.
    Retorna dict de componentes com tempos totais (ns) em TODAS as camadas.
    """
    ids = np.asarray(tokens[:k], dtype=np.int64)
    S, W, max_cap = eng.S, eng.W, eng.max_cap

    # Preparar cache profunda (não mutar o original)
    caches = [(kc.copy(), vc.copy()) for kc, vc in caches]
    x = eng.embed[ids][np.newaxis]  # [1,k,d_model]

    pos_sink = np.arange(S, dtype=np.int64)
    cand_slots = (win_ptr + np.arange(k, dtype=np.int64)) % W + S
    pos_q = np.full(k, max_cap - 1, dtype=np.int64)
    rankings = [(eng.wbi + win_ptr + j + 1) % W + S for j in range(k)]

    totals = {c: 0 for c in ["rms_norm", "qkv_proj", "reshape_transpose",
                               "rope_q", "rope_k", "qk_score", "softmax",
                               "av_out", "proj_out", "ffn", "write_cache"]}

    for li, layer in enumerate(eng.layers):
        kc, vc = caches[li]

        # ── rms_norm (UMA VEZ para todas as k queries) ──
        t0 = time.perf_counter_ns()
        x_n = _jit_rms_norm(x, layer.rms_attn)
        totals["rms_norm"] += time.perf_counter_ns() - t0

        # ── qkv_proj (3 GEMMs, UMA VEZ para todas as k queries) ──
        t0 = time.perf_counter_ns()
        _q = x_n @ layer.W_q
        _k = x_n @ layer.W_k
        _v = x_n @ layer.W_v
        if layer.b_q is not None:
            _q = _q + layer.b_q
            _k = _k + layer.b_k
            _v = _v + layer.b_v
        totals["qkv_proj"] += time.perf_counter_ns() - t0

        # ── reshape_transpose (UMA VEZ) ──
        t0 = time.perf_counter_ns()
        q = _q.reshape(1, k, layer.n_heads, layer.d_k).transpose(0, 2, 1, 3)
        kk = _k.reshape(1, k, layer.n_heads, layer.d_k).transpose(0, 2, 1, 3)
        vv = _v.reshape(1, k, layer.n_heads, layer.d_k).transpose(0, 2, 1, 3)
        totals["reshape_transpose"] += time.perf_counter_ns() - t0

        # ── stationary branch: loop j=0..k-1 ──
        v_sink = vc[:, :, :S, :]
        v_win = vc[:, :, S:, :].copy()
        sc_rows = []
        v_rows = []

        for j in range(k):
            si = cand_slots[j] - S

            # ── write_cache (key + value para slot candidato) ──
            t0 = time.perf_counter_ns()
            kc[:, :, cand_slots[j]:cand_slots[j]+1, :] = kk[:, :, j:j+1, :]
            v_win[:, :, si:si+1, :] = vv[:, :, j:j+1, :]
            totals["write_cache"] += time.perf_counter_ns() - t0

            # ── rope_q (1 query, posição congelada) ──
            t0 = time.perf_counter_ns()
            qr_j = _rope_numpy(q[:, :, j:j+1, :], pos_q[j:j+1], eng.inv_freq)
            totals["rope_q"] += time.perf_counter_ns() - t0

            # ── rope_k (S+W keys da janela) ──
            t0 = time.perf_counter_ns()
            kr_j = _rope_numpy(kc[:, :, S:, :], rankings[j], eng.inv_freq)
            totals["rope_k"] += time.perf_counter_ns() - t0

            # ── qk_score: rope_sink + 2 GEMMs (sink_scores + win_scores) ──
            t0 = time.perf_counter_ns()
            ks_rope = _rope_numpy(kc[:, :, :S, :], pos_sink, eng.inv_freq)
            sc_win = (qr_j @ kr_j.transpose(0, 1, 3, 2)) / math.sqrt(layer.d_k)
            sk = (qr_j @ ks_rope.transpose(0, 1, 3, 2)) / math.sqrt(layer.d_k)
            sc_j = np.concatenate([sk, sc_win], axis=-1)
            totals["qk_score"] += time.perf_counter_ns() - t0

            sc_rows.append(sc_j)
            v_rows.append(np.concatenate([v_sink, v_win.copy()], axis=2))

        # ── softmax (batch: k × S+W) ──
        sc_all = np.concatenate(sc_rows, axis=2)
        t0 = time.perf_counter_ns()
        at = np.exp(sc_all - sc_all.max(axis=-1, keepdims=True))
        at /= at.sum(axis=-1, keepdims=True)
        totals["softmax"] += time.perf_counter_ns() - t0

        # ── av_out: k individual matmuls + concat + proj_out ──
        t0 = time.perf_counter_ns()
        out_rows = [at[:, :, j:j+1, :] @ v_rows[j] for j in range(k)]
        out = np.concatenate(out_rows, axis=2).transpose(0, 2, 1, 3).reshape(1, k, eng.d_model) @ layer.W_o
        totals["av_out"] += time.perf_counter_ns() - t0

        # ── residual + FFN ──
        t0 = time.perf_counter_ns()
        x = x + out
        x = x + layer.ffn(x)
        totals["ffn"] += time.perf_counter_ns() - t0

    return totals


def bench_step_single(eng, tokens, caches, win_ptr, n_ctx, k):
    """Mede k steps sequenciais (baseline) para comparar com verify_chunk."""
    caches = [(kc.copy(), vc.copy()) for kc, vc in caches]
    t0 = time.perf_counter_ns()
    wp = win_ptr
    nc = n_ctx
    for i in range(k):
        _, caches, wp, _ = eng.step(int(tokens[i]), caches, wp, nc + i)
    elapsed = time.perf_counter_ns() - t0
    return elapsed


def main():
    print("=" * 72)
    print("B0 — Microbenchmark instrumentado de verify_chunk (stationary)")
    print("=" * 72)
    print(f"Modelo:  little_hawk_weights.npz (135M)")
    print(f"Dimensões: d_model=576, n_heads=9, d_k=64, n_layers=30")
    print(f"S={4}, W={508}, max_cap={512}")
    print(f"k values: {KS}")
    print(f"Repetições: {R}")
    print()

    print("Carregando engine e preenchendo cache...")
    eng, caches, win_ptr, n_ctx = setup_engine_and_cache(n_ctx_target=600)
    print(f"Cache pronto: n_ctx={n_ctx}, win_ptr={win_ptr}, stationary={n_ctx > eng.max_cap}")
    print()

    # Tokens candidatos (dummy, não importa a qualidade — medimos custo)
    rng = np.random.default_rng(42)
    all_tokens = rng.integers(3, eng.V, size=max(KS) + 100).tolist()

    results = {}

    for k in KS:
        print(f"--- k={k} ---")
        step_times = []
        verify_times = []
        component_lists = []

        for rep in range(R):
            # Deep copy caches para cada repetição
            caches_rep = [(kc.copy(), vc.copy()) for kc, vc in caches]

            # Baseline: k steps sequenciais
            t_step = bench_step_single(eng, all_tokens, caches, win_ptr, n_ctx, k)
            step_times.append(t_step)

            # verify_chunk instrumentado
            comps = bench_verify_chunk_instrumented(eng, all_tokens, caches_rep, win_ptr, n_ctx, k)
            component_lists.append(comps)

            # Tempo total do verify = soma dos componentes
            t_verify = sum(comps.values())
            verify_times.append(t_verify)

        step_arr = np.array(step_times, dtype=np.float64)
        ver_arr = np.array(verify_times, dtype=np.float64)

        median_step = np.median(step_arr) / 1e6
        median_ver = np.median(ver_arr) / 1e6
        iqr_step_lo = np.percentile(step_arr, 25) / 1e6
        iqr_step_hi = np.percentile(step_arr, 75) / 1e6
        iqr_ver_lo = np.percentile(ver_arr, 25) / 1e6
        iqr_ver_hi = np.percentile(ver_arr, 75) / 1e6
        ratio = median_ver / median_step if median_step > 0 else float("inf")

        print(f"  step    : {median_step:.3f} ms  [IQR {iqr_step_lo:.3f}–{iqr_step_hi:.3f}]")
        print(f"  verify  : {median_ver:.3f} ms  [IQR {iqr_ver_lo:.3f}–{iqr_ver_hi:.3f}]")
        print(f"  ratio   : {ratio:.3f}×")

        # Agregar componentes (mediana sobre R repetições)
        comp_names = list(component_lists[0].keys())
        comp_medians = {}
        for cn in comp_names:
            vals = np.array([cl[cn] for cl in component_lists], dtype=np.float64)
            comp_medians[cn] = float(np.median(vals))

        total_ns = sum(comp_medians.values())
        total_ms = total_ns / 1e6

        print(f"  Component breakdown (mediana sobre {R} reps, total={total_ms:.3f} ms):")
        for cn in comp_names:
            ms_val = comp_medians[cn] / 1e6
            pct = (comp_medians[cn] / total_ns * 100) if total_ns > 0 else 0
            print(f"    {cn:20s}  {ms_val:8.3f} ms  {pct:5.1f}%")
        print()

        results[k] = {
            "step_ms": median_step,
            "step_iqr": [iqr_step_lo, iqr_step_hi],
            "verify_ms": median_ver,
            "verify_iqr": [iqr_ver_lo, iqr_ver_hi],
            "ratio": ratio,
            "components_ms": {cn: comp_medians[cn] / 1e6 for cn in comp_names},
            "components_pct": {cn: (comp_medians[cn] / total_ns * 100) if total_ns > 0 else 0
                               for cn in comp_names},
        }

    # ── Tabela resumo ──
    print("=" * 72)
    print("RESUMO")
    print("=" * 72)
    header = f"{'k':>3s} | {'step (ms)':>12s} | {'verify (ms)':>12s} | {'ratio':>7s}"
    print(header)
    print("-" * len(header))
    for k in KS:
        r = results[k]
        print(f"{k:3d} | {r['step_ms']:10.3f} ±{r['step_iqr'][1]-r['step_iqr'][0]:5.3f}"
              f" | {r['verify_ms']:10.3f} ±{r['verify_iqr'][1]-r['verify_iqr'][0]:5.3f}"
              f" | {r['ratio']:6.3f}×")

    # ── Percentual por componente (k=4 = baseline do 1.90×) ──
    print()
    print(f"Component breakdown (% of verify total) for k=4:")
    r4 = results[4]
    sorted_comps = sorted(r4["components_pct"].items(), key=lambda x: -x[1])
    for cn, pct in sorted_comps:
        ms = r4["components_ms"][cn]
        print(f"  {cn:20s}  {ms:8.3f} ms  {pct:5.1f}%")

    # ── Veredicto B0 ──
    print()
    print("=" * 72)
    print("B0 VERDICT")
    print("=" * 72)
    r4 = results[4]
    sorted_comps = sorted(r4["components_pct"].items(), key=lambda x: -x[1])
    dominant_name, dominant_pct = sorted_comps[0]
    second_name, second_pct = sorted_comps[1] if len(sorted_comps) > 1 else ("N/A", 0)

    # Scaling factor k=1 → k=8
    if results[1]["verify_ms"] > 0:
        scaling = results[8]["verify_ms"] / results[1]["verify_ms"]
    else:
        scaling = float("inf")

    print(f"Dominant component: {dominant_name}")
    print(f"Share (k=4): {dominant_pct:.1f}%")
    print(f"Second:      {second_name} = {second_pct:.1f}%")
    print(f"Scaling k=1→8: {scaling:.2f}×")
    print()

    if dominant_pct >= 45:
        print(f"B1 candidate: batch/optimize {dominant_name}")
    elif dominant_pct >= 30 and second_pct >= 25:
        print(f"B1 candidates: {dominant_name} ({dominant_pct:.1f}%) + {second_name} ({second_pct:.1f}%)")
    else:
        print("NO CLEAR LOCAL TARGET.")
        print("Largest components are spread — overhead/granularity may dominate.")

    print()
    print("NO CODE CHANGES MADE.")
    print("WAITING FOR B1 DECISION.")

    # Salvar JSON
    out_path = os.path.join(ROOT, "bench_verify_b0_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados salvos em {out_path}")


if __name__ == "__main__":
    main()
