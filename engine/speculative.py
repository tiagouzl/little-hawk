"""
engine/speculative.py — Verificador causal batched sobre estado de cache existente

Fase A do roadmap speculative (PARECER_ROADMAP v1.2): provar que

    verify_chunk(engine, [t1..tk], caches, win_ptr, n_ctx)

é equivalente a k chamadas sequenciais de `engine.step()` em logits, cache,
win_ptr e posições — ANTES de qualquer política de proposal (N-gram) existir.

Diferença central vs `prefill()`: aqui o cache JÁ TEM estado. Os candidatos
formam uma única sequência causal t0→t1→…→t_{k-1} estendendo o contexto vivo;
a máscara permite à consulta j enxergar t0..t_j mas não t_{j+1..k-1}.

Restrições v1 (documentadas, fallback = loop sequencial no caller):
- FIFO apenas: evicção reservoir escolhe vítimas por step com scores que mudam
  a cada camada-0 — batching sob reservoir é problema distinto;
- chunk não pode cruzar a fronteira fill→estacionária nem começar em n_ctx<=S.
"""

import math

import numpy as np

from .jit_kernels import _jit_rms_norm, _jit_silu_mul, _rope_numpy


def can_verify(engine, n_ctx, k):
    """Condições v1 para verificação batched; caso falso, caller usa steps."""
    if engine.eviction is not None:
        return False  # reservoir: slots por step dependem de scores intra-chunk
    if n_ctx <= engine.S:
        return False  # escrevendo sinks — raro e barato sequencial
    # cruzamento fill→estacionária muda fórmula de slot no meio do chunk
    if n_ctx <= engine.max_cap < n_ctx + k:
        return False
    return True


def verify_chunk(engine, tokens, caches, win_ptr, n_ctx):
    """Verifica k candidatos num único forward batched sobre o estado atual.

    Retorna (logits [k,V], new_caches, new_win_ptr, sm_last) equivalente ao
    empilhamento de k `engine.step()`. Mutação nos buffers recebidos, como step().
    """
    ids = np.asarray(tokens, dtype=np.int64)
    k = int(ids.size)
    S, W, max_cap = engine.S, engine.W, engine.max_cap
    stationary = n_ctx > max_cap
    if not can_verify(engine, n_ctx, k):
        raise ValueError(
            f"verify_chunk indisponível para n_ctx={n_ctx}, k={k}, "
            f"eviction={engine.eviction_name!r} — usar steps sequenciais"
        )
    x = engine.embed[ids][np.newaxis]  # [1,k,d]
    new_caches = []

    # ── Slots de escrita + layout de leitura/posições ─────────────────────────
    pos_sink = np.arange(S, dtype=np.int64)
    if stationary:
        # candidatos ocupam slots consecutivos (mod W) a partir de win_ptr
        cand_slots = (win_ptr + np.arange(k, dtype=np.int64)) % W + S
        pos_q = np.full(k, max_cap - 1, dtype=np.int64)  # freeze por token

        # NOTA DE EQUIVALÊNCIA (duas sutilezas do sequencial):
        # 1) o passo de t_i ranqueia a janela com win_ptr corrente (s_i no topo,
        #    posição S+W-1) → rotação POR QUERY, não única;
        # 2) a query t_i enxerga o conteúdo STALE dos slots que serão sobrescritos
        #    pelos candidatos futuros — logo NÃO há máscara: escreve t_j, atende,
        #    avança, replicando a ordem temporal exata do step().
        rankings = [(engine.wbi + win_ptr + j + 1) % W + S for j in range(k)]
        pos_rank = np.arange(S, S + W, dtype=np.int64)
    else:
        # fase fill: posições são absolutas (slot == posição) — rotação única vale
        cand_slots = np.arange(n_ctx, n_ctx + k, dtype=np.int64)
        live = np.arange(S, n_ctx + k, dtype=np.int64)  # todos os slots vivos
        ctx = np.concatenate([np.arange(S), live])
        pos_win = live.copy()  # posição == slot no fill
        pos_q = np.arange(n_ctx, n_ctx + k, dtype=np.int64)  # posições reais
        # query j vê slots ≤ n_ctx+j (candidatos futuros bloqueados)
        allow = ctx[None, :] <= (n_ctx + np.arange(k))[:, None]

    at0 = None
    for li, layer in enumerate(engine.layers):
        kc, vc = caches[li]
        x_n = _jit_rms_norm(x, layer.rms_attn)
        _q = x_n @ layer.W_q
        _kk = x_n @ layer.W_k
        _vv = x_n @ layer.W_v
        if layer.b_q is not None:
            _q = _q + layer.b_q
            _kk = _kk + layer.b_k
            _vv = _vv + layer.b_v
        q = _q.reshape(1, k, layer.n_heads, layer.d_k).transpose(0, 2, 1, 3)
        kk = _kk.reshape(1, k, layer.n_heads, layer.d_k).transpose(0, 2, 1, 3)
        vv = _vv.reshape(1, k, layer.n_heads, layer.d_k).transpose(0, 2, 1, 3)

        if stationary:
            # write-per-query replica a ordem temporal do step(): t_j é escrito,
            # então atende. Sem máscara — a query enxerga o conteúdo STALE dos
            # slots futuros (chaves e VALORES), exatamente como o sequencial.
            v_sink = vc[:, :, :S, :]
            v_win = vc[:, :, S:, :].copy()  # snapshot stale
            sc_rows = []
            v_rows = []
            for j in range(k):
                si = cand_slots[j] - S
                kc[:, :, cand_slots[j] : cand_slots[j] + 1, :] = kk[:, :, j : j + 1, :]
                v_win[:, :, si : si + 1, :] = vv[:, :, j : j + 1, :]  # valor fresco p/ esta query
                qr_j = _rope_numpy(q[:, :, j : j + 1, :], pos_q[j : j + 1], engine.inv_freq)
                kr_j = _rope_numpy(kc[:, :, S:, :], rankings[j], engine.inv_freq)
                sc_win = (qr_j @ kr_j.transpose(0, 1, 3, 2)) / math.sqrt(layer.d_k)
                # sinks também recebem RoPE (posições 0..S-1); só a posição 0 é identidade
                ks_rope = _rope_numpy(kc[:, :, :S, :], pos_sink, engine.inv_freq)
                sk = (qr_j @ ks_rope.transpose(0, 1, 3, 2)) / math.sqrt(layer.d_k)
                sc_rows.append(np.concatenate([sk, sc_win], axis=-1))
                v_rows.append(np.concatenate([v_sink, v_win.copy()], axis=2))
            sc = np.concatenate(sc_rows, axis=2)  # [1,H,k,S+W]
            at = np.exp(sc - sc.max(axis=-1, keepdims=True))
            at /= at.sum(axis=-1, keepdims=True)
            out_rows = [at[:, :, j : j + 1, :] @ v_rows[j] for j in range(k)]
            out = np.concatenate(out_rows, axis=2).transpose(0, 2, 1, 3).reshape(1, k, engine.d_model) @ layer.W_o
        else:
            kc[:, :, cand_slots, :] = kk  # fill: máscara explícita
            vc[:, :, cand_slots, :] = vv  # cobre os futuros
            kc_sel = kc[:, :, ctx, :]
            vc_sel = vc[:, :, ctx, :]
            qr = _rope_numpy(q, pos_q, engine.inv_freq)
            kr = _rope_numpy(kc_sel, np.concatenate([pos_sink, pos_win]), engine.inv_freq)
            sc = (qr @ kr.transpose(0, 1, 3, 2)) / math.sqrt(layer.d_k)
            sc = np.where(allow[None, None], sc, np.float32(-np.inf))
            at = np.exp(sc - sc.max(axis=-1, keepdims=True))
            at /= at.sum(axis=-1, keepdims=True)
            out = (at @ vc_sel).transpose(0, 2, 1, 3).reshape(1, k, engine.d_model) @ layer.W_o

        x = x + out
        x = x + layer.ffn(x)
        if li == 0:
            at0 = at
        new_caches.append((kc, vc))

    xn_all = _jit_rms_norm(x[0], engine.norm_w)  # [k,d]
    logits = (xn_all @ engine.W_lm_t.T).astype(np.float32)  # [k,V] — todos os passos
    # telemetria: atenção ao sink da última consulta (análogo ao at[:,:,0,0] do step)
    sm_last = float(at0[0, :, -1, 0].mean() * 100)
    return logits, new_caches, (win_ptr + k) % W, sm_last
