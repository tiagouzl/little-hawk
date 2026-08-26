"""
runtime/speculative.py — Fase B: N-gram Speculator + métricas E2

Pré-registro: ANALISE.md §21.2. Desacoplado do Sampler — rascunhos são
verificados por argmax do modelo-alvo; o Sampler só escolhe o token bônus.

Economia por rodada (v1, honesta): SEMPRE 2 forwards do alvo (verify_chunk +
step do bônus), emitidos = m+1 onde m = rascunhos aceitos ⇒ speedup médio ≈
avg(m+1)/2 e rodadas com m=0 custam 0.5×. Rollback por restauração dos k
slots candidatos (snapshot O(k·d_k)), custo contado nas métricas.
"""

import time

import numpy as np


class NGramSpeculator:
    """Tabela n-gram → próximo token (última ocorrência vence; determinística)."""

    def __init__(self, n: int = 3):
        assert n >= 2
        self.n = n
        self.table: dict[tuple, int] = {}

    def observe(self, window):
        """Registra o par formado por uma janela de exatamente n tokens
        (os n-1 primeiros são o contexto, o último é a continuação)."""
        w = list(window)
        assert len(w) == self.n
        self.table[tuple(w[:-1])] = w[-1]

    def propose(self, tail_ids, k: int) -> list[int]:
        """Propõe até k candidatos encadeados a partir da cauda do contexto."""
        out = []
        buf = list(tail_ids)[-(self.n - 1):]
        for _ in range(k):
            nxt = self.table.get(tuple(buf))
            if nxt is None:
                break
            out.append(int(nxt))
            buf = (buf + [nxt])[-(self.n - 1):]   # mantém janela n-1 — sem isso a cadeia morre
        return out


class SpeculativeStats:
    """Métricas obrigatórias E2 (parecer v1.1, emenda E2)."""

    def __init__(self):
        self.rounds = 0  # rodadas especulativas (com rascunhos)
        self.fallback_steps = 0  # forwards em modo step simples
        self.forwards = 0  # avaliações do alvo (verify conta 1; step conta 1)
        self.proposed = 0
        self.accepted = 0
        self.emitted = 0
        self.rollback_restores = 0
        self.wall_s = 0.0

    @property
    def acceptance_rate(self):
        return self.accepted / self.proposed if self.proposed else float("nan")

    def as_dict(self):
        return {
            "rounds": self.rounds,
            "fallback_steps": self.fallback_steps,
            "forwards": self.forwards,
            "proposed": self.proposed,
            "accepted": self.accepted,
            "emitted": self.emitted,
            "acceptance_rate": round(self.acceptance_rate, 4) if self.proposed else None,
            "tokens_per_forward": round(self.emitted / self.forwards, 3) if self.forwards else None,
            "rollback_restores": self.rollback_restores,
            "wall_s": round(self.wall_s, 2),
        }


def restore_slot(caches, li, slot, snap):
    """Restaura um único slot a partir do snapshot pré-verificação."""
    kc_snap, vc_snap = snap[li]
    caches[li][0][:, :, slot : slot + 1, :] = kc_snap[:, :, slot : slot + 1, :]
    caches[li][1][:, :, slot : slot + 1, :] = vc_snap[:, :, slot : slot + 1, :]
