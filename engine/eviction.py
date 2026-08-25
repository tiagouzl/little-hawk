"""
engine/eviction.py — Políticas de evicção para StreamingKVCache

FIFO (atual): win_ptr circular, sobrescreve o mais antigo sem julgamento.
Nexus (reservoir ponderado): usa scores de atenção para decidir vítima,
mantendo sinks + janela recente intactos e gerindo o anel intermediário
via reservoir sampling ponderado (arXiv 2606.23961).
"""
import numpy as np


class FIFOEviction:
    """Política atual — FIFO puro, O(1) e determinística."""
    def __init__(self, S=4, W=508):
        self.S, self.W = S, W
        self.win_ptr = 0

    def next_slot(self, n_ctx):
        if n_ctx <= self.S:
            return n_ctx - 1, self.win_ptr
        slot = self.S + self.win_ptr
        new_ptr = (self.win_ptr + 1) % self.W
        return slot, new_ptr

    def update_scores(self, scores, ctx, at_weights):
        pass  # FIFO ignora scores


class NexusEviction:
    """
    Reservoir ponderado — mantém S sinks + R recentes intactos,
    anel intermediário (W-R) gerido por scores de atenção.

    - scores: array [max_cap] com massa de atenção acumulada (EMA)
    - R: tamanho da janela recente protegida (ex: 64)
    - alpha: decaimento EMA (0.9 = memória longa)
    """
    def __init__(self, S=4, W=508, R=64, alpha=0.9, seed=42):
        # R adaptativo: para W pequeno (demo 28) usa W//2, senão 64
        R = min(R, max(1, W // 2))
        self.S, self.W, self.R = S, W, R
        self.alpha = alpha
        self.max_cap = S + W
        self.win_ptr = 0
        self.rng = np.random.default_rng(seed)
        # scores por slot (apenas janela, sinks não participam)
        self.scores = np.zeros(self.max_cap, dtype=np.float32)
        # contador de inserções no anel intermediário
        self.n_reservoir = 0

    def next_slot(self, n_ctx):
        # Fase de enchimento: sequencial como FIFO
        if n_ctx <= self.max_cap:
            if n_ctx <= self.S:
                return n_ctx - 1, self.win_ptr
            # Ainda enchendo a janela — sequencial
            if n_ctx <= self.S + self.W:
                slot = n_ctx - 1
                # win_ptr só avança após S
                new_ptr = (self.win_ptr + 1) % self.W if n_ctx > self.S else self.win_ptr
                # Para n_ctx <= max_cap, win_ptr reflete (n_ctx - S) % W como antes
                # Mas aqui n_ctx <= max_cap então win_ptr = n_ctx - S -1? Mantém compatível:
                # O engine calcula win_ptr externamente, aqui só retornamos slot
                # e novo win_ptr para compatibilidade — para reservoir na fase estacionária
                # o win_ptr é usado apenas para janela recente.
                return slot, new_ptr
        # Fase estacionária: n_ctx > max_cap — escolher vítima
        # Protege R slots mais recentes (cauda da janela)
        # Slots candidatos: S .. S+W-R-1 (anel intermediário)
        # Vítima escolhida por reservoir ponderado inverso ao score
        # (menor score = maior chance de evicção)
        # Implementação simplificada: amostra ponderada onde peso = 1/(score+eps)
        # ou score baixo = vítima.
        # Para evitar O(W) por step em Python, usamos amostragem entre os piores k
        # Aqui: escolhe uniformemente entre os 32 menores scores no anel
        candidate_start = self.S
        candidate_end = self.S + self.W - self.R
        # Se W-R <=0, cai no FIFO da janela recente
        if candidate_end <= candidate_start:
            slot = self.S + self.win_ptr
            return slot, (self.win_ptr + 1) % self.W
        # Pega scores do anel intermediário
        ring_scores = self.scores[candidate_start:candidate_end]
        # Escolhe entre os menores — Nexus usa p ∝ atenção, nós invertemos para evicção
        # k = min(32, len)
        k = min(32, len(ring_scores))
        # Índices dos k menores scores
        idx_sorted = np.argpartition(ring_scores, k-1)[:k]
        # Escolhe uniformemente entre eles (poderia ser ponderado por 1/score)
        # Para determinismo com seed, usa rng
        chosen = self.rng.choice(idx_sorted)
        slot = candidate_start + int(chosen)
        # win_ptr avança apenas para janela recente (não para reservoir)
        # Mantém win_ptr circular para os R recentes
        new_ptr = (self.win_ptr + 1) % self.W
        self.n_reservoir += 1
        return slot, new_ptr

    def update_scores(self, ctx, at_weights):
        """
        Atualiza EMA dos scores com pesos de atenção do step atual.
        ctx: índices de slots no cache (ex: [0,1,2,3, 4..511])
        at_weights: [1, n_heads, 1, n_ctx] (nosso caso) ou variantes — média sobre heads
        """
        # Normaliza para [n_ctx]
        w = np.asarray(at_weights)
        # at tipicamente [1, H, 1, n_ctx] → squeeze e média
        while w.ndim > 1 and w.shape[0] == 1:
            w = w[0]
        if w.ndim == 2:
            # [H, n_ctx] → média heads
            w = w.mean(axis=0)
        elif w.ndim == 3:
            # [H, 1, n_ctx] ou [1, H, n_ctx]
            w = w.mean(axis=tuple(range(w.ndim-1)))
        w = w.reshape(-1)  # [n_ctx]
        # EMA: scores[ctx] = alpha*scores[ctx] + (1-alpha)*w
        if len(ctx) != len(w):
            m = min(len(ctx), len(w))
            ctx = ctx[:m]
            w = w[:m]
        self.scores[ctx] = self.alpha * self.scores[ctx] + (1 - self.alpha) * w.astype(np.float32)
        # Sinks mantêm score alto para nunca serem vítimas (não estão no anel)
        self.scores[:self.S] = 1.0
