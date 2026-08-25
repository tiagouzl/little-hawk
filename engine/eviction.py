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
        # Ordem de recência dos slots vivos (exclui sinks), mais antigo → mais novo.
        # Fonte da verdade para LEITURA (ctx_array) e ESCrita (next_slot) — corrige a
        # inconsistência em que win_ptr FIFO era usado para ler após escritas por override.
        self.order: list[int] = []

    def sync_after_fill(self, n_filled):
        """Sincroniza a ordem após prefill batched (slots 0..n_filled-1 escritos sequenciais)."""
        self.order = list(range(self.S, max(self.S, min(n_filled, self.max_cap))))

    def reset(self):
        """Zera estado entre gerações — scores são por slot-id e sem reset uma
        geração herda a proteção acumulada pelos tokens da geração anterior."""
        self.order = []
        self.scores = np.zeros(self.max_cap, dtype=np.float32)
        self.n_reservoir = 0
        self.win_ptr = 0

    def ctx_array(self):
        """Array completo de slots a atender: sinks + vivos em ordem de recência."""
        import numpy as _np
        return _np.array(list(range(self.S)) + self.order, dtype=_np.int64)

    def next_slot(self, n_ctx):
        # Fase de sinks: slots 0..S-1 sequenciais, fora da ordem de janela
        if n_ctx <= self.S:
            return n_ctx - 1, self.win_ptr
        # Ainda enchendo a janela (caminho direto sem prefill): slot sequencial
        if len(self.order) < self.W:
            slot = self.S + len(self.order)
            self.order.append(slot)
            return slot, (self.win_ptr + 1) % self.W
        # Fase estacionária: vítima apenas no anel intermediário — os R slots
        # mais recentes (cauda de self.order) NUNCA são vítimas, e todo token
        # novo entra como mais novo na ordem (corrige o congelamento da zona protegida).
        ring_size = self.W - self.R
        ring = self.order[:ring_size]
        ring_scores = self.scores[ring]
        k = min(32, len(ring))
        idx_sorted = np.argpartition(ring_scores, k - 1)[:k]
        chosen = self.rng.choice(idx_sorted)
        victim = int(ring[int(chosen)])
        # Reutiliza o slot da vítima para o novo token, mas recencia-o no fim da ordem
        self.order.remove(victim)
        self.order.append(victim)
        self.n_reservoir += 1
        return victim, (self.win_ptr + 1) % self.W

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
