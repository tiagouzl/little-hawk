"""Rwkv7Engine — inferência RWKV-7 (x070, g1) em NumPy puro, sem torch.

Fórmulas: referência oficial RWKV-v7/run_rwkv7_qwen35.py (classe RWKV7) e
RWKV-v7/rwkv_v7_numpy.py (johanwind), ambos Apache-2.0, verificados pelo
autor contra o pacote `rwkv`.

Interface espelha MultiLayerEngine no mínimo necessário ao harness:
init_cache / prefill / step / load_weights. Sem KV cache: estado recorrente
de tamanho constante (~2.4 MB p/ L12-D768). S=W=max_cap=0 sinaliza "sem
janela"; eviction=None; sem verify_chunk (speculative desabilitado).
"""
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lerp(a, b, t):
    return a + (b - a) * t


def _layer_norm(x, w, b, eps=1e-5):
    return (x - x.mean()) / np.sqrt(x.var() + eps) * w + b


def _group_norm(y, w, b, eps=64e-5):
    # y: (H, N)
    yn = (y - y.mean(axis=1, keepdims=True)) / np.sqrt(y.var(axis=1, keepdims=True) + eps)
    return (yn.reshape(-1) * w + b)


def _l2(x):
    return x * np.maximum(np.sqrt(np.sum(x * x, axis=1, keepdims=True)), 1e-12) ** -1.0


def _dplr(S, r, w, k, v, kk, b):
    # S: (H,N,N); r,w,k,v,kk,b: (H,N). w decai ao longo de k (última dim).
    S = (S * w[:, None, :]
         + np.einsum("hva,ha,hb->hvb", S, kk, b)
         + np.einsum("hv,hk->hvk", v, k))
    return np.einsum("hvk,hk->hv", S, r), S


class Rwkv7Engine:
    S = 0
    W = 0
    max_cap = 0
    eviction = None
    eviction_name = "none"

    def __init__(self, n_layer=12, n_embd=768, vocab_size=65536):
        self.n_layers = n_layer
        self.n_embd = n_embd
        self.V = vocab_size
        self.H = n_embd // 64
        self.N = 64
        # RWKV World não tem BOS/EOS — sentinelas fora do vocab: parada só por max_tokens
        self.bos_id = -1
        self.eos_id = -2
        self.W = {}

    # -- pesos ---------------------------------------------------------
    def load_weights(self, path):
        data = np.load(path, allow_pickle=False)
        W = {k: np.squeeze(np.asarray(data[k], dtype=np.float32)) for k in data.files}
        need = ["emb.weight", "ln_out.weight", "ln_out.bias", "head.weight",
                "blocks.0.ln0.weight", "blocks.0.ln0.bias"]
        missing = [k for k in need if k not in W]
        if missing:
            raise ValueError(f"pesos RWKV inválidos, ausentes: {missing}")
        n_layer = 1 + max(int(k.split(".")[1]) for k in W if k.startswith("blocks."))
        C = W["emb.weight"].shape[1]
        self.n_layers = n_layer
        self.n_embd = C
        self.H = C // 64
        self.V = W["emb.weight"].shape[0]
        # Convenção x@W: só os nn.Linear do torch vêm em (out,in) e precisam
        # de transpose. Pares LoRA (w1/w2/a1/a2/v1/v2/g1/g2) e vetores x_*
        # já estão em (in,out)/(C,) — transpose blanket os quebra.
        _T = ("att.receptance.weight", "att.key.weight", "att.value.weight",
              "att.output.weight", "ffn.key.weight", "ffn.value.weight")
        T = {}
        for k, v in W.items():
            if k == "emb.weight":
                T[k] = _layer_norm(v, W["blocks.0.ln0.weight"], W["blocks.0.ln0.bias"])
            elif v.ndim == 2 and any(k.endswith(s) for s in _T):
                T[k] = v.T.copy()
            else:
                T[k] = v
        T["head_T"] = W["head.weight"].T.copy()
        self.W = T

    # -- estado ----------------------------------------------------------
    def init_cache(self):
        C, H, N = self.n_embd, self.H, self.N
        st = []
        for _ in range(self.n_layers):
            st.append([{"x": np.zeros(C, np.float32),
                        "rnn": np.zeros((H, N, N), np.float32)},
                       {"x": np.zeros(C, np.float32)}])
        return st

    def state_bytes(self):
        C, H, N = self.n_embd, self.H, self.N
        return self.n_layers * (2 * C * 4 + H * N * N * 4)

    # -- forward de 1 token ----------------------------------------------
    def _tmix(self, i, x0, v_first, s):
        W, H, N = self.W, self.H, self.N
        p = f"blocks.{i}.att."
        x = _layer_norm(x0, W[f"blocks.{i}.ln1.weight"], W[f"blocks.{i}.ln1.bias"])
        prev = s["x"].copy()
        s["x"] = x.copy()
        xr = _lerp(x, prev, W[p + "x_r"])
        xw = _lerp(x, prev, W[p + "x_w"])
        xk = _lerp(x, prev, W[p + "x_k"])
        xv = _lerp(x, prev, W[p + "x_v"])
        xa = _lerp(x, prev, W[p + "x_a"])
        xg = _lerp(x, prev, W[p + "x_g"])
        r = xr @ W[p + "receptance.weight"]
        k = xk @ W[p + "key.weight"]
        v = xv @ W[p + "value.weight"]
        if v_first is None:
            v_first = v.copy()
        else:
            v = _lerp(v, v_first, _sigmoid(W[p + "v0"] + (xv @ W[p + "v1"]) @ W[p + "v2"]))
        w = np.exp(-_sigmoid(W[p + "w0"] + np.tanh(xw @ W[p + "w1"]) @ W[p + "w2"]) / np.e ** 0.5)
        a = _sigmoid(W[p + "a0"] + (xa @ W[p + "a1"]) @ W[p + "a2"])
        kk = _l2((k * W[p + "k_k"]).reshape(H, N)).reshape(-1)
        k = _lerp(k, k * a, W[p + "k_a"])
        r, w, k, v, kk, a = (z.reshape(H, N) for z in (r, w, k, v, kk, a))
        y, s["rnn"] = _dplr(s["rnn"], r, w, k, v, kk, -kk * a)
        y = _group_norm(y, W[p + "ln_x.weight"], W[p + "ln_x.bias"])
        y = y + ((np.sum(r * k * W[p + "r_k"], axis=1, keepdims=True)) * v).reshape(-1)
        g = _sigmoid(xg @ W[p + "g1"]) @ W[p + "g2"]
        return x0 + (y * g) @ W[p + "output.weight"], v_first

    def _cmix(self, i, x0, s):
        W = self.W
        p = f"blocks.{i}.ffn."
        x = _layer_norm(x0, W[f"blocks.{i}.ln2.weight"], W[f"blocks.{i}.ln2.bias"])
        prev = s["x"].copy()
        s["x"] = x.copy()
        x = _lerp(x, prev, W[p + "x_k"])
        h = x @ W[p + "key.weight"]
        return x0 + (np.maximum(h, 0) ** 2) @ W[p + "value.weight"]

    def _forward_one(self, token, state):
        W = self.W
        x = W["emb.weight"][int(token)].copy()
        v_first = None
        for i in range(self.n_layers):
            x, v_first = self._tmix(i, x, v_first, state[i][0])
            x = self._cmix(i, x, state[i][1])
        x = _layer_norm(x, W["ln_out.weight"], W["ln_out.bias"])
        return x @ W["head_T"], state

    # -- interface harness -------------------------------------------------
    def prefill(self, tokens, caches=None):
        state = self.init_cache() if caches is None else caches
        logits = None
        for t in tokens:
            logits, state = self._forward_one(t, state)
        return logits.reshape(1, -1), state, 0, 0.0

    def step(self, token_id, caches, win_ptr, n_ctx):
        logits, state = self._forward_one(token_id, caches)
        return logits.reshape(1, -1), state, 0, 0.0
