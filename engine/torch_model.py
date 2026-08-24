"""
engine/torch_model.py — Torch mirror de Little Hawk para ONNX export

Carrega pesos .npz e espelha `engine/transformer.py` em torch.nn para
`torch.onnx.export`. Suporta cache dinâmico (past_key_values) para bench.

Uso:
  from engine.torch_model import LittleHawkTorch
  model = LittleHawkTorch("little_hawk_weights.npz")
  torch.onnx.export(model, (x, past), "model.onnx")
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LlamaLayerTorch(nn.Module):
    def __init__(self, W_q, W_k, W_v, W_o, rms_attn, gate, up, down, rms_ffn, n_heads, d_k, b_q=None, b_k=None, b_v=None, inv_freq=None):
        super().__init__()
        self.n_heads, self.d_k = n_heads, d_k
        # Mirrors engine/transformer.py: W_q stored as [d,d].T -> [d,d] (square), gate/up as [d,inter], down as [inter,d]
        self.W_q = nn.Parameter(torch.from_numpy(W_q.T.copy()))
        self.W_k = nn.Parameter(torch.from_numpy(W_k.T.copy()))
        self.W_v = nn.Parameter(torch.from_numpy(W_v.T.copy()))
        self.W_o = nn.Parameter(torch.from_numpy(W_o.T.copy()))
        self.rms_attn = nn.Parameter(torch.from_numpy(rms_attn.copy()))
        self.rms_ffn = nn.Parameter(torch.from_numpy(rms_ffn.copy()))
        self.gate = nn.Parameter(torch.from_numpy(gate.T.copy()))  # [d,inter]
        self.up = nn.Parameter(torch.from_numpy(up.T.copy()))  # [d,inter]
        self.down = nn.Parameter(torch.from_numpy(down.T.copy()))  # [inter,d]
        self.b_q = nn.Parameter(torch.from_numpy(b_q.copy())) if b_q is not None else None
        self.b_k = nn.Parameter(torch.from_numpy(b_k.copy())) if b_k is not None else None
        self.b_v = nn.Parameter(torch.from_numpy(b_v.copy())) if b_v is not None else None
        self.inv_freq = torch.from_numpy(inv_freq) if inv_freq is not None else None

    def rms_norm(self, x, w):
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + 1e-6) * w

    def rope(self, x, pos):
        # x: [B, H, L, d_k], pos: [L]
        ang = torch.outer(pos.float(), self.inv_freq)  # [L, d_k/2]
        emb = torch.cat([ang, ang], dim=-1)[None, None, :, :]  # [1,1,L,d_k]
        s, c = emb.sin(), emb.cos()
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return x * c + torch.cat([-x2, x1], dim=-1) * s

    def forward(self, x, k_cache, v_cache):
        # x: [B,1,d], k/v_cache: [B,H,S,d_k] — pos computado internamente para export
        B = x.shape[0]
        S = k_cache.shape[2]
        pos_q = torch.tensor([S], device=x.device)
        pos_cache = torch.arange(S, device=x.device)
        x_n = self.rms_norm(x, self.rms_attn)
        q = (x_n @ self.W_q).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        k = (x_n @ self.W_k).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        v = (x_n @ self.W_v).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        if self.b_q is not None:
            q = q + self.b_q.view(1, self.n_heads, 1, self.d_k)
            k = k + self.b_k.view(1, self.n_heads, 1, self.d_k)
            v = v + self.b_v.view(1, self.n_heads, 1, self.d_k)
        q = self.rope(q, pos_q)
        k_cache_r = self.rope(k_cache, pos_cache)
        k = self.rope(k, pos_q)
        k_cache = torch.cat([k_cache_r, k], dim=2)
        v_cache = torch.cat([v_cache, v], dim=2)
        sc = (q @ k_cache.transpose(-2, -1)) / (self.d_k ** 0.5)
        at = F.softmax(sc, dim=-1)
        out = (at @ v_cache).transpose(1, 2).reshape(B, 1, -1) @ self.W_o
        x = x + out
        x_n2 = self.rms_norm(x, self.rms_ffn)
        g = x_n2 @ self.gate
        u = x_n2 @ self.up
        h = (g / (1 + torch.exp(-g))) * u
        x = x + h @ self.down
        return x, k_cache, v_cache


class LittleHawkTorch(nn.Module):
    def __init__(self, npz_path):
        super().__init__()
        import numpy as np

        data = np.load(npz_path, allow_pickle=False)
        meta_path = str(npz_path).replace(".npz", "_meta.json")
        meta = json.loads(Path(meta_path).read_text())
        d_model = int(data["_meta_d_model"])
        n_heads = int(data["_meta_n_heads"])
        n_layers = int(data["_meta_n_layers"])
        d_k = d_model // n_heads
        rope_base = float(data["_meta_rope_base"])
        inv_freq = 1.0 / (rope_base ** (np.arange(0, d_k, 2, dtype=np.float32) / d_k))
        self.n_layers, self.n_heads, self.d_k, self.d_model = n_layers, n_heads, d_k, d_model
        self.vocab_size = int(data["_meta_vocab_size"])
        # embed + norm + lm_head
        embed = torch.from_numpy(data["embed"].astype(np.float32))
        self.embed = nn.Embedding.from_pretrained(embed, freeze=True)
        self.norm_w = nn.Parameter(torch.from_numpy(data["norm_w"].astype(np.float32)))
        lm_head = torch.from_numpy(data["lm_head"].astype(np.float32))  # [V,d]
        self.lm_head = nn.Parameter(lm_head)
        # layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                LlamaLayerTorch(
                    W_q=data[f"L{i}_W_q"].astype(np.float32),
                    W_k=data[f"L{i}_W_k"].astype(np.float32),
                    W_v=data[f"L{i}_W_v"].astype(np.float32),
                    W_o=data[f"L{i}_W_o"].astype(np.float32),
                    rms_attn=data[f"L{i}_rms_attn"].astype(np.float32),
                    gate=data[f"L{i}_gate"].astype(np.float32),
                    up=data[f"L{i}_up"].astype(np.float32),
                    down=data[f"L{i}_down"].astype(np.float32),
                    rms_ffn=data[f"L{i}_rms_ffn"].astype(np.float32),
                    n_heads=n_heads, d_k=d_k,
                    b_q=data[f"L{i}_b_q"].astype(np.float32) if f"L{i}_b_q" in data else None,
                    b_k=data[f"L{i}_b_k"].astype(np.float32) if f"L{i}_b_k" in data else None,
                    b_v=data[f"L{i}_b_v"].astype(np.float32) if f"L{i}_b_v" in data else None,
                    inv_freq=inv_freq,
                )
            )

    def forward(self, input_ids, k_caches, v_caches, pos_q, pos_cache):
        # input_ids: [B,1], k/v_caches: [n_layers, B, H, S, d_k] como lista de tensores? Para ONNX, flatten
        # Para simplificar export 1 layer, usamos 1 cache
        x = self.embed(input_ids)  # [B,1,d]
        new_k, new_v = [], []
        for i, layer in enumerate(self.layers):
            x, k_out, v_out = layer(x, k_caches[i], v_caches[i], pos_q, pos_cache)
            new_k.append(k_out)
            new_v.append(v_out)
        # final norm + lm_head
        var = x.pow(2).mean(-1, keepdim=True)
        xn = x * torch.rsqrt(var + 1e-6) * self.norm_w
        logits = xn @ self.lm_head.T  # [B,1,V]
        return logits, new_k, new_v
