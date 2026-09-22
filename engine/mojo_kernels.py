"""
engine/mojo_kernels.py — Kernel do forward por camada do prefill, isolado
para permitir port para Mojo (mesmo padrão de engine/jit_kernels.py:
HAS_MOJO + LITTLE_HAWK_MOJO_PREFILL=1, fallback NumPy com semântica idêntica).

Extraído literalmente do corpo do loop `for li,layer in enumerate(self.layers)`
em MultiLayerEngine.prefill() (engine/engine.py) — mesma matemática, mesma
ordem de operações. Isolar aqui não muda comportamento nenhum enquanto
HAS_MOJO=False; é só o ponto de injeção para o experimento.

Escopo do kernel (o que entra no "forward GEMM do prefill"):
  1. RMSNorm + projeção QKV (x_n @ W_q/W_k/W_v)      -> maior GEMM por camada
  2. RoPE + atenção causal batelada (qr@kr.T, at@v)
  3. RMSNorm + FFN SwiGLU (gate/up/down)

Critério de aprovação (já definido): TTFT com este kernel em Mojo precisa
ser >= 1.5x mais rápido que o baseline NumPy batelado
(MultiLayerEngine.prefill), que é o caminho rápido real. O ONNX é só
referência informativa — OnnxEngine.prefill() é sequencial e já é ~7x
mais lento que o NumPy no prefill, então usá-lo como baseline tornaria
o critério vácuo. Ver scripts/bench_prefill_ttft.py.
"""
import math
import os

import numpy as np

from .jit_kernels import _rope_numpy as _rope

HAS_MOJO = False
if os.getenv("LITTLE_HAWK_MOJO_PREFILL") == "1":
    try:
        # Placeholder: nome do binding real ainda não existe.
        # Import esperado, ex.: `from little_hawk_mojo import prefill_layer_forward`
        import little_hawk_mojo

        HAS_MOJO = True
    except ImportError:
        pass


def _prefill_layer_forward_numpy(x, layer, pos, inv_freq, causal, T, n_heads, d_k, d_model):
    """
    Fallback NumPy — cópia exata do corpo do loop em MultiLayerEngine.prefill().
    kc/vc não são escritos aqui (isso é responsabilidade do chamador, que
    ainda mantém o cache); esta função é pura em x e retorna só o x atualizado
    mais os k/v da camada para quem for escrever o cache.
    Pré-condição: caches frescos (zeros). O original aplica RoPE no slice do
    cache pós-escrita (kc[:,:,:T,:]); aqui aplica no k local — idêntico com
    cache fresco, mas não com cache reutilizado.
    """
    x_n = layer._rms_norm(x, layer.rms_attn)
    _q = x_n @ layer.W_q
    _k = x_n @ layer.W_k
    _v = x_n @ layer.W_v
    if layer.b_q is not None:
        _q = _q + layer.b_q
        _k = _k + layer.b_k
        _v = _v + layer.b_v
    q = _q.reshape(1, T, n_heads, d_k).transpose(0, 2, 1, 3)
    k = _k.reshape(1, T, n_heads, d_k).transpose(0, 2, 1, 3)
    v = _v.reshape(1, T, n_heads, d_k).transpose(0, 2, 1, 3)

    qr = _rope(q, pos, inv_freq)
    kr = _rope(k, pos, inv_freq)
    sc = (qr @ kr.transpose(0, 1, 3, 2)) / math.sqrt(d_k)
    sc = np.where(causal, sc, np.float32(-np.inf))
    sc = sc - sc.max(axis=-1, keepdims=True)
    at = np.exp(sc)
    at /= at.sum(axis=-1, keepdims=True)
    out = (at @ v).transpose(0, 2, 1, 3).reshape(1, T, d_model) @ layer.W_o

    x = x + out
    x = x + layer.ffn(x)
    sm0 = float(at[:, :, 0, :].mean() * 100)
    return x, k[0], v[0], sm0


def _prefill_layer_forward_mojo(x, layer, pos, inv_freq, causal, T, n_heads, d_k, d_model):
    """
    Port Mojo do mesmo forward. Ainda não implementado — levanta
    NotImplementedError de propósito em vez de degradar silenciosamente
    para NumPy, porque um benchmark que cai pro fallback sem avisar é
    exatamente a classe de bug que o projeto já corrigiu uma vez
    (ONNX+nexus-salience caindo pra FIFO sem avisar, commit da revisão externa).
    """
    raise NotImplementedError(
        "Kernel Mojo do prefill ainda não implementado. "
        "Implemente aqui chamando o binding little_hawk_mojo e remova este raise, "
        "ou rode o benchmark sem LITTLE_HAWK_MOJO_PREFILL=1 para usar o NumPy puro."
    )


def prefill_layer_forward(x, layer, pos, inv_freq, causal, T, n_heads, d_k, d_model):
    """Ponto único de dispatch — usado por engine.py no lugar do bloco inline."""
    if HAS_MOJO:
        return _prefill_layer_forward_mojo(x, layer, pos, inv_freq, causal, T, n_heads, d_k, d_model)
    return _prefill_layer_forward_numpy(x, layer, pos, inv_freq, causal, T, n_heads, d_k, d_model)
