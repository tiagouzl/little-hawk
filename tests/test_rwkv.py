"""Testes RWKV-7 (§5-6): tokenizer, sanidade, equivalência numérica.

Equivalência vs pacote `rwkv` oficial (cpu fp32). Tolerâncias explícitas:
op-ordem NumPy-vs-torch difere em fp e a recorrência amplifica; o que o
benchmark consome (greedy/top-k) é coberto por top1 + top5.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.rwkv_engine import Rwkv7Engine
from engine.rwkv_tokenizer import RwkvTokenizer

WEIGHTS = "data/rwkv/rwkv7_g1d_01b_fp32.npz"
VOCAB = "data/rwkv/rwkv_vocab_v20230424.txt"
PTH = "data/rwkv/rwkv7-g1d-0.1b-20260129-ctx8192"

needs_files = pytest.mark.skipif(
    not (os.path.exists(WEIGHTS) and os.path.exists(VOCAB)),
    reason="pesos/vocab RWKV ausentes",
)
needs_ref = pytest.mark.skipif(
    not os.path.exists(PTH + ".pth"),
    reason="checkpoint .pth de referência ausente",
)


@pytest.fixture(scope="module")
def eng():
    e = Rwkv7Engine()
    e.load_weights(WEIGHTS)
    return e


@pytest.fixture(scope="module")
def tok():
    return RwkvTokenizer(VOCAB)


@needs_files
def test_tokenizer_byte_equal_reference(tok):
    from rwkv.utils import PIPELINE

    ref = PIPELINE(None, "rwkv_vocab_v20230424")
    for s in [
        "Olá, tudo bem?",
        "The Eiffel tower is in the city of",
        "Quanto é 7 + 8?",
        "HAWK-7319",
        "café pela manhã ☃",
    ]:
        assert tok.encode(s) == ref.encode(s), s
        assert tok.decode(tok.encode(s)) == s, s


@needs_files
def test_state_shape_const(eng):
    s0 = eng.init_cache()
    assert len(s0) == eng.n_layers
    assert s0[0][0]["rnn"].shape == (eng.H, eng.N, eng.N)
    assert eng.state_bytes() == eng.n_layers * (2 * eng.n_embd * 4 + eng.H * eng.N * eng.N * 4)


@needs_files
def test_sanity_no_nan_deterministic(eng, tok):
    ids = tok.encode("Teste de sanidade.")
    st = eng.init_cache()
    l1, st, _, _ = eng.prefill(ids, st)
    assert np.isfinite(l1).all()
    assert any(np.abs(s[0]["rnn"]).sum() > 0 for s in st)
    sa, sb = eng.init_cache(), eng.init_cache()
    la, _ = eng._forward_one(ids[0], sa)
    lb, _ = eng._forward_one(ids[0], sb)
    assert np.array_equal(la, lb)


@needs_files
@needs_ref
def test_equivalence_vs_official(eng, tok):
    os.environ["RWKV_V7_ON"] = "1"
    from rwkv.model import RWKV as RefRWKV

    ref = RefRWKV(model=PTH, strategy="cpu fp32")
    ids = tok.encode("Meu nome é Tiago e eu moro em Caicó.")
    st = eng.init_cache()
    mine, _, _, _ = eng.prefill(ids, st)
    out, _ = ref.forward(ids, None)
    rl = out.detach().numpy()
    rl = rl[-1] if rl.ndim == 2 else rl
    mine = mine[0]
    rel = np.abs(mine - rl) / (np.abs(rl) + 1.0)
    assert float(np.median(rel)) < 0.02, float(np.median(rel))
    t5m = set(np.argpartition(mine, -5)[-5:])
    t5r = set(np.argpartition(rl, -5)[-5:])
    assert len(t5m & t5r) >= 4
    assert int(np.argmax(mine)) == int(np.argmax(rl))
