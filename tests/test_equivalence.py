"""Equivalência numérica: Little Hawk vs HF Transformers (SmolLM-135M).

Requer: pip install torch --index-url https://download.pytorch.org/whl/cpu
        pip install transformers
        pesos transplantados (little_hawk_weights.npz + _meta.json).
Sem torch/transformers instalados, os testes são pulados.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch", reason="torch não instalado")
transformers = pytest.importorskip("transformers", reason="transformers não instalado")

from engine.engine import MultiLayerEngine
from runtime.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent.parent
NPZ = ROOT / "little_hawk_weights.npz"
META = ROOT / "little_hawk_weights_meta.json"
MODEL_ID = "HuggingFaceTB/SmolLM-135M"
PROMPTS = ["The quick brown fox", "attention and memory are"]


def _forward_hawk(tok, eng, ids):
    caches = eng.init_cache()
    wp = 0
    logits = None
    for n_ctx, tid in enumerate(ids, start=1):
        out, caches, wp, _ = eng.step(tid, caches, wp, n_ctx)
        logits = out[0]
    return logits


@pytest.fixture(scope="module")
def hawk():
    if not (NPZ.exists() and META.exists()):
        pytest.skip("pesos transplantados ausentes — rode little_hawk_transplant.py")
    tok = BPETokenizer()
    tok.load_donor_vocab(str(META))
    m = json.loads(META.read_text(encoding="utf-8"))
    eng = MultiLayerEngine(
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        sink_size=4,
        window_size=508,
        vocab_size=m["vocab_size"],
    )
    eng.load_weights(str(NPZ))
    return tok, eng


@pytest.fixture(scope="module")
def reference():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rtok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    return rtok, model


@pytest.mark.parametrize("text", PROMPTS)
def test_logits_match_reference(hawk, reference, text):
    htok, eng = hawk
    rtok, model = reference

    ids = rtok.encode(text)  # donor mode: sem BOS explícito
    assert ids == htok.encode(text)  # tokenizers idênticos

    ours = _forward_hawk(htok, eng, ids)
    with torch.no_grad():
        ref = model(torch.tensor([ids])).logits[0, -1].numpy()

    rel = np.abs(ours - ref) / np.maximum(np.abs(ref), 1e-6)
    absdiff = np.abs(ours - ref)
    # fp32 com ordem de somatório diferente; erro relativo explode perto de
    # logits ≈ 0, então usamos mediana relativa + máximo absoluto
    assert np.median(rel) < 2e-3, f"mediana de divergência relativa: {np.median(rel):.2e}"
    assert np.max(absdiff) < 1e-1, f"divergência máxima absoluta: {np.max(absdiff):.2e}"


def test_top5_overlap(hawk, reference):
    htok, eng = hawk
    rtok, model = reference
    text = PROMPTS[0]
    ids = rtok.encode(text)
    ours = _forward_hawk(htok, eng, ids)
    with torch.no_grad():
        ref = model(torch.tensor([ids])).logits[0, -1].numpy()
    top_ours = set(np.argsort(ours)[::-1][:5])
    top_ref = set(np.argsort(ref)[::-1][:5])
    assert len(top_ours & top_ref) >= 3, f"top-5 divergente: ours={top_ours} ref={top_ref}"
