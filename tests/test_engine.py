"""Testes do motor: cache O(1), validação de pesos e amostragem."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.engine import MultiLayerEngine
from runtime.inference import LittleHawkInference
from runtime.tokenizer import CORPUS, BPETokenizer


@pytest.fixture()
def engine():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=512, verbose=False)
    eng = MultiLayerEngine(d_model=128, n_heads=4, n_layers=2, sink_size=4, window_size=28, vocab_size=len(tok.vocab))
    return tok, eng


class TestStreamingCache:
    def test_cache_shape_never_grows(self, engine):
        _, eng = engine
        caches = eng.init_cache()
        win_ptr = 0
        first_id = id(caches[0][0])
        for n_ctx, tid in enumerate(range(200), start=1):  # bem além de max_cap = 32
            _, caches, win_ptr, _ = eng.step(tid % eng.V, caches, win_ptr, n_ctx)
        assert caches[0][0].shape == (1, eng.n_heads, eng.max_cap, eng.d_k)
        assert id(caches[0][0]) == first_id  # mesmos buffers, zero realocação

    def test_win_ptr_cycles(self, engine):
        _, eng = engine
        caches = eng.init_cache()
        win_ptr = 0
        for n_ctx in range(1, 200):
            _, caches, win_ptr, _ = eng.step(n_ctx % eng.V, caches, win_ptr, n_ctx)
        assert 0 <= win_ptr < eng.W

    def test_logits_finite(self, engine):
        _, eng = engine
        caches = eng.init_cache()
        win_ptr = 0
        logits, *_ = eng.step(3, caches, win_ptr, 1)
        assert logits.shape == (1, eng.V)
        assert np.isfinite(logits).all()


class TestWeightValidation:
    def test_rejects_missing_key(self, tmp_path, engine):
        # monta npz mínimo válido e remove uma chave obrigatória
        _, eng = engine
        path = str(tmp_path / "w.npz")
        pkg = {
            "_meta_d_model": np.float32(128),
            "_meta_n_heads": np.float32(4),
            "_meta_n_layers": np.float32(2),
            "_meta_vocab_size": np.float32(eng.V),
            "_meta_bos_id": np.float32(1),
            "_meta_eos_id": np.float32(2),
            "_meta_rope_base": np.float32(10000.0),
            "embed": eng.embed,
            "lm_head": eng.W_lm.T,
            "norm_w": eng.norm_w,
        }
        for i in range(2):
            L = eng.layers[i]
            names = ["W_q", "W_k", "W_v", "W_o", "rms_attn", "gate", "up", "down", "rms_ffn"]
            for n in names:
                arr = L.__dict__[n]
                arr = arr.T if arr.ndim == 2 else arr  # matrizes voltam à orientação [out,in]
                pkg[f"L{i}_{n}"] = np.ascontiguousarray(arr, dtype=np.float32)
        del pkg["L1_W_o"]
        np.savez(path, **pkg)
        with pytest.raises(ValueError, match="L1_W_o"):
            MultiLayerEngine(d_model=128, n_heads=4, n_layers=2).load_weights(path)

    def test_accepts_own_weights_roundtrip(self, tmp_path, engine):
        _, eng = engine
        path = str(tmp_path / "ok.npz")
        pkg = {
            "_meta_d_model": np.float32(128),
            "_meta_n_heads": np.float32(4),
            "_meta_n_layers": np.float32(2),
            "_meta_vocab_size": np.float32(eng.V),
            "_meta_bos_id": np.float32(1),
            "_meta_eos_id": np.float32(2),
            "_meta_rope_base": np.float32(10000.0),
            "embed": eng.embed,
            "lm_head": eng.W_lm.T,
            "norm_w": eng.norm_w,
        }
        for i in range(2):
            L = eng.layers[i]
            for n in ["W_q", "W_k", "W_v", "W_o", "rms_attn", "gate", "up", "down", "rms_ffn"]:
                arr = L.__dict__[n]
                arr = arr.T if arr.ndim == 2 else arr  # matrizes voltam à orientação [out,in]
                pkg[f"L{i}_{n}"] = np.ascontiguousarray(arr, dtype=np.float32)
        np.savez(path, **pkg)
        fresh = MultiLayerEngine(d_model=128, n_heads=4, n_layers=2, sink_size=4, window_size=28, vocab_size=eng.V)
        fresh.load_weights(path)  # não deve lançar


class TestSampling:
    def test_deterministic_with_seed(self, engine):
        from runtime.inference import Sampler, SamplingConfig

        _ = Sampler  # sampling agora vive em runtime.inference
        s = Sampler(SamplingConfig(temperature=0.8))
        rng_logits = np.random.default_rng(0).normal(size=engine[1].V)
        np.random.seed(123)
        a = [s.sample(rng_logits.copy()) for _ in range(20)]
        np.random.seed(123)
        b = [s.sample(rng_logits.copy()) for _ in range(20)]
        assert a == b

    def test_rep_penalty_discourages_repeat(self, engine):
        from runtime.inference import Sampler, SamplingConfig

        eng = engine[1]
        s_base = Sampler(SamplingConfig(rep_penalty=1.0))
        s_pen = Sampler(SamplingConfig(rep_penalty=2.0))
        logits = np.zeros(eng.V)
        logits[3] = 5.0
        base = sum(s_base.sample(logits.copy()) == 3 for _ in range(500)) / 500
        pen = sum(s_pen.sample(logits.copy(), generated=[3] * 10) == 3 for _ in range(500)) / 500
        assert pen < base * 0.5  # penalidade derruba a prob. ao menos à metade
