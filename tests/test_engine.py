"""Testes do motor: cache O(1), validação de pesos e amostragem."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.engine import MultiLayerEngine
from engine.speculative import verify_chunk
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


class TestMinP:
    def test_min_p_filters_tail(self, engine):
        from runtime.inference import Sampler, SamplingConfig

        eng = engine[1]
        logits = np.full(eng.V, -10.0)
        logits[7] = 2.0  # p_max
        logits[8] = 1.5  # acima do corte min_p=0.3 (0.3·p_max≈0.11 → e^1.5/e^2≈0.6 passa)
        logits[9] = -1.0  # abaixo do corte (e^-1/e^2≈0.05 < 0.11)
        s = Sampler(SamplingConfig(min_p=0.3, top_k=0, top_p=1.0))
        picks = {s.sample(logits.copy()) for _ in range(400)}
        assert 9 not in picks and 8 in picks

    def test_min_p_zero_keeps_all(self, engine):
        from runtime.inference import Sampler, SamplingConfig

        eng = engine[1]
        logits = np.zeros(eng.V)
        s = Sampler(SamplingConfig(min_p=0.0, top_k=0, top_p=1.0))
        picks = {s.sample(logits.copy()) for _ in range(200)}
        assert len(picks) > 50  # distribuição uniforme intacta


class TestLMHead:
    def test_transposed_orientation_equivalent(self, engine):
        _, eng = engine
        x = np.random.default_rng(0).normal(size=(1, eng.d_model)).astype(np.float32)
        fast = (eng.W_lm_t @ x[0].reshape(-1, 1)).T  # caminho novo do step()
        ref = x @ eng.W_lm  # fórmula original
        np.testing.assert_allclose(fast, ref, rtol=1e-5, atol=1e-6)

    def test_wlm_t_contiguous(self, engine):
        import numpy as _np

        assert engine[1].W_lm_t.flags["C_CONTIGUOUS"]


class TestPositionFreeze:
    def test_stationary_phase_stable(self, engine):
        _, eng = engine
        caches = eng.init_cache()
        win_ptr = 0
        logits = None
        # roda 3× max_cap para exercitar enchimento + estacionária + múltiplos wraps
        for n_ctx in range(1, eng.max_cap * 3 + 1):
            logits, caches, win_ptr, _ = eng.step(n_ctx % eng.V, caches, win_ptr, n_ctx)
            assert np.isfinite(logits).all(), f"NaN/Inf em n_ctx={n_ctx}"
        assert 0 <= win_ptr < eng.W
        # evicções lógicas = n_ctx - max_cap
        assert eng.max_cap == eng.S + eng.W

    def test_rope_freeze_positions(self):
        # valida diretamente a semântica de position freeze do transformer
        from engine.jit_kernels import _rope_numpy

        d_k, n_heads = 8, 2
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, d_k, 2, dtype=np.float32) / d_k))
        x = np.random.default_rng(0).normal(size=(1, n_heads, 1, d_k)).astype(np.float32)
        # posição congelada vs crescente devem dar embeddings diferentes
        r_freeze = _rope_numpy(x, np.array([31], dtype=np.int64), inv_freq)
        r_growing = _rope_numpy(x, np.array([600], dtype=np.int64), inv_freq)
        assert not np.allclose(r_freeze, r_growing)
        # mesma posição congelada deve ser determinística
        r2 = _rope_numpy(x, np.array([31], dtype=np.int64), inv_freq)
        np.testing.assert_allclose(r_freeze, r2)


class TestPrefill:
    def test_prefill_matches_sequential(self, engine):
        _, eng = engine
        rng = np.random.default_rng(3)
        for T in (1, 5, eng.max_cap):
            toks = rng.integers(0, eng.V, size=T).tolist()
            caches_a = eng.init_cache()
            wp_a = 0
            lg_a = None
            for n, t in enumerate(toks, start=1):
                lg_a, caches_a, wp_a, _ = eng.step(t, caches_a, wp_a, n)
            lg_b, caches_b, wp_b, _ = eng.prefill(toks)
            np.testing.assert_allclose(lg_b[0], lg_a[0], atol=2e-3)
            assert wp_a == wp_b
            d_k = max(np.abs(caches_a[i][0] - caches_b[i][0]).max() for i in range(eng.n_layers))
            assert d_k < 2e-4

    def test_prefill_then_step_matches_all_sequential(self, engine):
        _, eng = engine
        rng = np.random.default_rng(4)
        toks = rng.integers(0, eng.V, size=10).tolist() + [7, 7, 7]
        # tudo sequencial
        caches_a = eng.init_cache()
        wp_a = 0
        lg_a = None
        for n, t in enumerate(toks, start=1):
            lg_a, caches_a, wp_a, _ = eng.step(t, caches_a, wp_a, n)
        # prefill do prompt + steps na geração
        caches_b = eng.init_cache()
        lg_b, caches_b, wp_b, _ = eng.prefill(toks[:10])
        n = 10
        for t in toks[10:]:
            n += 1
            lg_b, caches_b, wp_b, _ = eng.step(t, caches_b, wp_b, n)
        np.testing.assert_allclose(lg_b[0], lg_a[0], atol=2e-3)
        assert wp_a == wp_b

    def test_prefill_chunked_beyond_max_cap(self, engine):
        _, eng = engine
        rng = np.random.default_rng(5)
        T = eng.max_cap + 20  # excede janela — exercita branch chunked
        toks = rng.integers(0, eng.V, size=T).tolist()
        caches_a = eng.init_cache()
        wp_a = 0
        lg_a = None
        for n, t in enumerate(toks, start=1):
            lg_a, caches_a, wp_a, _ = eng.step(t, caches_a, wp_a, n)
        lg_b, caches_b, wp_b, _ = eng.prefill(toks)
        np.testing.assert_allclose(lg_b[0], lg_a[0], atol=2e-3)
        assert wp_a == wp_b
        d_k = max(np.abs(caches_a[i][0] - caches_b[i][0]).max() for i in range(eng.n_layers))
        assert d_k < 2e-4


class TestNexusSalience:
    def _policy(self):
        from engine.eviction import NexusSalienceEviction

        # W=64 → anel W-R=56 > k=32 (com anel < k, argpartition degeneraria em
        # escolha uniforme sobre tudo e a saliência não influiria — caso só
        # possível em configs de teste, não na produção W=508)
        p = NexusSalienceEviction(S=4, W=64, R=8, seed=7)
        # enche a janela
        for n in range(5, 5 + 64 + 1):
            p.next_slot(n)
        return p

    def test_salience_protects_high_surprise_slot(self):
        p = self._policy()
        ring = p.order[: p.W - p.R]
        # slot protegido: surpresa máxima; resto zerado
        protected = ring[0]
        p.set_salience(protected, 15.0)
        victims = {p.next_slot(100)[0] for _ in range(200)}
        assert protected not in victims, "slot com surpresa alta não deve ser vítima"

    def test_low_salience_still_evictable(self):
        p = self._policy()
        ring = p.order[: p.W - p.R]
        target = ring[0]
        p.set_salience(target, 0.0)
        victims = {p.next_slot(100)[0] for _ in range(200)}
        assert target in victims, "slot sem surpresa deve ser evictável"

    def test_reset_clears_salience(self):
        p = self._policy()
        p.set_salience(10, 12.0)
        p.reset()
        assert p.salience.sum() == 0 and p.order == []

    def test_engine_mode_wiring(self):
        from engine.engine import MultiLayerEngine
        import numpy as np

        eng = MultiLayerEngine(
            d_model=128, n_heads=4, n_layers=2, sink_size=4, window_size=28, vocab_size=512, eviction="nexus-salience"
        )
        caches = eng.init_cache()
        wp = 0
        lg, caches, wp, _ = eng.prefill(list(range(20)), caches)
        assert eng.eviction.salience[1:20].sum() > 0, "prefill deve popular saliência"
        for n in range(21, 60):
            logits, caches, wp, _ = eng.step(n % eng.V, caches, wp, n)
            assert np.isfinite(logits).all()
        assert len(eng.eviction.order) == 28 and len(set(eng.eviction.order)) == 28

    def test_score_reset_on_reuse(self):
        """Regressão do vazamento de EMA: vítima reaproveitada não herda score."""
        from engine.eviction import NexusEviction

        p = NexusEviction(S=4, W=64, R=8, seed=1)
        for n in range(5, 5 + 64 + 1):
            p.next_slot(n)
        ring = p.order[: p.W - p.R]
        # força score alto em todo anel para que qualquer vítima tenha fantasma
        for s in ring:
            p.scores[s] = 0.8
        victim, _ = p.next_slot(100)
        assert p.scores[victim] == 0.0, f"score vazou: {p.scores[victim]} != 0.0"

    def test_salience_score_reset_on_reuse(self):
        from engine.eviction import NexusSalienceEviction

        p = NexusSalienceEviction(S=4, W=64, R=8, seed=1)
        for n in range(5, 5 + 64 + 1):
            p.next_slot(n)
        ring = p.order[: p.W - p.R]
        for s in ring:
            p.scores[s] = 0.8
            p.salience[s] = 5.0
        victim, _ = p.next_slot(100)
        assert p.scores[victim] == 0.0
        assert p.salience[victim] == 0.0


class TestVerifyChunk:
    """Fase A speculative: verify_chunk ≡ steps sequenciais (obrigatório)."""

    def _setup(self, engine, n_prefill, n_steps):
        eng = engine
        caches = eng.init_cache()
        wp = 0
        rng = np.random.default_rng(11)
        ids = rng.integers(0, eng.V, size=n_prefill).tolist()
        lg, caches, wp, _ = eng.prefill(ids, caches)
        for i in range(n_steps):
            t = int(rng.integers(0, eng.V))
            lg, caches, wp, _ = eng.step(t, caches, wp, n_prefill + i + 1)
        return caches, wp, n_prefill + n_steps

    def _seq_vs_verify(self, eng, n_ctx, wp, caches_a, k):
        rng = np.random.default_rng(99)
        toks = [int(t) for t in rng.integers(0, eng.V, size=k)]
        # sequencial (referência)
        ca = [(k_.copy(), v.copy()) for k_, v in caches_a]
        seq_logits = []
        n = n_ctx
        wpa = wp
        for t in toks:
            n += 1
            lg, ca, wpa, _ = eng.step(t, ca, wpa, n)
            seq_logits.append(lg[0])
        # batched (candidatos)
        cb = [(k_.copy(), v.copy()) for k_, v in caches_a]
        lg_b, cb, wpb, sm = verify_chunk(eng, toks, cb, wp, n_ctx)
        return toks, np.array(seq_logits), lg_b, ca, cb, wpa, wpb

    def test_stationary_equivalence_with_wrap(self):
        eng = MultiLayerEngine(d_model=128, n_heads=4, n_layers=2, sink_size=4, window_size=28, vocab_size=512)
        # 32 prefill + 25 steps → win_ptr=25; chunk k=4 força wrap 25→2
        caches, wp, n_ctx = self._setup(eng, n_prefill=32, n_steps=25)
        assert wp == 25 and n_ctx > eng.max_cap
        toks, seq_lg, bat_lg, ca, cb, wpa, wpb = self._seq_vs_verify(eng, n_ctx, wp, caches, k=4)
        assert wpb == wpa == (25 + 4) % 28
        # Contrato de tolerância (medido, não arbitrário): com k=1 o caminho
        # batched já diverge 1.4e-2 do sequencial por REORDENAÇÃO de ops fp
        # (rope/matmul/softmax separados p/ sinks×janela). O que speculation
        # greedy consome é o ARGMAX — exigido exato em todas as posições.
        for j in range(len(toks)):
            assert seq_lg[j].argmax() == bat_lg[j].argmax(), f"top-1 divergiu em pos {j}"
        np.testing.assert_allclose(bat_lg, seq_lg, atol=5e-2, rtol=1e-2)
        d = max(np.abs(ca[i][0] - cb[i][0]).max() for i in range(eng.n_layers))
        assert d < 5e-2, f"cache divergiu: {d}"

    def test_fill_phase_equivalence(self):
        eng = MultiLayerEngine(d_model=128, n_heads=4, n_layers=2, sink_size=4, window_size=28, vocab_size=512)
        caches, wp, n_ctx = self._setup(eng, n_prefill=10, n_steps=0)
        toks, seq_lg, bat_lg, ca, cb, wpa, wpb = self._seq_vs_verify(eng, n_ctx, wp, caches, k=5)
        assert wpb == wpa
        np.testing.assert_allclose(bat_lg, seq_lg, atol=2e-3, rtol=1e-3)
        d = max(np.abs(ca[i][0] - cb[i][0]).max() for i in range(eng.n_layers))
        assert d < 2e-4

    def test_boundary_crossing_raises(self):
        from engine.speculative import verify_chunk

        eng = MultiLayerEngine(d_model=128, n_heads=4, n_layers=2, sink_size=4, window_size=28, vocab_size=512)
        caches, wp, n_ctx = self._setup(eng, n_prefill=30, n_steps=0)  # n_ctx=30, cap=32
        with pytest.raises(ValueError, match="indisponível"):
            verify_chunk(
                eng,
                [1, 2, 3, 4],
                [(c.copy()) for c, _ in []] or [(np.zeros((1, 4, 32, 32), np.float32),) * 2],
                wp,
                n_ctx,
            )

    def test_eviction_falls_back(self):
        from engine.speculative import can_verify

        eng = MultiLayerEngine(
            d_model=128, n_heads=4, n_layers=2, sink_size=4, window_size=28, vocab_size=512, eviction="nexus"
        )
        assert can_verify(eng, n_ctx=100, k=4) is False


class TestSpeculativePhaseB:
    def _hawk(self):
        tok = BPETokenizer()
        tok.train(CORPUS, vocab_size=512, verbose=False)
        eng = MultiLayerEngine(
            d_model=128, n_heads=4, n_layers=2, sink_size=4, window_size=28, vocab_size=len(tok.vocab)
        )
        return LittleHawkInference(tok, eng), tok

    def test_speculative_runs_and_reports_stats(self):
        from runtime.inference import SamplingConfig

        hawk, _ = self._hawk()
        cfg = SamplingConfig(max_tokens=30, temperature=0.7)
        # regime H1: prompt repetitivo garante hits de n-gram
        prompt = ("memória e atenção fluem pelo cache. " * 12).strip()
        out, st = hawk._generate_speculative(prompt, cfg, None, k=4)
        assert len(out) > 0 and st.emitted == 30
        assert st.rounds > 0, "prompt repetitivo deve gerar hits de n-gram"
        # Caso B do contrato E2 é válido aqui: no brinquedo, proposer (segmentação
        # do prompt) e trajetória greedy podem divergir por fronteira de tokens —
        # aceitação 0 é resultado legítimo; o veredito H1 é no bench com pesos reais.
        assert st.accepted <= st.proposed

    def test_speculative_fallback_on_sparse_prompt(self):
        from runtime.inference import SamplingConfig

        hawk, _ = self._hawk()
        cfg = SamplingConfig(max_tokens=10, temperature=0.7)
        out, st = hawk._generate_speculative("zzq", cfg, None, k=4)
        assert st.emitted == 10 and st.rounds == 0  # sem hits → puro fallback

    def test_speculative_deterministic(self):
        from runtime.inference import SamplingConfig

        hawk, _ = self._hawk()
        cfg = SamplingConfig(max_tokens=20, temperature=0.7)
        np.random.seed(7)
        a, sa = hawk._generate_speculative(("tokens fluem pelo cache. " * 10).strip(), cfg, None, k=4)
        np.random.seed(7)
        b, sb = hawk._generate_speculative(("tokens fluem pelo cache. " * 10).strip(), cfg, None, k=4)
        assert a == b
        da, db = sa.as_dict(), sb.as_dict()
        da.pop("wall_s")
        db.pop("wall_s")  # timing não é determinístico
        assert da == db

    def test_k_zero_uses_normal_path(self):
        from runtime.inference import SamplingConfig

        hawk, _ = self._hawk()
        cfg = SamplingConfig(max_tokens=10)
        out = hawk.generate("memória", sampling_config=cfg, panel=False, speculative_k=0)
        assert isinstance(out, str) and len(out) > 0
