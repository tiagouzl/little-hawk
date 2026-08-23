"""Testes do tokenizer e do decode incremental."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from little_hawk import CORPUS, BPETokenizer, StreamDecoder


class TestBPETokenizer:
    def test_roundtrip(self):
        tok = BPETokenizer()
        tok.train(CORPUS, vocab_size=512, verbose=False)
        ids = tok.encode("memória e atenção", add_bos=True)
        assert ids[0] == tok.bos_id
        decoded = tok.decode(ids)
        assert "mem" in decoded and "ten" in decoded

    def test_unknown_chars_map_to_unk(self):
        tok = BPETokenizer()
        tok.train(CORPUS, vocab_size=256, verbose=False)
        unk = tok.vocab.get(tok.UNK)
        assert unk in tok.encode("zzzqqq ☃☃☃", add_bos=False)

    def test_load_rejects_empty_vocab(self, tmp_path):
        meta = tmp_path / "m_meta.json"
        meta.write_text('{"donor": "X/Y", "vocab": {}, "bos_id": 1, "eos_id": 2}', encoding="utf-8")
        with pytest.raises(RuntimeError, match="Vocabulário vazio"):
            BPETokenizer().load_donor_vocab(str(meta))


def _donor_tok(mapping):
    tok = BPETokenizer()
    tok._trained = True
    tok._donor_mode = True
    tok.id_to_token = {int(k): v for k, v in mapping.items()}
    return tok


class TestStreamDecoder:
    """’ = U+2019 = E2 80 99 → alfabeto GPT-2: â(chr 226) Ģ(chr 290) Ļ(chr 315)."""

    def test_multibyte_split_across_tokens(self):
        tok = _donor_tok({5: "â", 6: "Ģ", 7: "Ļ", 8: "memory"})
        sd = StreamDecoder(tok)
        assert sd.push(5) == ""  # byte parcial → retido
        assert sd.push(6) == ""  # ainda incompleto
        assert sd.push(7) == "\u2019"  # sequência completa
        assert sd.push(8) == "memory"
        assert sd.flush() == ""

    def test_ascii_passes_through(self):
        tok = _donor_tok({1: "Ġhello", 2: "Ġworld"})
        sd = StreamDecoder(tok)
        assert sd.push(1) + sd.push(2) == " hello world"

    def test_special_token_falls_back_to_literal(self):
        sd = StreamDecoder(_donor_tok({9: "<|im_end|>"}))
        assert sd.push(9) == "<|im_end|>"

    def test_demo_mode_normalizes_markers(self):
        tok = BPETokenizer()
        tok.train(CORPUS, vocab_size=512, verbose=False)
        tid = next(i for i, t in tok.id_to_token.items() if t.startswith("▁"))
        out = StreamDecoder(tok).push(tid)
        assert out and not out.startswith("▁")

    def test_unknown_id_returns_empty(self):
        assert StreamDecoder(_donor_tok({})).push(42) == ""
