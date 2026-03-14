#!/usr/bin/env python3
"""
tests/test_tokenizer.py — Testes para BPETokenizer
"""
import pytest
from runtime.tokenizer import BPETokenizer, CORPUS

def test_tokenizer_train():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=100, verbose=False)
    assert tok._trained
    assert len(tok.vocab) > 0
    assert tok.bos_id in tok.vocab.values()
    assert tok.eos_id in tok.vocab.values()

def test_tokenizer_encode_decode():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=100, verbose=False)
    text = "atenção e memória"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert decoded.strip() == text  # Deve ser igual ou muito próximo

def test_tokenizer_save_load():
    import tempfile
    import os
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=100, verbose=False)
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        path = f.name
    try:
        tok.save(path)
        tok2 = BPETokenizer()
        tok2.load(path)
        assert tok.vocab == tok2.vocab
        assert tok.merges == tok2.merges
    finally:
        os.unlink(path)