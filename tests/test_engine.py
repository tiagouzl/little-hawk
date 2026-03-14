#!/usr/bin/env python3
"""
tests/test_engine.py — Testes para MultiLayerEngine
"""
import numpy as np
from engine.engine import MultiLayerEngine

def test_engine_init():
    engine = MultiLayerEngine(d_model=64, n_heads=2, n_layers=1, sink_size=2, window_size=4, vocab_size=100)
    assert engine.d_model == 64
    assert engine.n_heads == 2
    assert engine.n_layers == 1
    assert engine.max_cap == 6  # sink + window

def test_engine_step():
    engine = MultiLayerEngine(d_model=64, n_heads=2, n_layers=1, sink_size=2, window_size=4, vocab_size=100)
    caches = engine.init_cache()
    win_ptr = 0
    n_ctx = 1
    logits, new_caches, new_win_ptr, sm = engine.step(1, caches, win_ptr, n_ctx)
    assert logits.shape == (1, 100)  # vocab_size
    assert len(new_caches) == 1  # n_layers
    assert isinstance(sm, float)