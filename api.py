#!/usr/bin/env python3
"""API FastAPI simples para o Little Hawk.

- /health         → status JSON
- /generate (POST)→ text/event-stream (SSE) com tokens gerados
  body: {"prompt": "...", "max_tokens": 80, "temperature": 0.7, "top_k": 40, "top_p": 0.92, "rep_penalty": 1.15}

Reutiliza o motor definido em little_hawk_cli.py. Se little_hawk_weights.npz
não existir, cai no modo demo (pesos aleatórios).
"""
import os
import asyncio
from typing import AsyncGenerator

from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from little_hawk_cli import (
    BPETokenizer,
    MultiLayerEngine,
    LittleHawkInference,
    CORPUS,
)

app = FastAPI(title="Little Hawk API", version="0.1.0")

# Modelo carregado uma vez por processo
_tokenizer = None
_engine = None
_hawk = None

DEFAULT_WEIGHTS = os.getenv("LITTLE_HAWK_WEIGHTS", "little_hawk_weights.npz")


def load_model(weights_path: str | None = None):
    """Carrega tokenizer/engine. Fallback para modo demo se pesos ausentes."""
    global _tokenizer, _engine, _hawk
    if _hawk is not None:
        return
    tok = BPETokenizer()
    if weights_path and os.path.exists(weights_path):
        meta = weights_path.replace(".npz", "_meta.json")
        if not os.path.exists(meta):
            raise FileNotFoundError(f"Meta não encontrado: {meta}")
        tok.load_donor_vocab(meta)
        with open(meta, encoding="utf-8") as f:
            m = __import__("json").load(f)
        _dm = int(m.get("d_model", 576)); _nh = int(m.get("n_heads", 9))
        _nl = int(m.get("n_layers", 30)); _vs = int(m.get("vocab_size", len(tok.vocab)))
        eng = MultiLayerEngine(d_model=_dm, n_heads=_nh, n_layers=_nl,
                               sink_size=4, window_size=508, vocab_size=_vs)
        eng.load_weights(weights_path)
    else:
        tok.train(CORPUS, vocab_size=512, verbose=False)
        eng = MultiLayerEngine(d_model=128, n_heads=4, n_layers=2,
                               sink_size=4, window_size=28, vocab_size=len(tok.vocab))
    _tokenizer, _engine = tok, eng
    _hawk = LittleHawkInference(tokenizer=tok, engine=eng)


def _stream_generator(prompt: str, max_tokens: int, temperature: float,
                      top_k: int, top_p: float, rep_penalty: float) -> AsyncGenerator[str, None]:
    hawk = _hawk
    tok = _tokenizer
    eng = _engine
    caches = eng.init_cache(); win_ptr = 0
    ids = tok.encode(prompt, add_bos=True)
    generated = [t for t in ids if t not in (tok.bos_id, tok.eos_id)]
    n_ctx = 0
    for tid in ids:
        n_ctx += 1
        logits, caches, win_ptr, _ = eng.step(tid, caches, win_ptr, n_ctx)
        last_logits = logits[0]
    pl = 0
    async def gen():
        nonlocal caches, win_ptr, n_ctx, last_logits
        for step in range(max_tokens):
            nid = hawk._sample(last_logits.copy(), temperature, top_k, top_p,
                               rep_penalty=rep_penalty, generated=generated)
            n_ctx += 1
            logits, caches, win_ptr, _ = eng.step(nid, caches, win_ptr, n_ctx)
            last_logits = logits[0]
            if n_ctx > eng.max_cap:
                pass
            if nid == tok.eos_id:
                break
            generated.append(nid)
            if tok._donor_mode and getattr(tok, "_hf_tok", None):
                decoded = tok._hf_tok.decode([nid])
            else:
                ts = tok.id_to_token.get(nid, tok.UNK)
                decoded = ts.replace("Ġ", " ").replace("Ċ", "\n").replace(tok.SPACE, " ")
            yield f"data: {decoded}\n\n"
        yield "data: [DONE]\n\n"
    return gen()


@app.on_event("startup")
def _startup():
    load_model(DEFAULT_WEIGHTS)


@app.get("/health")
async def health():
    return {"status": "ok", "mode": "weights" if os.path.exists(DEFAULT_WEIGHTS) else "demo"}


@app.post("/generate")
async def generate(body: dict = Body(...)):
    prompt = body.get("prompt")
    if not prompt:
        raise HTTPException(400, "prompt é obrigatório")
    max_tokens = int(body.get("max_tokens", 80))
    temperature = float(body.get("temperature", 0.7))
    top_k = int(body.get("top_k", 40))
    top_p = float(body.get("top_p", 0.92))
    rep_penalty = float(body.get("rep_penalty", 1.15))
    gen = _stream_generator(prompt, max_tokens, temperature, top_k, top_p, rep_penalty)
    return StreamingResponse(gen, media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
