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
import threading
import json
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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


def _blocking_stream(prompt: str, max_tokens: int, temperature: float,
                     top_k: int, top_p: float, rep_penalty: float):
    """Gerador síncrono de tokens (CPU-bound)."""
    hawk = _hawk; tok = _tokenizer; eng = _engine
    caches = eng.init_cache(); win_ptr = 0
    ids = tok.encode(prompt, add_bos=True)
    generated = [t for t in ids if t not in (tok.bos_id, tok.eos_id)]
    n_ctx = 0
    for tid in ids:
        n_ctx += 1
        logits, caches, win_ptr, _ = eng.step(tid, caches, win_ptr, n_ctx)
        last_logits = logits[0]
    for _ in range(max_tokens):
        nid = hawk._sample(last_logits.copy(), temperature, top_k, top_p,
                           rep_penalty=rep_penalty, generated=generated)
        n_ctx += 1
        logits, caches, win_ptr, _ = eng.step(nid, caches, win_ptr, n_ctx)
        last_logits = logits[0]
        if nid == tok.eos_id:
            break
        generated.append(nid)
        if tok._donor_mode and getattr(tok, "_hf_tok", None):
            decoded = tok._hf_tok.decode([nid])
        else:
            ts = tok.id_to_token.get(nid, tok.UNK)
            decoded = ts.replace("Ġ", " ").replace("Ċ", "\n").replace(tok.SPACE, " ")
        yield decoded


def _stream_sse(prompt: str, max_tokens: int, temperature: float,
                top_k: int, top_p: float, rep_penalty: float) -> AsyncGenerator[str, None]:
    """Produz SSE sem bloquear o event loop, usando thread para CPU-bound."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def producer():
        try:
            for token in _blocking_stream(prompt, max_tokens, temperature, top_k, top_p, rep_penalty):
                payload = json.dumps({"token": token}, ensure_ascii=False)
                asyncio.run_coroutine_threadsafe(queue.put(f"data: {payload}\n\n"), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put("data: {\"token\": \"[DONE]\"}\n\n"), loop)
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    threading.Thread(target=producer, daemon=True).start()

    async def consumer():
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    return consumer()


@app.on_event("startup")
def _startup():
    load_model(DEFAULT_WEIGHTS)


@app.get("/health")
async def health():
    return {"status": "ok", "mode": "weights" if os.path.exists(DEFAULT_WEIGHTS) else "demo"}


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Texto de entrada")
    max_tokens: int = Field(80, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0)
    top_k: int = Field(40, ge=1)
    top_p: float = Field(0.92, ge=0.0, le=1.0)
    rep_penalty: float = Field(1.15, ge=0.0)


@app.post("/generate")
async def generate(req: GenerateRequest):
    if not req.prompt:
        raise HTTPException(400, "prompt é obrigatório")
    gen = _stream_sse(req.prompt, req.max_tokens, req.temperature, req.top_k, req.top_p, req.rep_penalty)
    return StreamingResponse(gen, media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
