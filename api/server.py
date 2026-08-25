#!/usr/bin/env python3
"""API FastAPI para o Little Hawk.

- /health         → status JSON
- /generate (POST)→ text/event-stream (SSE) com tokens gerados
  body: {"prompt": "...", "max_tokens": 80, "temperature": 0.7, "top_k": 40, "top_p": 0.92, "rep_penalty": 1.15}

Reutiliza os módulos runtime/ e engine/. Se LITTLE_HAWK_WEIGHTS não existir,
cai no modo demo (pesos aleatórios).

Concorrência: um semáforo limita gerações simultâneas (LITTLE_HAWK_MAX_CONCURRENCY,
padrão 2). Desconexão do cliente cancela a inferência de forma cooperativa.
Cada requisição usa RNG própria — resultados não interferem entre si.

Limitação de escopo (intencional, educacional):
- Modelo é estado global single-process (`_hawk`/`_tok` carregados uma vez no lifespan).
  Não há sharding, replicação ou auth/rate-limit — é demo, não produção.
- O semáforo + timeout (`LITTLE_HAWK_TIMEOUT_SECS`) são os únicos controles
  operacionais; escalar horizontalmente requer múltiplos processos/containers
  com balanceador externo.
"""

import asyncio
import json
import os
import queue as _queue_mod
import threading
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from engine import get_engine
from engine.engine import MultiLayerEngine
from runtime.inference import LittleHawkInference, SamplingConfig
from runtime.tokenizer import BPETokenizer, CORPUS

MAX_CONCURRENCY = int(os.getenv("LITTLE_HAWK_MAX_CONCURRENCY", "2"))
TIMEOUT_SECS = float(os.getenv("LITTLE_HAWK_TIMEOUT_SECS", "300"))
DEFAULT_WEIGHTS = os.getenv("LITTLE_HAWK_WEIGHTS", "little_hawk_weights.npz")
EVICTION = os.getenv("LITTLE_HAWK_EVICTION", "fifo")

# Modelo carregado uma vez por processo (single-process, escopo educacional — ver docstring)
_hawk = None
_tok = None
_gen_semaphore: asyncio.Semaphore | None = None
_load_lock = threading.Lock()


def _ensure_semaphore() -> asyncio.Semaphore:
    global _gen_semaphore
    if _gen_semaphore is None:
        _gen_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    return _gen_semaphore


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model(DEFAULT_WEIGHTS)
    _ensure_semaphore()
    yield


app = FastAPI(title="Little Hawk API", version="0.4.0", lifespan=lifespan)


class ClientDisconnected(Exception):
    pass


def load_model(weights_path: str | None = None):
    """Carrega tokenizer/engine. Fallback para modo demo se pesos ausentes.

    Thread-safe via _load_lock — evita corrida no lifespan + primeira requisição.
    Estado global single-process é intencional (demo); para produção use múltiplos
    workers/containers com balanceador externo.
    """
    global _hawk, _tok
    if _hawk is not None:
        return
    with _load_lock:
        if _hawk is not None:  # double-checked após adquirir lock
            return
        tok = BPETokenizer()
        if weights_path and os.path.exists(weights_path):
            meta = weights_path.replace(".npz", "_meta.json")
            if not os.path.exists(meta):
                raise FileNotFoundError(f"Meta não encontrado: {meta}")
            tok.load_donor_vocab(meta)
            with open(meta, encoding="utf-8") as f:
                m = json.load(f)
            eng = get_engine(
                d_model=int(m.get("d_model", 576)),
                n_heads=int(m.get("n_heads", 9)),
                n_layers=int(m.get("n_layers", 30)),
                sink_size=4,
                window_size=508,
                vocab_size=int(m.get("vocab_size", len(tok.vocab))),
                eviction=EVICTION,
            )
            eng.load_weights(weights_path)
        else:
            tok.train(CORPUS, vocab_size=512, verbose=False)
            eng = get_engine(
                d_model=128,
                n_heads=4,
                n_layers=2,
                sink_size=4,
                window_size=28,
                vocab_size=len(tok.vocab),
                eviction=EVICTION,
            )
        _tok = tok
        _hawk = LittleHawkInference(tokenizer=tok, engine=eng)


def _blocking_stream(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    rep_penalty: float,
    out_q: _queue_mod.Queue,
    cancel: threading.Event,
    min_p: float = 0.0,
):
    """Roda em thread (CPU-bound). Empurra chunks decodificados; aborta via cancel.

    Um `ClientDisconnected` levantado no callback propaga por generate() e encerra
    a inferência no token seguinte — cancelamento cooperativo sem desperdiçar CPU.
    """
    cfg = SamplingConfig(
        max_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p, rep_penalty=rep_penalty, min_p=min_p
    )

    def on_token(text: str, step: int, stats: dict):
        if text:
            out_q.put(text)
        if cancel.is_set():
            raise ClientDisconnected()

    try:
        _hawk.generate(prompt, sampling_config=cfg, telemetry=None, on_token=on_token)
        out_q.put(None)  # fim normal
    except ClientDisconnected:
        out_q.put(None)


_DONE = 'data: {"token": "[DONE]"}\n\n'
_TIMEOUT_MSG = 'data: {"error": "timeout de inferência — aumente LITTLE_HAWK_TIMEOUT_SECS ou reduza max_tokens"}\n\n'


async def _stream_sse(
    prompt: str, max_tokens: int, temperature: float, top_k: int, top_p: float, rep_penalty: float, min_p: float = 0.0
):
    """Produz SSE segurando o semáforo durante todo o stream.

    O produtor roda em thread; se o cliente desconecta, o generator é fechado,
    `cancel` é ativado e a thread de inferência aborta no próximo token.
    Excede o timeout global (LITTLE_HAWK_TIMEOUT_SECS) → evento de erro + fim.
    """
    loop = asyncio.get_running_loop()
    out_q: _queue_mod.Queue = _queue_mod.Queue()
    cancel = threading.Event()

    async with _ensure_semaphore():
        producer = threading.Thread(
            target=_blocking_stream,
            args=(prompt, max_tokens, temperature, top_k, top_p, rep_penalty, out_q, cancel, min_p),
            daemon=True,
        )
        producer.start()
        deadline = loop.time() + TIMEOUT_SECS
        timed_out = False
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    chunk = await asyncio.wait_for(loop.run_in_executor(None, out_q.get), timeout=remaining)
                except asyncio.TimeoutError:
                    timed_out = True
                    break
                if chunk is None:
                    break
                payload = json.dumps({"token": chunk}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
            if timed_out:
                yield _TIMEOUT_MSG
            yield _DONE
        finally:
            cancel.set()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mode": "weights" if os.path.exists(DEFAULT_WEIGHTS) else "demo",
        "max_concurrency": MAX_CONCURRENCY,
        "timeout_secs": TIMEOUT_SECS,
        "eviction": EVICTION,
    }


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Texto de entrada", max_length=8000)
    max_tokens: int = Field(80, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0)
    top_k: int = Field(40, ge=1)
    top_p: float = Field(0.92, ge=0.0, le=1.0)
    rep_penalty: float = Field(1.15, ge=0.0)
    min_p: float = Field(0.0, ge=0.0, le=1.0)


@app.post("/generate")
async def generate(req: GenerateRequest):
    if not req.prompt:
        raise HTTPException(400, "prompt é obrigatório")
    return StreamingResponse(
        _stream_sse(req.prompt, req.max_tokens, req.temperature, req.top_k, req.top_p, req.rep_penalty, req.min_p),
        media_type="text/event-stream",
    )


@app.get("/")
async def root():
    demo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sse_demo.html")
    if os.path.exists(demo_path):
        from fastapi.responses import FileResponse

        return FileResponse(demo_path)
    return {"message": "Little Hawk API v0.4.0"}
