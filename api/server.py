# API FastAPI para Little Hawk

from fastapi import FastAPI, Request
from pydantic import BaseModel
from runtime.inference import LittleHawkInference, SamplingConfig, ConsoleTelemetry
from runtime.tokenizer import BPETokenizer
from engine.engine import MultiLayerEngine
import numpy as np

app = FastAPI(title="Little Hawk API")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 80
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.92
    rep_penalty: float = 1.15

@app.post("/generate")
async def generate(req: GenerateRequest):
    # Exemplo: inicialização dummy (substitua por loading real de pesos/tokenizer)
    tok = BPETokenizer(); tok.train("exemplo", vocab_size=128, verbose=False)
    engine = MultiLayerEngine(d_model=128, n_heads=4, n_layers=2, sink_size=4, window_size=28, vocab_size=128)
    hawk = LittleHawkInference(tokenizer=tok, engine=engine)
    cfg = SamplingConfig(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        rep_penalty=req.rep_penalty
    )
    output = hawk.generate(req.prompt, sampling_config=cfg, telemetry=None)
    return {"output": output}
