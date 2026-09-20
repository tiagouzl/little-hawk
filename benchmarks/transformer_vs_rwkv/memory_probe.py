#!/usr/bin/env python3
"""Sonda de memória externa A/B/C/D (§12) — determinística, sem embeddings.

Fato: "O projeto se chama Little Hawk." Pergunta 6 turnos depois.
A: fato no histórico. B: histórico + eviction (fifo). C: histórico SEM o
fato + bloco MEMORY: injetado. D: 5 fatos armazenados, recuperação por
keyword (sem o fato → pergunta; com keyword → injeta o fato recuperado).
Braços: transformer-fifo e RWKV. Greedy, gen curto, hit = contém "little hawk".
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from run_benchmark import TransformerArm, RwkvArm

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(BASE, "benchmarks/transformer_vs_rwkv/prompts.json")) as _f:
    PROMPTS = json.load(_f)

FACT = "O projeto se chama Little Hawk."
QUESTION = "Qual é o nome do projeto?"
DISTRACT = [
    "Gosto de café pela manhã.",
    "Preciso organizar minha semana.",
    "Choveu muito ontem.",
    "Estou pensando em ler mais.",
    "Você gosta de música?",
    "Hoje o dia está bem quente.",
]
STORE = {
    "cor": "Minha cor favorita é verde.",
    "cidade": "Eu moro em Caicó.",
    "numero": "Meu número da sorte é 47.",
    "projeto": FACT,
    "hora": "Eu acordo às 5h30.",
}


def convo(encode, hist_users, final_q, memory_block=None):
    # monta diálogo multi-turno como texto corrido (ambos os tokenizers são PT-capazes)
    parts = []
    for u in hist_users:
        parts.append(f"Usuário: {u}\nAssistente: Entendido.\n")
    if memory_block:
        parts.append(f"MEMORY:\n{memory_block}\n")
    parts.append(f"Usuário: {final_q}\nAssistente:")
    return encode("".join(parts))


def greedy_text(arm, ids, gen=24):
    logits, st, wp, nctx, _ = arm.prefill(ids)
    out = []
    for _ in range(gen):
        p = logits.astype(np.float64)
        nid = int(np.argmax(p - p.max()))
        if arm.eos() is not None and nid == arm.eos():
            break
        logits, st, wp, nctx, _ = arm.step(nid, st, wp, nctx)
        out.append(nid)
    return arm.tok.decode(out)


def retrieve(query):
    keys = [w.strip("?.!,").lower() for w in query.split() if len(w) > 3]
    for k, fact in STORE.items():
        if k in keys:
            return fact
    return None


def main():
    arms = {"transformer": TransformerArm(), "rwkv": RwkvArm()}
    for name, arm in arms.items():
        print(f"load {name}…", flush=True)
        arm.load()
    dist = DISTRACT
    cases = {
        "A_hist": ([FACT] + dist[:6], None),
        "B_hist_evict": ([FACT] + dist[:6], None),  # mesmo input; eviction age no cache
        "C_injected": (dist[:6], FACT),
        "D_retrieved": (dist[:6], retrieve(QUESTION)),
    }
    rows = []
    for name, arm in arms.items():
        for case, (hist, mem) in cases.items():
            ids = convo(arm.encode, hist, QUESTION, mem)
            text = greedy_text(arm, ids)
            hit = "little hawk" in text.lower()
            rows.append({"arm": arm.name, "case": case, "hit": hit, "text": text[:120]})
            print(f"{arm.name} {case}: {'HIT' if hit else 'miss'} :: {text[:90]!r}", flush=True)
    with open(os.path.join(BASE, "benchmarks/transformer_vs_rwkv/results_memory.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
