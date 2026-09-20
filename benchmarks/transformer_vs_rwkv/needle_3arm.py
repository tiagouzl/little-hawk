#!/usr/bin/env python3
"""Needle 3 braços: fifo vs nexus-salience (Transformer) vs RWKV (sem eviction).

Reuso dos arms de run_benchmark.py (in-process, greedy). RWKV é braço
arquitetural — NÃO é evictor. Agulha HAWK-7319 em filler PT-BR.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from run_benchmark import TransformerArm, RwkvArm, reset_peak

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(BASE, "benchmarks/transformer_vs_rwkv/prompts.json")) as _f:
    PROMPTS = json.load(_f)
NEEDLE = "HAWK-7319"


class SalienceArm(TransformerArm):
    name = "smollm2-135m-nexussalience"

    def load(self):
        super().load()
        self.eng.eviction_name = "nexus-salience"
        from engine.eviction import NexusSalienceEviction

        self.eng.eviction = NexusSalienceEviction(S=4, W=508, seed=42)


def build_needle(encode, filler, ctx_tokens, depth):
    words, i = [], 0
    while len(encode(" ".join(words))) < ctx_tokens:
        words.extend(filler[i % len(filler)].split())
        i += 1
    ids = encode(" ".join(words))[:ctx_tokens]
    needle_ids = encode(" " + NEEDLE + " ")
    pos = int(len(ids) * depth)
    q = encode("\n\nPergunta: qual é o identificador secreto mencionado acima? Responda apenas com ele.")
    return ids[:pos] + needle_ids + ids[pos:] + q


def greedy(arm, ids, gen, sampler=None, rng=None):
    logits, st, wp, nctx, _ = arm.prefill(ids)
    out = []
    for _ in range(gen):
        if sampler is None:
            p = logits.astype(np.float64)
            nid = int(np.argmax(p - p.max()))
        else:
            nid = sampler.sample(logits, generated=out, rng=rng)
        if arm.eos() is not None and nid == arm.eos():
            break
        logits, st, wp, nctx, _ = arm.step(nid, st, wp, nctx)
        out.append(nid)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, nargs="+", default=[600, 1200])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gen", type=int, default=12)
    ap.add_argument("--arms", nargs="+", default=["fifo", "salience", "rwkv"])
    ap.add_argument(
        "--sample",
        action="store_true",
        help="sampling estabilizado do repo (temp 0.7/top40/min_p 0.05, seed fixa) em vez de greedy",
    )
    ap.add_argument("--out", default="benchmarks/transformer_vs_rwkv/results_needle.json")
    a = ap.parse_args()

    arms = {"fifo": TransformerArm(), "salience": SalienceArm(), "rwkv": RwkvArm()}
    for name in a.arms:
        print(f"load {name}…", flush=True)
        arms[name].load()

    sampler, rng = None, None
    if a.sample:
        from runtime.inference import Sampler, SamplingConfig

        sampler = Sampler(
            SamplingConfig(max_tokens=a.gen, temperature=0.7, top_k=40, top_p=0.92, rep_penalty=1.15, min_p=0.05)
        )
        rng = np.random.default_rng(42)

    trials = []
    for ctx in a.contexts:
        for depth in a.depths:
            for rep in range(a.reps):
                for name in a.arms:
                    arm = arms[name]
                    ids = build_needle(arm.encode, PROMPTS["filler_pt"], ctx, depth)
                    reset_peak()
                    out = greedy(arm, ids, a.gen, sampler, rng)
                    text = arm.tok.decode(out) if hasattr(arm.tok, "decode") else ""
                    # RWKV decode via tokenizer dedicado
                    if name == "rwkv":
                        text = arm.tok.decode(out)
                    hit = NEEDLE in text
                    trials.append(
                        {"arm": arm.name, "ctx": ctx, "depth": depth, "rep": rep, "hit": hit, "text": text[:120]}
                    )
                    print(
                        f"ctx={ctx} d={depth} rep={rep} {arm.name}: {'HIT' if hit else 'miss'} :: {text[:80]!r}",
                        flush=True,
                    )

    print(f"\n{'arm':<28}{'ctx':>6}{'depth':>7}{'acc':>6}")
    for name in a.arms:
        for ctx in a.contexts:
            for depth in a.depths:
                ts = [t for t in trials if t["arm"] == arms[name].name and t["ctx"] == ctx and t["depth"] == depth]
                acc = sum(t["hit"] for t in ts) / len(ts)
                print(f"{arms[name].name:<28}{ctx:>6}{depth:>7.1f}{acc:>6.2f}")

    with open(os.path.join(BASE, a.out), "w") as f:
        json.dump({"config": vars(a), "trials": trials}, f, indent=1)
    print(os.path.join(BASE, a.out))


if __name__ == "__main__":
    main()
