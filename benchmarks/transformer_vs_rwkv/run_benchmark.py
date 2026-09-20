#!/usr/bin/env python3
"""Protocolo comum Transformer (SmolLM2-135M) vs RWKV-7 g1 0.1B.

In-process, greedy (argmax — determinístico; qualidade contaminada por
greedy é aceita, latência/RSS não). Mesma máquina, A/B intercalado por
contexto. Sem torch no runtime (conversão foi offline em §4).

Uso:
    venv/bin/python benchmarks/transformer_vs_rwkv/run_benchmark.py \
        --contexts 128 512 --reps 2 --gen 16 --out benchmarks/transformer_vs_rwkv/results.json

Métricas por (modelo, contexto, rep): prefill_ms, TTFT_ms (=prefill p/ ambos;
RWKV prefill é sequencial por construção — medição honesta), decode_ms,
tok_s, RSS por estágio (VmHWM com reset), KV/state bytes teóricos.
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def reset_peak():
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
    except Exception:
        pass


def vmhwm():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return None


class TransformerArm:
    name = "smollm2-135m-instruct"

    def load(self):
        from runtime.tokenizer import BPETokenizer
        from engine import MultiLayerEngine
        import json as _j

        w = os.path.join(BASE, "smollm2_135m_instruct_weights.npz")
        with open(w.replace(".npz", "_meta.json"), encoding="utf-8") as _f:
            m = _j.load(_f)
        self.tok = BPETokenizer()
        self.tok.load_donor_vocab(w.replace(".npz", "_meta.json"))
        self.eng = MultiLayerEngine(
            d_model=m["d_model"],
            n_heads=m["n_heads"],
            n_layers=m["n_layers"],
            sink_size=4,
            window_size=508,
            vocab_size=m["vocab_size"],
            rope_base=m["rope_base"],
            eviction="fifo",
        )
        self.eng.load_weights(w)
        import numpy as _np

        _d = _np.load(w, allow_pickle=False)
        self._params = int(sum(_d[k].size for k in _d.files if not k.startswith("_meta")))

    def params(self):
        return self._params  # contado do .npz (sem _meta)

    def state_bytes_theory(self):
        e = self.eng
        return e.n_layers * 2 * e.max_cap * e.n_heads * (e.d_model // e.n_heads) * 4

    def encode(self, s):
        return self.tok.encode(s, add_bos=False)

    def prefill(self, ids):
        st = self.eng.init_cache()
        t0 = time.perf_counter()
        logits, st, wp, _ = self.eng.prefill(ids, st)
        return logits[0], st, wp, len(ids), (time.perf_counter() - t0) * 1000

    def step(self, tid, st, wp, nctx):
        t0 = time.perf_counter()
        logits, st, wp, _ = self.eng.step(tid, st, wp, nctx)
        return logits[0], st, wp, nctx + 1, (time.perf_counter() - t0) * 1000

    def eos(self):
        return self.tok.eos_id


class RwkvArm:
    name = "rwkv7-g1d-0.1b"

    def load(self):
        from engine.rwkv_engine import Rwkv7Engine
        from engine.rwkv_tokenizer import RwkvTokenizer

        self.tok = RwkvTokenizer(os.path.join(BASE, "data/rwkv/rwkv_vocab_v20230424.txt"))
        self.eng = Rwkv7Engine()
        self.eng.load_weights(os.path.join(BASE, "data/rwkv/rwkv7_g1d_01b_fp32.npz"))

    def params(self):
        return 191_084_544  # medido do checkpoint (§4)

    def state_bytes_theory(self):
        return self.eng.state_bytes()

    def encode(self, s):
        return self.tok.encode(s)

    def prefill(self, ids):
        st = self.eng.init_cache()
        t0 = time.perf_counter()
        logits, st, wp, _ = self.eng.prefill(ids, st)
        return logits[0], st, wp, len(ids), (time.perf_counter() - t0) * 1000

    def step(self, tid, st, wp, nctx):
        t0 = time.perf_counter()
        logits, st, wp, _ = self.eng.step(tid, st, wp, nctx)
        return logits[0], st, wp, nctx + 1, (time.perf_counter() - t0) * 1000

    def eos(self):
        return None


def build_context(tok_encode, filler, target_tokens):
    words, i = [], 0
    while True:
        words.extend(filler[i % len(filler)].split())
        i += 1
        ids = tok_encode(" ".join(words))
        if len(ids) >= target_tokens:
            return ids[:target_tokens]


def run_trial(arm, ids, gen):
    reset_peak()
    logits, st, wp, nctx, pre_ms = arm.prefill(ids)
    rss_prefill = vmhwm()
    reset_peak()
    gen_ms, out = 0.0, []
    for _ in range(gen):
        p = logits.astype(np.float64)
        p -= p.max()
        nid = int(np.argmax(p))
        if arm.eos() is not None and nid == arm.eos():
            break
        logits, st, wp, nctx, ms = arm.step(nid, st, wp, nctx)
        gen_ms += ms
        out.append(nid)
    return {
        "prefill_ms": round(pre_ms, 1),
        "ttft_ms": round(pre_ms, 1),
        "decode_ms": round(gen_ms, 1),
        "tok_s": round(len(out) / (gen_ms / 1000), 2) if gen_ms > 0 else 0.0,
        "generated": len(out),
        "rss_prefill_mb": rss_prefill,
        "rss_decode_mb": vmhwm(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, nargs="+", default=[128, 512, 1024, 2048])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--arms", nargs="+", default=["transformer", "rwkv"])
    ap.add_argument("--out", default="benchmarks/transformer_vs_rwkv/results.json")
    a = ap.parse_args()

    with open(os.path.join(BASE, "benchmarks/transformer_vs_rwkv/prompts.json")) as f:
        prompts = json.load(f)
    arms = {"transformer": TransformerArm(), "rwkv": RwkvArm()}
    for name in a.arms:
        print(f"load {name}…")
        arms[name].load()
    reset_peak()
    rss_baseline = vmhwm()

    results = {"config": vars(a), "rss_baseline_mb": rss_baseline, "trials": []}
    for ctx in a.contexts:
        for rep in range(a.reps):
            for name in a.arms:  # A/B intercalado por (ctx, rep)
                arm = arms[name]
                ids = build_context(arm.encode, prompts["filler_pt"], ctx)
                t = run_trial(arm, ids, a.gen)
                t.update(
                    {
                        "model": arm.name,
                        "params": arm.params(),
                        "context": ctx,
                        "rep": rep,
                        "state_bytes_theory": arm.state_bytes_theory(),
                    }
                )
                results["trials"].append(t)
                print(
                    f"ctx={ctx} rep={rep} {arm.name}: prefill={t['prefill_ms']}ms "
                    f"tok/s={t['tok_s']} rss_pre={t['rss_prefill_mb']}MB",
                    flush=True,
                )

    # resumo média ± desvio
    print(f"\n{'modelo':<24}{'ctx':>6}{'prefill_ms':>12}{'tok/s':>8}{'rss_pre_MB':>12}")
    for name in a.arms:
        for ctx in a.contexts:
            ts = [t for t in results["trials"] if t["model"] == arms[name].name and t["context"] == ctx]
            pm = statistics.mean(t["prefill_ms"] for t in ts)
            tm = statistics.mean(t["tok_s"] for t in ts)
            rm = statistics.mean(t["rss_prefill_mb"] for t in ts if t["rss_prefill_mb"])
            results.setdefault("summary", {})[f"{arms[name].name}@{ctx}"] = {
                "prefill_ms_mean": round(pm, 1),
                "tok_s_mean": round(tm, 2),
                "rss_prefill_mb_mean": round(rm, 1),
                "n": len(ts),
            }
            print(f"{arms[name].name:<24}{ctx:>6}{pm:>12.1f}{tm:>8.2f}{rm:>12.1f}")

    out = os.path.join(BASE, a.out)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\n{out}")


if __name__ == "__main__":
    main()
