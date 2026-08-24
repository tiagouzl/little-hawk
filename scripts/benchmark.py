#!/usr/bin/env python3
"""
scripts/benchmark.py — Benchmarks automatizados do Little Hawk.

Métricas:
  memória    pico RSS (carregamento + inferência) e pegada teórica do cache O(1)
  latência   ms/token por fase (enchimento vs estacionária), percentis p50/p95
  qualidade  teacher-forced NLL em texto único; drift da geração livre (%ascii, loops)
  opcional   comparação de NLL contra o HF Transformers com contexto completo (--compare-hf)

Uso:
  python scripts/benchmark.py --weights little_hawk_weights.npz [--gen-tokens 600] [--json saida.json]
"""
import argparse
import json
import math
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.engine import MultiLayerEngine
from runtime.inference import Sampler, SamplingConfig
from runtime.tokenizer import BPETokenizer, StreamDecoder

PROMPT = "The development of modern science began when thinkers decided"
PARAS = [
    "The Amazon rainforest produces roughly twenty percent of terrestrial oxygen through photosynthesis.",
    "Medieval blacksmiths heated iron in charcoal forges until it glowed yellow, hammering out impurities layer by layer.",
    "The Voyager probes launched in 1977 carrying golden records with greetings in fifty-five languages.",
    "Octopuses edit their own RNA extensively, which may explain their complex behavior despite a small genome.",
    "Roman concrete incorporated volcanic ash that reacted with seawater, strengthening harbor walls over centuries.",
    "Honeybees encode the direction of flowers in a waggle dance angled relative to the sun.",
    "Deep ocean vents host ecosystems powered by dissolved minerals rather than sunlight.",
    "Gutenberg's press relied on oil based ink and an alloy that cast crisp reusable letters daily.",
]
NLL_TEXT = " ".join(PARAS)  # passagem única — sem repetição


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def percentile(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Benchmark Little Hawk")
    ap.add_argument("--weights", default="little_hawk_weights.npz")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--gen-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--min-p", type=float, default=0.05,
                    help="min_p para a geração livre do benchmark (padrão 0.05)")
    ap.add_argument("--compare-hf", action="store_true",
                    help="compara NLL com contexto completo via transformers (requere torch)")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    meta_path = args.weights.replace(".npz", "_meta.json")
    m = json.loads(Path(meta_path).read_text(encoding="utf-8"))

    result = {"config": {"weights": args.weights, "gen_tokens": args.gen_tokens,
                         "temperature": args.temperature, "min_p": args.min_p}}

    # ── Carregamento ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    tok = BPETokenizer()
    tok.load_donor_vocab(meta_path)
    eng = MultiLayerEngine(d_model=m["d_model"], n_heads=m["n_heads"], n_layers=m["n_layers"],
                           sink_size=4, window_size=508, vocab_size=m["vocab_size"])
    eng.load_weights(args.weights)
    load_s = time.perf_counter() - t0

    cache_bytes = eng.n_layers * 2 * eng.max_cap * eng.n_heads * eng.d_k * 4
    result["memoria"] = {
        "pico_rss_mb_apos_load": round(rss_mb(), 1),
        "cache_kv_bytes": cache_bytes,
        "cache_kv_mb": round(cache_bytes / 1e6, 1),
        "constante": True,
    }

    # ── Latência + geração livre atravessando max_cap ──────────────────────────
    sampler = Sampler(SamplingConfig(max_tokens=args.gen_tokens, temperature=args.temperature,
                                     top_k=40, top_p=0.92, rep_penalty=1.15, min_p=args.min_p))
    np.random.seed(3)
    caches = eng.init_cache(); buf_ids = [id(c[0]) for c in caches]
    wp = 0; sdec = StreamDecoder(tok)
    ids = tok.encode(args.prompt, add_bos=True)
    generated = list(ids); n_ctx = 0
    lat = {"fill": [], "stat": []}

    for tid in ids:
        n_ctx += 1
        out, caches, wp, _ = eng.step(tid, caches, wp, n_ctx)
        last = out[0]

    t_all = time.perf_counter()
    eos_at = None
    for step in range(args.gen_tokens):
        t = time.perf_counter()
        nid = sampler.sample(last.copy(), generated=generated)
        n_ctx += 1
        out, caches, wp, _ = eng.step(nid, caches, wp, n_ctx)
        last = out[0]
        ms = (time.perf_counter() - t) * 1000
        lat["fill" if n_ctx <= eng.max_cap else "stat"].append(ms)
        if nid == tok.eos_id:
            eos_at = step + 1
            break
        generated.append(nid)
        sdec.push(nid)
    sdec.flush()
    total_s = time.perf_counter() - t_all
    gen_only = generated[len(ids):]
    buffers_ok = all(id(c[0]) == b for c, b in zip(caches, buf_ids))

    def block(name, xs):
        return {"tokens": len(xs), "ms_p50": round(percentile(xs, 50), 1),
                "ms_p95": round(percentile(xs, 95), 1)}

    result["latencia"] = {
        "load_segundos": round(load_s, 1),
        "total_segundos": round(total_s, 1),
        "ms_token_medio": round(total_s / max(len(gen_only), 1) * 1000, 1),
        "enchimento": block("fill", lat["fill"]),
        "estacionaria": block("stat", lat["stat"]),
    }
    result["memoria"]["buffers_reutilizados"] = buffers_ok

    # ── Qualidade: geração livre ───────────────────────────────────────────────
    def ngram_top(seq, n=8):
        grams = Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))
        return grams.most_common(1)[0][1] if grams else 0

    def ascii_ratio(seq):
        """Fração de caracteres ASCII imprimíveis no TEXTO decodificado."""
        txt = tok.decode(seq)
        okc = sum(1 for c in txt if 32 <= ord(c) < 127 or c in "\n\r\t")
        return round(okc / max(len(txt), 1), 3)

    half = len(gen_only) // 2
    result["qualidade_geracao"] = {
        "tokens": len(gen_only),
        "eos_natural_em": eos_at,
        "ascii_primeira_metade": ascii_ratio(gen_only[:half]),
        "ascii_segunda_metade": ascii_ratio(gen_only[half:]),
        "top8gram_max": ngram_top(gen_only),
    }

    # ── Qualidade: NLL teacher-forced (texto único, sem repetição) ─────────────
    nll_ids = tok.encode(NLL_TEXT, add_bos=False)
    caches = eng.init_cache(); wp = 0
    nll = []
    for i, tid in enumerate(nll_ids, start=1):
        out, caches, wp, _ = eng.step(tid, caches, wp, i)
        if i < len(nll_ids):
            ls = out[0] - out[0].max()
            nll.append(float(np.log(np.exp(ls).sum()) - ls[nll_ids[i]]))
    result["qualidade_nll_streaming"] = {
        "tokens": len(nll),
        "media": round(float(np.mean(nll)), 3),
        "ppl_equivalente": round(math.exp(np.mean(nll)), 1),
    }

    # ── Opcional: baseline HF contexto completo ────────────────────────────────
    if args.compare_hf:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            mid = "HuggingFaceTB/SmolLM-135M"
            rtok = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32); model.eval()
            rids = rtok.encode(NLL_TEXT)
            past = None; pos = 0; hnll = []
            with torch.no_grad():
                for i in range(len(rids)):
                    o = model(input_ids=torch.tensor([[rids[i]]]),
                              position_ids=torch.tensor([[pos]]), past_key_values=past, use_cache=True)
                    past = o.past_key_values; pos += 1
                    if i + 1 < len(rids):
                        lp = torch.log_softmax(o.logits[0, -1], dim=-1)
                        hnll.append(float(-lp[rids[i + 1]]))
            result["qualidade_nll_fullctx"] = {
                "tokens": len(hnll), "media": round(float(np.mean(hnll)), 3),
                "delta_vs_streaming": round(float(np.mean(hnll)) - float(np.mean(nll)), 3),
            }
        except ImportError:
            result["qualidade_nll_fullctx"] = {"erro": "torch/transformers não instalados"}

    # ── Relatório ──────────────────────────────────────────────────────────────
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[salvo em {args.json_out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
