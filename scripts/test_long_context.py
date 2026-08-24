#!/usr/bin/env python3
"""
scripts/test_long_context.py — Teste de estabilidade em contexto longo (>512 tokens).

Exercita a fase estacionária do StreamingKVCache com pesos reais:
  - posições congeladas após max_cap (sink 0..3, janela 4..511)
  - memória O(1): mesmos buffers do início ao fim
  - heurística de degradação: frequência de 8-grams repetidos por janela

Uso:
  python scripts/test_long_context.py --weights little_hawk_weights.npz --tokens 1500
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.engine import MultiLayerEngine
from runtime.inference import Sampler, SamplingConfig
from runtime.tokenizer import BPETokenizer, StreamDecoder


def ngram_top(ids, n=8):
    """Contagem do 8-gram mais repetido na janela (loop detector)."""
    if len(ids) < n:
        return 0, 0
    grams = Counter(tuple(ids[i : i + n]) for i in range(len(ids) - n + 1))
    top = grams.most_common(1)[0][1]
    return top, len(grams)


def ascii_ratio(tok, ids):
    """Fração de tokens cujo texto é ASCII imprimível (detector de colapso byte-soup)."""
    ok = sum(1 for t in ids if all(32 <= ord(c) < 127 for c in tok.id_to_token.get(t, "?")))
    return ok / max(len(ids), 1)


def main():
    p = argparse.ArgumentParser(description="Teste de contexto longo — fase estacionária")
    p.add_argument("--weights", type=str, default="little_hawk_weights.npz")
    p.add_argument("--prompt", type=str, default="The history of artificial intelligence began")
    p.add_argument("--tokens", type=int, default=1500)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    meta_path = args.weights.replace(".npz", "_meta.json")
    tok = BPETokenizer()
    tok.load_donor_vocab(meta_path)
    m = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    eng = MultiLayerEngine(d_model=m["d_model"], n_heads=m["n_heads"],
                           n_layers=m["n_layers"], sink_size=4,
                           window_size=508, vocab_size=m["vocab_size"])
    eng.load_weights(args.weights)

    sampler = Sampler(SamplingConfig(max_tokens=args.tokens, temperature=args.temperature))
    np.random.seed(11)

    caches = eng.init_cache()
    buf_ids = [id(c[0]) for c in caches]          # identidade dos buffers K
    win_ptr = 0
    sdec = StreamDecoder(tok)
    ids = tok.encode(args.prompt, add_bos=True)
    generated = [t for t in ids if t not in (tok.bos_id, tok.eos_id)]
    n_ctx, ev, wraps = 0, 0, 0
    phase_lat = {"fill": [], "stationary": []}
    last_logits = None

    print(f"prompt={args.prompt!r}  alvo={args.tokens} tokens  max_cap={eng.max_cap}")
    t_all = time.perf_counter()
    for tid in ids:
        n_ctx += 1
        logits, caches, win_ptr, _ = eng.step(tid, caches, win_ptr, n_ctx)
        last_logits = logits[0]

    for step in range(args.tokens):
        t0 = time.perf_counter()
        nid = sampler.sample(last_logits.copy(), generated=generated)
        n_ctx += 1
        logits, caches, win_ptr, _ = eng.step(nid, caches, win_ptr, n_ctx)
        last_logits = logits[0]
        lat = (time.perf_counter() - t0) * 1000
        if n_ctx > eng.max_cap:
            ev += 1
            phase_lat["stationary"].append(lat)
        else:
            phase_lat["fill"].append(lat)
        prev_wp = win_ptr
        win_ptr = (win_ptr + 1) % eng.W if n_ctx > eng.S else win_ptr
        if win_ptr < prev_wp:
            wraps += 1
        if nid == tok.eos_id:
            print(f"\n[eos natural no step {step}]")
            break
        generated.append(nid)
        if (step + 1) % 250 == 0:
            el = time.perf_counter() - t_all
            print(f"  step {step + 1:>5}  win_ptr={win_ptr:>3}  {el:.0f}s decorridos")

    total = time.perf_counter() - t_all
    same_buffers = all(id(c[0]) == b for c, b in zip(caches, buf_ids))

    # ── Métricas de qualidade por janela ────────────────────────────────────────
    if step + 1 < args.tokens:  # eos natural antes do alvo
        gen_only = generated[len(generated) - (step + 1):]
    else:
        gen_only = generated[len(ids) - 1:]
    W = 500
    print("\n" + "═" * 64)
    print("RESULTADOS — contexto longo")
    print("═" * 64)
    print(f"tokens gerados:      {len(gen_only)}")
    print(f"tempo total:         {total:.0f}s  ({total / max(len(gen_only), 1) * 1000:.0f} ms/token médio)")
    fill = phase_lat["fill"]
    stat = phase_lat["stationary"]
    print(f"latência enchimento: {np.mean(fill):.0f} ms/tok  ({len(fill)} tokens)")
    print(f"latência estacionária: {np.mean(stat):.0f} ms/tok  ({len(stat)} tokens)")
    print(f"evicções lógicas:    {ev}  ·  wraps win_ptr: {wraps}")
    print(f"buffers K idênticos: {'✓ sim (O(1) confirmado)' if same_buffers else '✗ REALOCADOS'}")

    print("\ndegradacao por janela (top-8-gram | % ascii):")
    ok = True
    for wstart in range(0, len(gen_only), W):
        chunk = gen_only[wstart : wstart + W]
        top, uniq = ngram_top(chunk)
        ar = ascii_ratio(tok, chunk)
        flag = ""
        if top > len(chunk) * 0.25:  # mesmo 8-gram em >25% da janela = loop
            flag += "  ← ⚠ possível loop"
            ok = False
        if ar < 0.90:  # >10% de tokens não-ASCII = colapso byte-soup
            flag += "  ← ⚠ colapso para tokens raros"
            ok = False
        print(f"  [{wstart:>4}:{wstart + len(chunk):>4}]  top8gram={top:>3}x  ascii={ar * 100:5.1f}%{flag}")

    def show(label, seg):
        txt = "".join(sdec.push(i) for i in seg) + sdec.flush()
        print(f"  {label}: {txt[:90].replace(chr(10), ' ')!r}")

    print("\namostras de texto:")
    show("início ", gen_only[:40])
    mid = len(gen_only) // 2
    show("meio   ", gen_only[mid : mid + 40])
    show("final  ", gen_only[-40:])

    verdict = "ESTÁVEL ✓" if ok and same_buffers else "INSTÁVEL ✗"
    print(f"\nveredito: {verdict}")
    print("\nnota: colapso semântico em geração livre longa é esperado para modelos de 135M —")
    print("a validação matemática (ppl streaming vs contexto completo) está em ANALISE.md.")
    return 0 if ok and same_buffers else 1


if __name__ == "__main__":
    sys.exit(main())
