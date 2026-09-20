#!/usr/bin/env python3
"""Baseline de qualidade conversacional para modelos pequenos (135M/360M).

Uso:
    python chat_baseline.py --selftest
    python chat_baseline.py --adapter meu_adapter:generate --out res_135m.json

Contrato do adapter (função Python):
    generate(messages: list[dict], max_new_tokens: int) -> dict
        obrigatório: "text"
        opcionais:   "mean_logprob" (média do logprob dos tokens gerados),
                     "ttft_s", "decode_tok_s", "peak_rss_mb"
O adapter deve aplicar o chat template do checkpoint Instruct e decodificar
de forma greedy (temperature=0), para o resultado ser reprodutível.

Normalização: texto e padrões são comparados sem acento, em minúsculas.
Portanto os regex em prompts.jsonl estão escritos sem acento.
"""

import argparse
import importlib
import json
import re
import sys
import unicodedata
from collections import defaultdict

PT_STOP = {
    "de",
    "que",
    "o",
    "a",
    "e",
    "um",
    "uma",
    "para",
    "com",
    "nao",
    "em",
    "eu",
    "voce",
    "oi",
    "ola",
    "bom",
    "dia",
    "tudo",
    "bem",
    "obrigado",
    "obrigada",
    "ajuda",
    "posso",
    "sou",
    "ate",
    "mais",
    "foi",
    "prazer",
    "fico",
    "feliz",
    "nada",
    "como",
    "te",
    "se",
    "do",
    "da",
    "os",
    "as",
    "por",
}
EN_STOP = {
    "the",
    "is",
    "are",
    "you",
    "i",
    "and",
    "hello",
    "hi",
    "thanks",
    "thank",
    "what",
    "how",
    "can",
    "help",
    "with",
    "for",
    "to",
    "of",
    "it",
    "this",
    "that",
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.casefold()).strip()


def words(s: str):
    return re.findall(r"\w+", s, flags=re.UNICODE)


def degenerate(text: str) -> bool:
    w = [x.casefold() for x in words(text)]
    if len(w) < 12:
        return False
    tri = list(zip(w, w[1:], w[2:]))
    return (len(tri) - len(set(tri))) / len(tri) > 0.4


def check_item(chk: dict, raw: str):
    """Retorna (ok, lista_de_falhas)."""
    fails = []
    text = unicodedata.normalize("NFC", raw or "")
    t = norm(text)
    if not t:
        return False, ["vazio"]
    if degenerate(text):
        fails.append("degenerado")
    nw = len(words(text))

    if "any_of" in chk and not any(re.search(p, t) for p in chk["any_of"]):
        fails.append("any_of")
    if "all_of" in chk and not all(re.search(p, t) for p in chk["all_of"]):
        fails.append("all_of")
    if "none_of" in chk and any(re.search(p, t) for p in chk["none_of"]):
        fails.append("none_of")
    if "min_words" in chk and nw < chk["min_words"]:
        fails.append("min_words")
    if "max_words" in chk and nw > chk["max_words"]:
        fails.append("max_words")
    if "max_sentences" in chk:
        sents = [x for x in re.split(r"[.!?]+", text) if x.strip()]
        if len(sents) > chk["max_sentences"]:
            fails.append("max_sentences")
    if "n_lines" in chk and len([x for x in text.splitlines() if x.strip()]) != chk["n_lines"]:
        fails.append("n_lines")
    if "equals" in chk:
        bare = norm(re.sub(r"[^\w\s]", "", text))
        if bare not in chk["equals"]:
            fails.append("equals")
    if "equals_raw" in chk and text.strip().strip("\"'“”‘’.!") not in chk["equals_raw"]:
        fails.append("equals_raw")
    if "fullmatch" in chk and not re.fullmatch(chk["fullmatch"], t):
        fails.append("fullmatch")
    if chk.get("lang_pt"):
        toks = [x.casefold() for x in re.findall(r"\w+", t)]
        pt = sum(x in PT_STOP for x in toks)
        en = sum(x in EN_STOP for x in toks)
        if pt < 1 or en > pt:
            fails.append("lang_pt")
    return (len(fails) == 0), fails


def auroc(pos, neg):
    """P(score_pos > score_neg) com empates = 0.5."""
    if not pos or not neg:
        return None
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return s / (len(pos) * len(neg))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def run(items, gen, max_new_tokens):
    rows = []
    for it in items:
        out = gen(it["messages"], max_new_tokens)
        ok, fails = check_item(it["checks"], out["text"])
        rows.append(
            {
                "id": it["id"],
                "category": it["category"],
                "ok": ok,
                "fails": fails,
                "distance_turns": it.get("distance_turns"),
                "text": out["text"],
                **{
                    k: out.get(k)
                    for k in (
                        "mean_logprob",
                        "min_logprob",
                        "first_logprob",
                        "margin_top1_top2",
                        "ttft_s",
                        "decode_tok_s",
                        "peak_rss_mb",
                        "load_rss_mb",
                    )
                },
            }
        )
    return rows


CONF_SIGNALS = ("mean_logprob", "min_logprob", "first_logprob", "margin_top1_top2")


def summarize_conf(rows):
    """AUROC por categoria para cada sinal de confiança (sem re-rodar)."""
    print("\ncomparação de sinais de confiança (AUROC → acerto, por categoria):")
    cats = sorted({r["category"] for r in rows})
    print(f"  {'sinal':<18}" + "".join(f"{c:>10}" for c in cats))
    for sig in CONF_SIGNALS:
        line = f"  {sig:<18}"
        for c in cats:
            sub = [r for r in rows if r["category"] == c and r.get(sig) is not None]
            a = auroc([r[sig] for r in sub if r["ok"]], [r[sig] for r in sub if not r["ok"]])
            line += f"{(round(a, 3) if a is not None else 'n/a'):>10}"
        print(line)


def summarize(rows):
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["ok"])
    print(f"\n{'categoria':<12}{'acerto':>10}{'IC95% (Wilson)':>22}")
    for c, v in sorted(by_cat.items()):
        lo, hi = wilson(sum(v), len(v))
        print(f"{c:<12}{sum(v):>4}/{len(v):<5}{lo:>13.2f} – {hi:.2f}")
    tot = [r["ok"] for r in rows]
    lo, hi = wilson(sum(tot), len(tot))
    print(f"{'TOTAL':<12}{sum(tot):>4}/{len(tot):<5}{lo:>13.2f} – {hi:.2f}")

    rec = [r for r in rows if r["category"] == "recall"]
    if rec:
        print("\nrecall por distância (turnos de distração):")
        d = defaultdict(list)
        for r in rec:
            d[r["distance_turns"]].append(r["ok"])
        for k in sorted(d):
            print(f"  {k}: {sum(d[k])}/{len(d[k])}")

    lp = [r for r in rows if r.get("mean_logprob") is not None]
    if lp:
        print("\ncalibração por categoria (AUROC logprob → acerto):")
        print("  (o agregado mistura cópia-do-contexto com paramétrico — referência apenas)")
        for c in sorted({r["category"] for r in lp}):
            sub = [r for r in lp if r["category"] == c]
            a = auroc([r["mean_logprob"] for r in sub if r["ok"]], [r["mean_logprob"] for r in sub if not r["ok"]])
            print(f"  {c:<12}n={len(sub):<4}{a if a is None else round(a, 3)}")
        a_all = auroc([r["mean_logprob"] for r in lp if r["ok"]], [r["mean_logprob"] for r in lp if not r["ok"]])
        print(f"  {'agregado':<12}n={len(lp):<4}{a_all if a_all is None else round(a_all, 3)}  (NÃO usar p/ router)")
    summarize_conf(rows)
    for key, label in (
        ("ttft_s", "TTFT médio (s)"),
        ("decode_tok_s", "decode tok/s médio"),
        ("peak_rss_mb", "pico RSS (MB)"),
    ):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if vals:
            print(f"{label}: {sum(vals) / len(vals):.2f}")


def selftest(items):
    """Valida o scorer: o gold deve passar; texto vazio e degenerado devem falhar."""
    bad = 0
    for it in items:
        ok, f = check_item(it["checks"], it["gold"])
        if not ok:
            bad += 1
            print(f"[FALHA gold] {it['id']}: {f} <- {it['gold']!r}")
        if check_item(it["checks"], "")[0] or check_item(it["checks"], "blá " * 30)[0]:
            bad += 1
            print(f"[FALHA neg] {it['id']}: vazio/degenerado passou")
    print(f"selftest: {len(items) - bad}/{len(items)} itens consistentes")
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="chat_baseline_prompts.jsonl")
    ap.add_argument("--adapter", help="módulo:função")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    items = load(a.prompts)
    if a.selftest:
        sys.exit(0 if selftest(items) else 1)
    if not a.adapter:
        ap.error("--adapter é obrigatório fora do --selftest")
    mod, fn = a.adapter.split(":")
    sys.path.insert(0, ".")
    gen = getattr(importlib.import_module(mod), fn)
    rows = run(items, gen, a.max_new_tokens)
    summarize(rows)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print(f"\nresultados por item em {a.out}")


if __name__ == "__main__":
    main()
