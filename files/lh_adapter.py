"""Adapter Little Hawk para files/chat_baseline.py.

Uso:
    python3 files/chat_baseline.py --prompts files/chat_baseline_prompts.jsonl \
        --adapter files.lh_adapter:generate --max-new-tokens 64 --out res_135m.json

Config via env:
    LH_WEIGHTS=smollm2_135m_instruct_weights.npz (default)
    LH_EVICTION=fifo (default; baseline deve usar fifo)
    LH_MAXCTX=512 (trunca prompt além disso; baseline cabe folgado)

Contrato: generate(messages, max_new_tokens) -> {"text", + sinais}
Sinais de confiança (todos na mesma passada, sem re-rodar):
    mean_logprob, min_logprob, first_logprob, margin_top1_top2 (média)
Métricas: ttft_s, decode_tok_s, peak_rss_mb (VmHWM por item), load_rss_mb.

Template: ChatML do SmolLM2-Instruct. Verificado: <|im_start|>→id 1 (BOS),
<|im_end|>→id 2 (EOS) — ids únicos, sem fragmentação. A geração para no
id de EOS (== <|im_end|>) e no id de BOS (== <|im_start|>, turno fabricado),
SEM emitir o token de parada, para não quebrar max_words/none_of/n_lines.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

_CACHE = {}
_LOAD_RSS_MB = None


def render_chatml(messages):
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def verify_template(tok):
    """Garante ids únicos para os marcadores (chamado uma vez por load)."""
    enc = tok._hf_tok.encode if getattr(tok, "_hf_tok", None) else None
    if enc is None:
        return
    s = enc("<|im_start|>").ids
    e = enc("<|im_end|>").ids
    assert len(s) == 1 and len(e) == 1, f"template fragmentado: {s} {e}"
    assert s[0] == tok.bos_id and e[0] == tok.eos_id, f"ids inesperados: {s} {e}"


def _load():
    global _LOAD_RSS_MB
    key = (os.getenv("LH_WEIGHTS", "smollm2_135m_instruct_weights.npz"), os.getenv("LH_EVICTION", "fifo"))
    if key in _CACHE:
        return _CACHE[key]
    from runtime.tokenizer import BPETokenizer
    from engine import MultiLayerEngine

    weights, eviction = key
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wpath = weights if os.path.isabs(weights) else os.path.join(base, weights)
    mpath = wpath.replace(".npz", "_meta.json")
    import json

    with open(mpath, encoding="utf-8") as f:
        meta = json.load(f)
    tok = BPETokenizer()
    tok.load_donor_vocab(mpath)
    verify_template(tok)
    eng = MultiLayerEngine(
        d_model=int(meta["d_model"]),
        n_heads=int(meta["n_heads"]),
        n_layers=int(meta["n_layers"]),
        sink_size=4,
        window_size=508,
        vocab_size=int(meta["vocab_size"]),
        rope_base=float(meta["rope_base"]),
        eviction=eviction,
    )
    eng.load_weights(wpath)
    _LOAD_RSS_MB = _read_vmhwm_mb()
    _CACHE[key] = (tok, eng)
    return tok, eng


def _reset_peak():
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except Exception:
        return False


def _read_vmhwm_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return None


def generate(messages, max_new_tokens=64):
    tok, eng = _load()
    prompt = render_chatml(messages)
    maxctx = int(os.getenv("LH_MAXCTX", "512"))
    ids = tok.encode(prompt, add_bos=False)
    if len(ids) > maxctx - max_new_tokens:
        ids = ids[-(maxctx - max_new_tokens) :]
    _reset_peak()
    stop_ids = {tok.eos_id, tok.bos_id}  # im_end, im_start (turno fabricado)
    t0 = time.perf_counter()
    caches = eng.init_cache()
    logits, caches, win_ptr, _ = eng.prefill(ids, caches)
    last = logits[0].astype(np.float64)
    ttft = time.perf_counter() - t0
    new_ids = []
    logprobs = []
    margins = []
    t1 = time.perf_counter()
    n_ctx = len(ids)
    for _ in range(max_new_tokens):
        e = np.exp(last - last.max())
        p = e / e.sum()
        order = np.argpartition(p, -2)[-2:]
        top1 = order[np.argmax(p[order])]
        top2 = order[1 - np.argmax(p[order])]
        nid = int(top1)
        if nid in stop_ids:
            last = None  # sinaliza parada limpa (não emite o marcador)
            break
        logprobs.append(float(np.log(max(p[nid], 1e-12))))
        margins.append(float(np.log(max(p[top1], 1e-12)) - np.log(max(p[top2], 1e-12))))
        n_ctx += 1
        out, caches, win_ptr, _ = eng.step(nid, caches, win_ptr, n_ctx)
        last = out[0].astype(np.float64)
        new_ids.append(nid)
    decode_s = time.perf_counter() - t1
    if getattr(tok, "_hf_tok", None) is not None:
        text = tok._hf_tok.decode(new_ids)
    else:
        text = tok.decode(new_ids)
    res = {"text": text.strip()}
    if logprobs:
        res["mean_logprob"] = float(sum(logprobs) / len(logprobs))
        res["min_logprob"] = float(min(logprobs))
        res["first_logprob"] = float(logprobs[0])
        res["margin_top1_top2"] = float(sum(margins) / len(margins))
    res["ttft_s"] = round(ttft, 3)
    if decode_s > 0 and new_ids:
        res["decode_tok_s"] = round(len(new_ids) / decode_s, 2)
    peak = _read_vmhwm_mb()
    if peak is not None:
        res["peak_rss_mb"] = round(peak, 1)
    if _LOAD_RSS_MB is not None:
        res["load_rss_mb"] = round(_LOAD_RSS_MB, 1)
    return res
