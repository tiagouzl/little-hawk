#!/usr/bin/env python3
"""Converte checkpoint RWKV-7 .pth (torch, bf16) para .npz fp32 + _meta.json.

FERRAMENTA OFFLINE — usa torch SOMENTE aqui. O runtime (engine/rwkv_engine.py)
nunca importa torch.

Uso:
    venv/bin/python scripts/convert_rwkv.py \
        --src data/rwkv/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
        --out data/rwkv/rwkv7_g1d_01b_fp32.npz

Registra: SHA256 origem, tamanhos, dtype, nº tensores/parâmetros.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    import torch

    src = Path(a.src)
    print(f"SHA256 origem… ({src.stat().st_size / 1e6:.1f} MB)")
    digest = sha256(src)
    print(f"  {digest}")

    sd = torch.load(str(src), map_location="cpu", weights_only=True)
    print(f"tensores: {len(sd)}")
    params = sum(v.numel() for v in sd.values())
    print(f"parâmetros: {params:,}")

    arrs = {}
    for k, v in sd.items():
        arrs[k] = v.detach().cpu().float().numpy().astype(np.float32, copy=False)
    n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    emb_shape = list(sd["emb.weight"].shape)

    np.savez_compressed(a.out, **arrs)
    out = Path(a.out)
    print(f"convertido: {out} ({out.stat().st_size / 1e6:.1f} MB, fp32)")

    meta = {
        "donor": "BlinkDL/rwkv7-g1",
        "checkpoint": src.name,
        "sha256_src": digest,
        "size_src_MB": round(src.stat().st_size / 1e6, 1),
        "size_npz_MB": round(out.stat().st_size / 1e6, 1),
        "dtype_src": "bfloat16",
        "dtype_npz": "float32",
        "n_tensors": len(sd),
        "n_params": params,
        "arch": "rwkv7-g1 (x070)",
        "n_layer": n_layer,
        "n_embd": emb_shape[1],
        "vocab_size": emb_shape[0],
        "head_size": 64,
        "ctx_train": 8192,
        "tokenizer": "rwkv_vocab_v20230424",
        "license": "Apache-2.0",
    }
    mpath = str(out).replace(".npz", "_meta.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"meta: {mpath}")
    print("OK — runtime não precisa de torch.")


if __name__ == "__main__":
    sys.exit(main())
