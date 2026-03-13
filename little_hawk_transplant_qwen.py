#!/usr/bin/env python3
"""
little_hawk_transplant_qwen.py
══════════════════════════════════════════════════════════════════════════════
Transplanta pesos do Qwen2.5-0.5B para o formato Little Hawk (.npz + _meta.json)

Diferenças arquiteturais vs SmolLM-135M:
  ┌──────────────────────────────────────────────────────────────┐
  │              SmolLM-135M     Qwen2.5-0.5B                    │
  │ d_model         576              896                          │
  │ n_heads           9               14  (Q)                    │
  │ n_kv_heads        3                2  (GQA 7:1)              │
  │ intermediate   1536             4864                          │
  │ n_layers         30               24                          │
  │ vocab_size     49152           151936                         │
  │ rope_base      10000         1_000_000  (YaRN long-ctx)      │
  │ Q/K/V bias       não              SIM  ← diferença crítica   │
  │ lm_head tying    sim              não                         │
  └──────────────────────────────────────────────────────────────┘

Dependências:
  pip install safetensors huggingface_hub numpy

Uso:
  python little_hawk_transplant_qwen.py                  # 24 camadas (todas)
  python little_hawk_transplant_qwen.py --layers 8       # só 8 camadas
  python little_hawk_transplant_qwen.py --inspect        # inspeciona tensores
  python little_hawk_transplant_qwen.py --output qwen_weights.npz
══════════════════════════════════════════════════════════════════════════════
"""
import sys, os, json, argparse
from pathlib import Path
import numpy as np

RESET="\033[0m"; BOLD="\033[1m"; DIM="\033[2m"
CYAN="\033[36m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; WHITE="\033[97m"
def ok(s):   return f"  {GREEN}✓{RESET} {s}"
def err(s):  return f"  {RED}✗{RESET} {s}"
def inf(s):  return f"  {CYAN}·{RESET} {s}"
def warn(s): return f"  {YELLOW}⚠{RESET} {s}"
def hdr(s):  print(f"\n{BOLD}{s}{RESET}\n{DIM}{'─'*60}{RESET}")

# ── Configuração Qwen2.5-0.5B ───────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen2.5-0.5B"
D_MODEL       = 896
INTERMEDIATE  = 4864
N_HEADS       = 14      # heads de Q
N_KV_HEADS    = 2       # heads de KV (GQA 7:1)
D_K           = D_MODEL // N_HEADS   # 64
N_LAYERS      = 24
VOCAB_SIZE    = 151936
ROPE_BASE     = 1_000_000.0          # YaRN extended context
GQA_RATIO     = N_HEADS // N_KV_HEADS  # 7
BOS_ID        = 151643  # <|endoftext|>
EOS_ID        = 151645  # <|im_end|>
HAS_BIAS      = True    # q_proj/k_proj/v_proj têm bias

def download_file(filename, cache_dir=None):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=MODEL_ID, filename=filename, cache_dir=cache_dir)
    mb = Path(path).stat().st_size / 1e6
    print(ok(f"{filename}  ({mb:.0f} MB)"))
    return path

def find_cached(cache_dir=None):
    """Localiza o safetensors no cache local antes de baixar."""
    from huggingface_hub import try_to_load_from_cache
    # Qwen2.5-0.5B pode usar shards ou arquivo único
    for fname in ["model.safetensors", "model-00001-of-00002.safetensors"]:
        try:
            p = try_to_load_from_cache(MODEL_ID, fname, cache_dir=cache_dir)
            if p and Path(p).exists():
                print(ok(f"Cache local: {fname}"))
                return p, fname
        except Exception:
            pass
    return None, None

def expand_gqa(w, n_kv, n_heads):
    """Expande KV heads GQA → MHA por repetição de grupos.
    [n_kv * d_k, d_model] → [n_heads * d_k, d_model]
    """
    ratio  = n_heads // n_kv
    d_k_kv = w.shape[0] // n_kv
    w = w.reshape(n_kv, d_k_kv, w.shape[1])
    w = np.repeat(w, ratio, axis=0)
    return w.reshape(n_heads * d_k_kv, -1)

def load_safetensors(paths):
    """Lê safetensors como bytes raw, converte bf16→f32 sem depender do numpy dtype.
    O safetensors é: [8 bytes header_len][header JSON][dados raw contíguos]
    """
    import json, struct
    tensors = {}
    for path in paths:
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header     = json.loads(f.read(header_len))
            data_start = 8 + header_len
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                dtype  = meta["dtype"]
                shape  = meta["shape"]
                start, end = meta["data_offsets"]
                f.seek(data_start + start)
                raw = f.read(end - start)
                if dtype == "BF16":
                    # bf16 = bits 31..16 do float32; shift left 16 reconstrói f32
                    u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
                    arr = (u16 << 16).view(np.float32)
                elif dtype == "F16":
                    arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
                elif dtype == "F32":
                    arr = np.frombuffer(raw, dtype=np.float32)
                else:
                    continue  # ignora dtypes não numéricos (int8, etc.)
                tensors[name] = arr.reshape(shape).copy()
    return tensors

def expand_gqa_bias(b, n_kv, n_heads):
    """Expande bias de KV [n_kv * d_k] → [n_heads * d_k]"""
    ratio  = n_heads // n_kv
    d_k_kv = b.shape[0] // n_kv
    b = b.reshape(n_kv, d_k_kv)
    b = np.repeat(b, ratio, axis=0)
    return b.reshape(-1)

def inspect(st_path):
    import json, struct
    hdr("Inspecionando tensores do Qwen2.5-0.5B")
    with open(st_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    keys = sorted(k for k in header if k != "__metadata__")
    print(f"  Total: {len(keys)} tensores\n")
    for k in keys[:60]:
        m = header[k]
        print(f"  {DIM}{k:<55}{RESET}  {WHITE}{str(m['shape']):<20}{RESET}  {m['dtype']}")
    if len(keys) > 60:
        print(f"  {DIM}... +{len(keys)-60} tensores{RESET}")

def extract(st_paths, n_layers, output, cache_dir=None):
    hdr(f"Extração — {n_layers} camadas (Qwen2.5-0.5B)")
    data = load_safetensors(st_paths)
    all_keys = set(data.keys())
    print(inf(f"{len(all_keys)} tensores totais ({len(st_paths)} shard(s))"))
    def get_tensor(name):
        if name not in data:
            raise KeyError(f"Tensor não encontrado: {name}")
        return data[name]

    # ── Embeddings e cabeça ──────────────────────────────────────────────────
    embed  = get_tensor("model.embed_tokens.weight")
    norm_w = get_tensor("model.norm.weight")
    # Qwen2.5 tem lm_head separado (sem weight tying)
    if "lm_head.weight" in all_keys:
        lm_head = get_tensor("lm_head.weight")
        print(ok(f"lm_head  {lm_head.shape}"))
    else:
        lm_head = embed.copy()
        print(ok(f"lm_head  {lm_head.shape}  {DIM}(weight tying){RESET}"))
    print(ok(f"embed    {embed.shape}"))
    print(ok(f"norm_w   {norm_w.shape}"))

    pkg = {
        "embed": embed, "lm_head": lm_head, "norm_w": norm_w,
        "_meta_d_model":    np.array(D_MODEL,      dtype=np.int32),
        "_meta_n_heads":    np.array(N_HEADS,       dtype=np.int32),
        "_meta_n_layers":   np.array(n_layers,      dtype=np.int32),
        "_meta_vocab_size": np.array(VOCAB_SIZE,     dtype=np.int32),
        "_meta_bos_id":     np.array(BOS_ID,         dtype=np.int32),
        "_meta_eos_id":     np.array(EOS_ID,         dtype=np.int32),
        "_meta_rope_base":  np.array(ROPE_BASE,      dtype=np.float32),
    }

    # ── Camadas ──────────────────────────────────────────────────────────────
    print()
    for i in range(n_layers):
        pfx = f"model.layers.{i}"

        W_q = get_tensor(f"{pfx}.self_attn.q_proj.weight")  # [14*64, 896]
        W_k = get_tensor(f"{pfx}.self_attn.k_proj.weight")  # [2*64, 896]
        W_v = get_tensor(f"{pfx}.self_attn.v_proj.weight")  # [2*64, 896]
        W_o = get_tensor(f"{pfx}.self_attn.o_proj.weight")  # [896, 14*64]

        # Bias Q (já tem N_HEADS*D_K dimensões — sem expansão)
        b_q = get_tensor(f"{pfx}.self_attn.q_proj.bias")   # [14*64]
        # Bias K e V precisam de expansão GQA
        b_k_raw = get_tensor(f"{pfx}.self_attn.k_proj.bias")  # [2*64]
        b_v_raw = get_tensor(f"{pfx}.self_attn.v_proj.bias")  # [2*64]
        b_k = expand_gqa_bias(b_k_raw, N_KV_HEADS, N_HEADS)
        b_v = expand_gqa_bias(b_v_raw, N_KV_HEADS, N_HEADS)

        # Expande W_k e W_v de GQA → MHA
        W_k = expand_gqa(W_k, N_KV_HEADS, N_HEADS)
        W_v = expand_gqa(W_v, N_KV_HEADS, N_HEADS)

        rms_attn = get_tensor(f"{pfx}.input_layernorm.weight")
        rms_ffn  = get_tensor(f"{pfx}.post_attention_layernorm.weight")

        gate = get_tensor(f"{pfx}.mlp.gate_proj.weight")   # [4864, 896]
        up   = get_tensor(f"{pfx}.mlp.up_proj.weight")     # [4864, 896]
        down = get_tensor(f"{pfx}.mlp.down_proj.weight")   # [896, 4864]

        pkg.update({
            f"L{i}_W_q":     W_q,   f"L{i}_W_k":     W_k,
            f"L{i}_W_v":     W_v,   f"L{i}_W_o":     W_o,
            f"L{i}_b_q":     b_q,   f"L{i}_b_k":     b_k,
            f"L{i}_b_v":     b_v,
            f"L{i}_rms_attn":rms_attn, f"L{i}_rms_ffn": rms_ffn,
            f"L{i}_gate":    gate,  f"L{i}_up":      up,
            f"L{i}_down":    down,
        })
        print(ok(f"Camada {i:2d}  "
                 f"W_q{list(W_q.shape)}  W_k{list(W_k.shape)}  "
                 f"b_q{list(b_q.shape)}  b_k{list(b_k.shape)}"))

    # ── Salva .npz ───────────────────────────────────────────────────────────
    hdr(f"Salvando {output}")
    np.savez_compressed(output, **pkg)
    mb = Path(output).stat().st_size / 1e6
    print(ok(f"{output}  ({mb:.0f} MB)"))

    # ── Meta JSON ─────────────────────────────────────────────────────────────
    meta_path = output.replace(".npz", "_meta.json")
    meta = {
        "donor":       MODEL_ID,
        "d_model":     D_MODEL,
        "n_heads":     N_HEADS,
        "n_kv_heads":  N_KV_HEADS,
        "d_k":         D_K,
        "intermediate":INTERMEDIATE,
        "n_layers":    n_layers,
        "vocab_size":  VOCAB_SIZE,
        "rope_base":   ROPE_BASE,
        "bos_id":      BOS_ID,
        "eos_id":      EOS_ID,
        "has_bias":    HAS_BIAS,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(ok(f"Meta: {meta_path}"))
    print(f"\n  {DIM}Para rodar:{RESET}")
    print(f"  {CYAN}python little_hawk_cli.py --weights {output}{RESET}\n")

def validate(npz_path):
    hdr(f"Validando {npz_path}")
    data = np.load(npz_path, allow_pickle=False)
    nl   = int(data["_meta_n_layers"])
    dm   = int(data["_meta_d_model"])
    nh   = int(data["_meta_n_heads"])
    print(ok(f"n_layers={nl}  d_model={dm}  n_heads={nh}"))
    errs = 0
    for i in range(nl):
        for key in [f"L{i}_W_q",f"L{i}_W_k",f"L{i}_W_v",f"L{i}_W_o",
                    f"L{i}_b_q",f"L{i}_b_k",f"L{i}_b_v",
                    f"L{i}_rms_attn",f"L{i}_rms_ffn",
                    f"L{i}_gate",f"L{i}_up",f"L{i}_down"]:
            if key not in data:
                print(err(f"Faltando: {key}")); errs += 1
    if errs == 0:
        print(ok(f"Todas as chaves presentes — {nl} camadas completas"))
    else:
        print(err(f"{errs} chaves faltando"))

def main():
    p = argparse.ArgumentParser(description="Little Hawk Transplant — Qwen2.5-0.5B")
    p.add_argument("--layers",    type=int,  default=N_LAYERS,
                   help=f"Número de camadas a extrair (padrão: {N_LAYERS})")
    p.add_argument("--output",    type=str,  default="qwen_weights.npz")
    p.add_argument("--cache-dir", type=str,  default=None)
    p.add_argument("--inspect",   action="store_true")
    p.add_argument("--validate",  type=str,  default=None)
    args = p.parse_args()

    if args.validate:
        validate(args.validate); return

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"""
{BOLD}  Little Hawk — Transplante Qwen2.5-0.5B{RESET}
{DIM}  ┌─────────────────────────────────────────────────────┐
  │ d_model={D_MODEL}  n_heads={N_HEADS}(Q)/{N_KV_HEADS}(KV)  GQA {GQA_RATIO}:1        │
  │ intermediate={INTERMEDIATE}  rope_base={ROPE_BASE:.0e}         │
  │ vocab={VOCAB_SIZE:,}  n_layers={N_LAYERS}  bias=sim             │
  └─────────────────────────────────────────────────────┘{RESET}
""")

    # ── Verifica cache ────────────────────────────────────────────────────────
    hdr("Localizando modelo")
    cached_path, cached_name = find_cached(args.cache_dir)

    st_paths = []
    if cached_path:
        # Pode ser modelo shardado — procura todos os shards no mesmo diretório
        parent = Path(cached_path).parent
        shards = sorted(parent.glob("model*.safetensors"))
        if shards:
            st_paths = [str(s) for s in shards]
            print(ok(f"{len(st_paths)} shard(s) encontrado(s) em cache"))
        else:
            st_paths = [cached_path]
    else:
        print(inf(f"Baixando {MODEL_ID}..."))
        print(warn("Qwen2.5-0.5B ≈ 1 GB — pode levar alguns minutos na primeira vez"))
        # Tenta arquivo único primeiro, depois shards
        try:
            path = download_file("model.safetensors", args.cache_dir)
            st_paths = [path]
        except Exception:
            # Modelo shardado
            shard1 = download_file("model-00001-of-00002.safetensors", args.cache_dir)
            shard2 = download_file("model-00002-of-00002.safetensors", args.cache_dir)
            st_paths = [shard1, shard2]

    if args.inspect:
        inspect(st_paths[0]); return

    n = min(args.layers, N_LAYERS)
    if args.layers > N_LAYERS:
        print(warn(f"--layers {args.layers} > máximo {N_LAYERS}, usando {N_LAYERS}"))

    extract(st_paths, n, args.output, args.cache_dir)

if __name__ == "__main__":
    main()
