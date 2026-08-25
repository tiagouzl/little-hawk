#!/usr/bin/env python3
"""
transplants/smollm.py  (v2 — multi-layer com MLP SwiGLU)
══════════════════════════════════════════════════════════════════════════════
Extrai N camadas completas do SmolLM-135M:
  Atenção (Q,K,V,O,RMSNorm) + MLP SwiGLU (gate,up,down,RMSNorm)

O .npz armazena cada camada separadamente:
  L{i}_W_q  L{i}_W_k  L{i}_W_v  L{i}_W_o  L{i}_rms_attn
  L{i}_gate  L{i}_up  L{i}_down  L{i}_rms_ffn

O CLI empilha as camadas em forward pass completo.

Dependências:
  pip install safetensors huggingface_hub numpy

Uso:
  python -m transplants.smollm              # 4 camadas (padrão)
  python little_hawk_transplant.py --layers 8   # shim de compatibilidade
  python little_hawk_transplant.py --inspect
  python little_hawk_transplant.py --validate little_hawk_weights.npz
══════════════════════════════════════════════════════════════════════════════
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

RESET="\033[0m";BOLD="\033[1m";DIM="\033[2m"
CYAN="\033[36m";GREEN="\033[32m";YELLOW="\033[33m";RED="\033[31m";WHITE="\033[97m"
def ok(s):  return f"  {GREEN}✓{RESET} {s}"
def err(s): return f"  {RED}✗{RESET} {s}"
def inf(s): return f"  {CYAN}·{RESET} {s}"
def warn(s):return f"  {YELLOW}⚠{RESET} {s}"
def hdr(s): print(f"\n{BOLD}{s}{RESET}\n{DIM}{'─'*58}{RESET}")

MODEL_CONFIGS={
    "smollm-135m":  {"id":"HuggingFaceTB/SmolLM-135M",  "d_model":576, "inter":1536, "n_heads":9,  "n_kv":3,  "n_layers":30, "vocab":49152, "rope":10000.0},
    "smollm2-135m": {"id":"HuggingFaceTB/SmolLM2-135M", "d_model":576, "inter":1536, "n_heads":9,  "n_kv":3,  "n_layers":30, "vocab":49152, "rope":100000.0},
    "smollm2-360m": {"id":"HuggingFaceTB/SmolLM2-360M", "d_model":960, "inter":2560, "n_heads":15, "n_kv":5,  "n_layers":32, "vocab":49152, "rope":100000.0},
    "smollm2-1.7b": {"id":"HuggingFaceTB/SmolLM2-1.7B", "d_model":2048,"inter":8192, "n_heads":32, "n_kv":32, "n_layers":24, "vocab":49152, "rope":130000.0},
}
DEFAULT_MODEL="smollm-135m"
# Compatibilidade: expõe constantes do modelo padrão para importadores antigos
MODEL_ID=MODEL_CONFIGS[DEFAULT_MODEL]["id"]
D_MODEL=MODEL_CONFIGS[DEFAULT_MODEL]["d_model"];INTERMEDIATE=MODEL_CONFIGS[DEFAULT_MODEL]["inter"]
N_HEADS=MODEL_CONFIGS[DEFAULT_MODEL]["n_heads"];N_KV_HEADS=MODEL_CONFIGS[DEFAULT_MODEL]["n_kv"]
D_K=D_MODEL//N_HEADS;N_LAYERS=MODEL_CONFIGS[DEFAULT_MODEL]["n_layers"];VOCAB_SIZE=MODEL_CONFIGS[DEFAULT_MODEL]["vocab"]
ROPE_BASE=MODEL_CONFIGS[DEFAULT_MODEL]["rope"];GQA_RATIO=N_HEADS//N_KV_HEADS
BOS_ID=1;EOS_ID=2;DEFAULT_N_LAYERS=4

def download_file(filename,cache_dir=None):
    from huggingface_hub import hf_hub_download
    path=hf_hub_download(repo_id=MODEL_ID,filename=filename,cache_dir=cache_dir)
    mb=Path(path).stat().st_size/1e6;print(ok(f"{filename}  ({mb:.0f} MB)"))
    return path

def download_vocab():
    from huggingface_hub import hf_hub_download
    print(inf("Baixando tokenizer.json..."))
    path=hf_hub_download(repo_id=MODEL_ID,filename="tokenizer.json")
    with open(path,encoding="utf-8") as f:data=json.load(f)
    vocab={}
    if "model" in data and "vocab" in data["model"]:vocab=dict(data["model"]["vocab"])
    for e in data.get("added_tokens",[]):vocab[e["content"]]=e["id"]
    print(ok(f"Vocabulário: {len(vocab):,} tokens"))
    return vocab

def expand_gqa(w,n_kv,n_heads):
    ratio=n_heads//n_kv;d_k_kv=w.shape[0]//n_kv
    w=w.reshape(n_kv,d_k_kv,w.shape[1]);w=np.repeat(w,ratio,axis=0)
    return w.reshape(n_heads*d_k_kv,-1)

def load_safetensors(path):
    """Lê safetensors como bytes raw, converte BF16/F16→F32 sem depender de dtype numpy.
    Suporta BF16 (bits 31..16 do float32).
    """
    import json as _json
    import struct
    tensors={}
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = _json.loads(f.read(header_len))
        data_start = 8 + header_len
        for name, meta in header.items():
            if name == "__metadata__": continue
            dtype = meta["dtype"]; shape = meta["shape"]
            start, end = meta["data_offsets"]
            f.seek(data_start + start)
            raw = f.read(end - start)
            if dtype == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
                arr = (u16 << 16).view(np.float32)
            elif dtype == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            elif dtype == "F32":
                arr = np.frombuffer(raw, dtype=np.float32)
            else:
                continue
            tensors[name] = arr.reshape(shape).copy()
    return tensors

def extract(st_path,n_layers):
    hdr(f"Extração — {n_layers} camadas (Atenção + MLP SwiGLU)")
    data = load_safetensors(st_path)
    available=set(data.keys());print(inf(f"{len(available)} tensores no arquivo"))
    def get_tensor(name):
        if name not in data:
            raise KeyError(f"Tensor não encontrado: {name}")
        return data[name]
    embed=get_tensor("model.embed_tokens.weight")
    norm_w=get_tensor("model.norm.weight")
    if "lm_head.weight" in available:
        lm_head=get_tensor("lm_head.weight")
        print(ok(f"lm_head  {lm_head.shape}"))
    else:
        lm_head=embed.copy();print(ok(f"lm_head  {lm_head.shape}  {DIM}(weight tying){RESET}"))
    print(ok(f"embed    {embed.shape}"));print(ok(f"norm_w   {norm_w.shape}"))

    pkg={
        "embed":embed,"lm_head":lm_head,"norm_w":norm_w,
        "_meta_d_model":np.float32(D_MODEL),"_meta_n_heads":np.float32(N_HEADS),
        "_meta_intermediate":np.float32(INTERMEDIATE),"_meta_n_layers":np.float32(n_layers),
        "_meta_vocab_size":np.float32(VOCAB_SIZE),"_meta_rope_base":np.float32(ROPE_BASE),
        "_meta_bos_id":np.float32(BOS_ID),"_meta_eos_id":np.float32(EOS_ID),
    }

    hdr(f"Camadas 0 … {n_layers-1}")
    for i in range(n_layers):
        ap=f"model.layers.{i}.self_attn";mp=f"model.layers.{i}.mlp"
        ln1=f"model.layers.{i}.input_layernorm";ln2=f"model.layers.{i}.post_attention_layernorm"
        W_q=get_tensor(f"{ap}.q_proj.weight")
        W_k=get_tensor(f"{ap}.k_proj.weight")
        W_v=get_tensor(f"{ap}.v_proj.weight")
        W_o=get_tensor(f"{ap}.o_proj.weight")
        rms_attn=get_tensor(f"{ln1}.weight")
        orig_k=W_k.shape;W_k=expand_gqa(W_k,N_KV_HEADS,N_HEADS);W_v=expand_gqa(W_v,N_KV_HEADS,N_HEADS)
        gate=get_tensor(f"{mp}.gate_proj.weight")
        up=get_tensor(f"{mp}.up_proj.weight")
        down=get_tensor(f"{mp}.down_proj.weight")
        rms_ffn=get_tensor(f"{ln2}.weight")
        pkg.update({f"L{i}_W_q":W_q,f"L{i}_W_k":W_k,f"L{i}_W_v":W_v,f"L{i}_W_o":W_o,
                    f"L{i}_rms_attn":rms_attn,f"L{i}_gate":gate,f"L{i}_up":up,
                    f"L{i}_down":down,f"L{i}_rms_ffn":rms_ffn})
        ap_=(W_q.size+W_k.size+W_v.size+W_o.size)/1e6;mp_=(gate.size+up.size+down.size)/1e6
        print(ok(f"L{i}  attn={ap_:.2f}M  mlp={mp_:.2f}M  GQA {orig_k}→{W_k.shape}"))
    return pkg

def validate(path):
    hdr(f"Validação: {path}")
    data=np.load(path,allow_pickle=False);keys=sorted(data.keys());total=0
    nl=int(data.get("_meta_n_layers",1))
    for k in keys:
        arr=data[k];total+=arr.nbytes
        tag=ok("") if (k in ["embed","lm_head","norm_w"] or k.startswith("L")) else inf("")
        print(f"  {tag}{CYAN}{k:<22}{RESET} {WHITE}{arr.shape!s:<22}{RESET} {DIM}{arr.nbytes/1024:.0f} KB{RESET}")
    print(f"\n{inf(f'Total: {total/1e6:.1f} MB')}{inf(f'Camadas: {nl}')}")
    all_ok=True
    for i in range(nl):
        for s in ["W_q","W_k","W_v","W_o","rms_attn","gate","up","down","rms_ffn"]:
            k=f"L{i}_{s}"
            if k not in data:print(err(f"Ausente: {k}"));all_ok=False
    if all_ok:print(f"\n{ok(BOLD+'Arquivo íntegro — pronto para o CLI'+RESET)}")
    return all_ok

def main():
    parser=argparse.ArgumentParser(description="Little Hawk Transplant v2 — SmolLM/SmolLM2")
    parser.add_argument("--model",type=str,default=DEFAULT_MODEL, choices=list(MODEL_CONFIGS.keys()),
                        help="Modelo HuggingFace (smollm-135m, smollm2-135m/360m/1.7b)")
    parser.add_argument("--layers",type=int,default=DEFAULT_N_LAYERS)
    parser.add_argument("--output",default="little_hawk_weights.npz")
    parser.add_argument("--cache-dir",default=None)
    parser.add_argument("--inspect",action="store_true")
    parser.add_argument("--validate",type=str,default=None)
    args=parser.parse_args()

    # Configura globais para o modelo escolhido (mantém compatibilidade com funções legadas)
    global MODEL_ID, D_MODEL, INTERMEDIATE, N_HEADS, N_KV_HEADS, D_K, N_LAYERS, VOCAB_SIZE, ROPE_BASE, GQA_RATIO
    cfg = MODEL_CONFIGS[args.model]
    MODEL_ID = cfg["id"]; D_MODEL = cfg["d_model"]; INTERMEDIATE = cfg["inter"]
    N_HEADS = cfg["n_heads"]; N_KV_HEADS = cfg["n_kv"]; N_LAYERS = cfg["n_layers"]
    VOCAB_SIZE = cfg["vocab"]; ROPE_BASE = cfg["rope"]
    D_K = D_MODEL // N_HEADS; GQA_RATIO = N_HEADS // N_KV_HEADS

    if args.validate:validate(args.validate);return

    print(f"""{BOLD}{CYAN}
  ·  ʟɪᴛᴛʟᴇ  ·
  ██╗  ██╗ █████╗ ██╗    ██╗██╗  ██╗
  ██║  ██║██╔══██╗██║    ██║██║ ██╔╝
  ███████║███████║██║ █╗ ██║█████╔╝
  ██╔══██║██╔══██║██║███╗██║██╔═██╗
  ██║  ██║██║  ██║╚███╔███╔╝██║  ██╗
  ╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝
{RESET}{DIM}  Transplante v2 — {args.model} · {args.layers} camadas · Atenção + MLP SwiGLU
  Sem torch. Sem transformers. Sem GPU.{RESET}
""")

    if args.inspect:
        hdr(f"Arquitetura: {args.model} ({MODEL_ID})")
        for l,v in [("d_model",str(D_MODEL)),("intermediate",f"{INTERMEDIATE} (MLP)"),
                    ("n_heads (Q)",str(N_HEADS)),("n_kv_heads",f"{N_KV_HEADS} (GQA {GQA_RATIO}:1)"),
                    ("d_k",str(D_K)),("rope_base",str(ROPE_BASE)),("vocab",f"{VOCAB_SIZE:,}")]:
            print(f"  {DIM}{l:<16}{RESET} {WHITE}{v}{RESET}")
        attn=4*D_MODEL*D_MODEL;mlp=2*D_MODEL*INTERMEDIATE+INTERMEDIATE*D_MODEL
        print()
        for n in [1,2,4,8,12]:
            mb=(n*(attn+mlp)*4)/1e6;print(inf(f"{n:>2} camadas → ~{mb:.0f} MB no .npz"))
        return

    hdr("Download");print(warn("Usa cache local se já baixado\n"))
    try:st_path=download_file("model.safetensors",args.cache_dir)
    except Exception as e:print(err(f"Falhou: {e}"));sys.exit(1)

    pkg=extract(st_path,args.layers)

    hdr("Vocabulário")
    try:vocab=download_vocab()
    except Exception as e:
        print(warn(f"tokenizer.json falhou — vocab mínimo"))
        vocab={"<|endoftext|>":0,"<|im_start|>":1,"<|im_end|>":2}

    hdr("Salvando")
    np.savez_compressed(args.output,**pkg)
    mb=Path(args.output).stat().st_size/1e6;print(ok(f"{args.output}  ({mb:.1f} MB)"))
    meta=args.output.replace(".npz","_meta.json")
    with open(meta,"w",encoding="utf-8") as f:
        json.dump({"donor":MODEL_ID,"n_layers":args.layers,"d_model":D_MODEL,
                   "intermediate":INTERMEDIATE,"n_heads":N_HEADS,"d_k":D_K,
                   "vocab_size":VOCAB_SIZE,"rope_base":ROPE_BASE,
                   "bos_id":BOS_ID,"eos_id":EOS_ID,"vocab":vocab},f,ensure_ascii=False,indent=2)
    print(ok(meta));validate(args.output)
    print(f"\n{BOLD}{'═'*58}{RESET}")
    print(f"  {CYAN}python little_hawk_cli.py --weights {args.output}{RESET}\n")

if __name__=="__main__":main()
