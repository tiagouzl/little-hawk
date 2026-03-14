#!/usr/bin/env python3
"""
little_hawk_transplant.py  (v2 — multi-layer com MLP SwiGLU)
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
  python little_hawk_transplant.py              # 4 camadas (padrão)
  python little_hawk_transplant.py --layers 8   # 8 camadas
  python little_hawk_transplant.py --inspect
  python little_hawk_transplant.py --validate little_hawk_weights.npz
══════════════════════════════════════════════════════════════════════════════
"""
import sys,os,json,argparse
from pathlib import Path
import numpy as np

RESET="\033[0m";BOLD="\033[1m";DIM="\033[2m"
CYAN="\033[36m";GREEN="\033[32m";YELLOW="\033[33m";RED="\033[31m";WHITE="\033[97m"
def ok(s):  return f"  {GREEN}✓{RESET} {s}"
def err(s): return f"  {RED}✗{RESET} {s}"
def inf(s): return f"  {CYAN}·{RESET} {s}"
def warn(s):return f"  {YELLOW}⚠{RESET} {s}"
def hdr(s): print(f"\n{BOLD}{s}{RESET}\n{DIM}{'─'*58}{RESET}")

MODEL_ID="HuggingFaceTB/SmolLM-135M"
D_MODEL=576;INTERMEDIATE=1536;N_HEADS=9;N_KV_HEADS=3
D_K=D_MODEL//N_HEADS;N_LAYERS=30;VOCAB_SIZE=49152
ROPE_BASE=10000.0;GQA_RATIO=N_HEADS//N_KV_HEADS
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

def extract(st_path,n_layers):
    from safetensors import safe_open
    hdr(f"Extração — {n_layers} camadas (Atenção + MLP SwiGLU)")
    sf=safe_open(st_path,framework="numpy",device="cpu")
    available=list(sf.keys());print(inf(f"{len(available)} tensores no arquivo"))

    embed=sf.get_tensor("model.embed_tokens.weight").astype(np.float32)
    norm_w=sf.get_tensor("model.norm.weight").astype(np.float32)
    if "lm_head.weight" in available:
        lm_head=sf.get_tensor("lm_head.weight").astype(np.float32)
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
        W_q=sf.get_tensor(f"{ap}.q_proj.weight").astype(np.float32)
        W_k=sf.get_tensor(f"{ap}.k_proj.weight").astype(np.float32)
        W_v=sf.get_tensor(f"{ap}.v_proj.weight").astype(np.float32)
        W_o=sf.get_tensor(f"{ap}.o_proj.weight").astype(np.float32)
        rms_attn=sf.get_tensor(f"{ln1}.weight").astype(np.float32)
        orig_k=W_k.shape;W_k=expand_gqa(W_k,N_KV_HEADS,N_HEADS);W_v=expand_gqa(W_v,N_KV_HEADS,N_HEADS)
        gate=sf.get_tensor(f"{mp}.gate_proj.weight").astype(np.float32)
        up=sf.get_tensor(f"{mp}.up_proj.weight").astype(np.float32)
        down=sf.get_tensor(f"{mp}.down_proj.weight").astype(np.float32)
        rms_ffn=sf.get_tensor(f"{ln2}.weight").astype(np.float32)
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
        print(f"  {tag}{CYAN}{k:<22}{RESET} {WHITE}{str(arr.shape):<22}{RESET} {DIM}{arr.nbytes/1024:.0f} KB{RESET}")
    print(f"\n{inf(f'Total: {total/1e6:.1f} MB')}{inf(f'Camadas: {nl}')}")
    all_ok=True
    for i in range(nl):
        for s in ["W_q","W_k","W_v","W_o","rms_attn","gate","up","down","rms_ffn"]:
            k=f"L{i}_{s}"
            if k not in data:print(err(f"Ausente: {k}"));all_ok=False
    if all_ok:print(f"\n{ok(BOLD+'Arquivo íntegro — pronto para o CLI'+RESET)}")
    return all_ok

def main():
    parser=argparse.ArgumentParser(description="Little Hawk Transplant v2")
    parser.add_argument("--layers",type=int,default=DEFAULT_N_LAYERS)
    parser.add_argument("--output",default="little_hawk_weights.npz")
    parser.add_argument("--cache-dir",default=None)
    parser.add_argument("--inspect",action="store_true")
    parser.add_argument("--validate",type=str,default=None)
    args=parser.parse_args()

    if args.validate:validate(args.validate);return

    print(f"""{BOLD}{CYAN}
  ·  ʟɪᴛᴛʟᴇ  ·
  ██╗  ██╗ █████╗ ██╗    ██╗██╗  ██╗
  ██║  ██║██╔══██╗██║    ██║██║ ██╔╝
  ███████║███████║██║ █╗ ██║█████╔╝
  ██╔══██║██╔══██║██║███╗██║██╔═██╗
  ██║  ██║██║  ██║╚███╔███╔╝██║  ██╗
  ╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝
{RESET}{DIM}  Transplante v2 — {args.layers} camadas · Atenção + MLP SwiGLU
  Sem torch. Sem transformers. Sem GPU.{RESET}
""")

    if args.inspect:
        hdr("Arquitetura: SmolLM-135M")
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
