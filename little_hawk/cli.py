#!/usr/bin/env python3
"""CLI do Little Hawk (constrói tokenizer/engine e roda inferência)."""
import argparse
import json
import os
import time
import numpy as np

from .tokenizer import BPETokenizer, CORPUS
from .engine import MultiLayerEngine
from .inference import LittleHawkInference
from .utils import BOLD, CYAN, DIM, GREEN, RED, RESET

def _hdr(title):
    w=60;print(f"\n{DIM}╔{'═'*w}╗{RESET}\n{DIM}║{RESET}{BOLD}{title:^{w}}{RESET}{DIM}║{RESET}\n{DIM}╚{'═'*w}╝{RESET}\n")

DEMO_PROMPTS=["atenção e memória","o modelo aprende"]

def build(args):
    tok=BPETokenizer()
    if args.weights:
        meta=args.weights.replace(".npz","_meta.json")
        for p,_ in [(args.weights,"weights"),(meta,"_meta.json")]:
            if not os.path.exists(p):
                print(f"  {RED}✗{RESET} Não encontrado: {p}")
                print(f"  Execute: {CYAN}python little_hawk_transplant.py{RESET}");raise SystemExit(1)
        tok.load_donor_vocab(meta)
        with open(meta,encoding="utf-8") as _mf:_m=json.load(_mf)
        _donor=_m.get("donor","?");_nl=int(_m.get("n_layers",30))
        _dm=int(_m.get("d_model",576));_nh=int(_m.get("n_heads",9));_vs=int(_m.get("vocab_size",49152))
        _hdr(f"Little Hawk — {_donor}")
        print(f"  {GREEN}✓{RESET} Vocab: {len(tok.vocab):,} tokens\n")
        engine=MultiLayerEngine(d_model=_dm,n_heads=_nh,n_layers=_nl,
                                sink_size=4,window_size=508,vocab_size=_vs)
        print(f"  {DIM}Carregando pesos...{RESET}")
        engine.load_weights(args.weights);print()
    else:
        if args.load_tokenizer and os.path.exists(args.load_tokenizer):
            tok.load(args.load_tokenizer)
        else:
            tok.train(CORPUS,vocab_size=args.vocab_size,verbose=True)
            if args.save_tokenizer:tok.save(args.save_tokenizer)
        _hdr("Little Hawk — Modo Demo  (pesos aleatórios)")
        engine=MultiLayerEngine(d_model=128,n_heads=4,n_layers=2,
                                sink_size=4,window_size=28,vocab_size=len(tok.vocab))
        print(f"  d_model=128  n_heads=4  camadas=2  d_k=32")
        print(f"  {DIM}(pesos aleatórios — pipeline correto, saída é ruído semântico){RESET}\n")
    return tok,engine

def main():
    p=argparse.ArgumentParser(description="Little Hawk CLI v2 — multi-layer Atenção + MLP SwiGLU")
    p.add_argument("--weights",type=str,default=None)
    p.add_argument("--prompt",type=str,default=None)
    p.add_argument("--max-tokens",type=int,default=80)
    p.add_argument("--temperature",type=float,default=0.7)
    p.add_argument("--top-k",type=int,default=40)
    p.add_argument("--top-p",type=float,default=0.92)
    p.add_argument("--rep-penalty",type=float,default=1.15,
                   help="Penalidade de repetição (>=1 = mais antiparaphrase)")
    p.add_argument("--no-panel",action="store_true")
    p.add_argument("--vocab-size",type=int,default=512)
    p.add_argument("--save-tokenizer",type=str,default=None)
    p.add_argument("--load-tokenizer",type=str,default=None)
    args=p.parse_args();np.random.seed(7)

    if args.max_tokens<1:
        print(f"{CYAN}⚠ max_tokens<1, ajustando para 1{RESET}");args.max_tokens=1
    if args.temperature<=0:
        print(f"{CYAN}⚠ temperature<=0, ajustando para 1e-3{RESET}");args.temperature=1e-3
    if not (0 < args.top_p <= 1.0):
        print(f"{CYAN}⚠ top_p fora de (0,1], ajustando para 0.92{RESET}");args.top_p=0.92
    if args.rep_penalty<=0:
        print(f"{CYAN}⚠ rep_penalty<=0, ajustando para 1.0 (desativado){RESET}");args.rep_penalty=1.0

    print(f"""{BOLD}{CYAN}
  ·  ʟɪᴛᴛʟᴇ  ·
  ██╗  ██╗ █████╗ ██╗    ██╗██╗  ██╗
  ██║  ██║██╔══██╗██║    ██║██║ ██╔╝
  ███████║███████║██║ █╗ ██║█████╔╝
  ██╔══██║██╔══██║██║███╗██║██╔═██╗
  ██║  ██║██║  ██║╚███╔███╔╝██║  ██╗
  ╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝
{RESET}{DIM}  H A W K  v2 — Atenção + MLP SwiGLU Empilhados
  StreamingKVCache O(1) · RoPE · multi-layer{RESET}
""")
    tok,engine=build(args)
    hawk=LittleHawkInference(tokenizer=tok,engine=engine)
    max_k=len(tok.vocab)
    if args.top_k<1:
        print(f"{CYAN}⚠ top_k<1, ajustando para 1{RESET}");args.top_k=1
    if args.top_k>max_k:
        print(f"{CYAN}⚠ top_k={args.top_k} > vocab={max_k:,}, ajustando para {max_k}{RESET}");args.top_k=max_k

    prompts=[args.prompt] if args.prompt else DEMO_PROMPTS
    for prompt in prompts:
        hawk.generate(prompt=prompt,max_tokens=args.max_tokens,
                      temperature=args.temperature,top_k=args.top_k,
                      top_p=args.top_p,rep_penalty=args.rep_penalty,
                      stream=True,panel=not args.no_panel)
        time.sleep(0.1)
    if not args.weights:
        print(f"{DIM}Para rodar com pesos reais:{RESET}")
        print(f"  {CYAN}python little_hawk_transplant.py --layers 4{RESET}")
        print(f"  {CYAN}python little_hawk_cli.py --weights little_hawk_weights.npz{RESET}\n")

if __name__=="__main__":
    main()
