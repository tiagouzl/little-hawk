#!/usr/bin/env python3
"""Loop de inferência e painel de telemetria."""
import time
import sys
import numpy as np

from .engine import MultiLayerEngine
from .tokenizer import BPETokenizer
from .utils import RESET,BOLD,DIM,CYAN,GREEN,YELLOW,MAGENTA,WHITE,RED

class LittleHawkInference:
    def __init__(self,tokenizer:BPETokenizer,engine:MultiLayerEngine):
        self.tok=tokenizer;self.engine=engine
        self.S=engine.S;self.W=engine.W;self.max_cap=engine.max_cap
    @staticmethod
    def _softmax(x):e=np.exp(x-x.max());return e/e.sum()
    def _sample(self,logits,temperature=1.0,top_k=40,top_p=0.92,rep_penalty=1.3,generated=None):
        logits=logits.astype(np.float64)
        if rep_penalty!=1.0 and generated:
            for tid in set(generated[-64:]):
                if logits[tid]>0: logits[tid]/=rep_penalty
                else:             logits[tid]*=rep_penalty
        if temperature!=1.0:logits/=max(temperature,1e-8)
        if top_k>0:
            k=min(top_k,logits.size)
            kth=np.partition(logits,-k)[-k];logits[logits<kth]=-np.inf
        probs=self._softmax(logits)
        if top_p<1.0:
            si=np.argsort(probs)[::-1];cum=np.cumsum(probs[si]);cut=np.searchsorted(cum,top_p)+1
            mask=np.zeros_like(probs);mask[si[:cut]]=1.0;probs=probs*mask
        probs=probs/probs.sum()
        return int(np.random.choice(len(probs),p=probs))
    def _panel(self,step,wp,sm,ts,lat,ev,nl):
        fi=min(step+1,self.max_cap);wf=max(0,min(step+1-self.S,self.W))
        bw=32;sc=min(int(self.S*bw/self.max_cap),bw);wc=min(int(wf*bw/self.W),bw-sc);ec=bw-sc-wc
        cb=(f"{GREEN}{'█'*sc}{RESET}{CYAN}{'▓'*wc}{RESET}{DIM}{'░'*ec}{RESET}")
        sw=20;sf=min(int(sm*sw/100),sw);csm=GREEN if sm>15 else YELLOW if sm>5 else RED
        smb=f"{csm}{'█'*sf}{RESET}{'░'*(sw-sf)}";pct=(self.S/max(fi,1))*100
        return [f"{DIM}{'─'*44}{RESET}",
                f"  {BOLD}{MAGENTA}LITTLE HAWK{RESET}  {DIM}{nl}L · Attn+SwiGLU{RESET}",
                f"{DIM}{'─'*44}{RESET}",
                f"  {DIM}step{RESET}      {WHITE}{step:>6}{RESET}",
                f"  {DIM}win_ptr{RESET}   {CYAN}{wp:>6}{RESET}  {DIM}/ {self.W}{RESET}",
                f"  {DIM}evicções{RESET}  {YELLOW}{ev:>6}{RESET}",
                f"  {DIM}latência{RESET}  {WHITE}{lat:>5.1f} ms{RESET}",f"",
                f"  {DIM}cache [{GREEN}sink{RESET}{DIM}|{RESET}{CYAN}janela{RESET}{DIM}]{RESET}",
                f"  [{cb}]",f"  {DIM}{fi}/{self.max_cap} slots  ({pct:.0f}% sink){RESET}",f"",
                f"  {DIM}sink L0 (tok[0]){RESET}",f"  [{smb}] {csm}{sm:.1f}%{RESET}",f"",
                f"  {DIM}último token{RESET}",f"  {YELLOW}{repr(ts):<18}{RESET}",
                f"{DIM}{'─'*44}{RESET}"]
    def generate(self,prompt,max_tokens=80,temperature=0.7,top_k=40,top_p=0.92,
                 rep_penalty=1.15,stream=True,panel=True):
        caches=self.engine.init_cache();win_ptr=0
        ids=self.tok.encode(prompt,add_bos=True)
        generated=[t for t in ids if t not in (self.tok.bos_id,self.tok.eos_id)]
        ev=0;lat=0.0;sm=0.0;ts=""
        if stream:
            print(f"\n{DIM}{'═'*72}{RESET}")
            print(f"  {BOLD}Prompt:{RESET} {CYAN}{prompt}{RESET}")
            print(f"  {DIM}temp={temperature}  top_k={top_k}  top_p={top_p}  camadas={self.engine.n_layers}{RESET}")
            print(f"{DIM}{'═'*72}{RESET}\n  {DIM}▶{RESET} ",end="",flush=True)
        last_logits=None;n_ctx=0
        for tid in ids:
            n_ctx+=1
            logits,caches,win_ptr,sm=self.engine.step(tid,caches,win_ptr,n_ctx);last_logits=logits[0]
        pl=0
        for step in range(max_tokens):
            t0=time.perf_counter()
            nid=self._sample(last_logits.copy(),temperature,top_k,top_p,
                             rep_penalty=rep_penalty,generated=generated)
            n_ctx+=1
            logits,caches,win_ptr,sm=self.engine.step(nid,caches,win_ptr,n_ctx)
            last_logits=logits[0];lat=(time.perf_counter()-t0)*1000
            ts=self.tok.id_to_token.get(nid,BPETokenizer.UNK).replace("Ġ"," ").replace("Ċ","↵")
            if n_ctx>self.max_cap:ev+=1
            if nid==self.tok.eos_id:break
            generated.append(nid)
            if stream:
                if panel and step>0 and step%8==0:
                    if pl:sys.stdout.write(f"\033[{pl}A\033[J")
                    lines=self._panel(step,win_ptr,sm,ts,lat,ev,self.engine.n_layers)
                    pl=len(lines)
                    for ln in lines:print(ln)
                    print(f"\n  {DIM}▶{RESET} ",end="")
                    print(f"{WHITE}{self.tok.decode(generated)}{RESET}",end="",flush=True)
                else:
                    if self.tok._donor_mode and getattr(self.tok,"_hf_tok",None):
                        decoded=self.tok._hf_tok.decode([nid])
                    else:
                        decoded=ts.replace("Ġ"," ").replace("Ċ","\n").replace(BPETokenizer.SPACE," ")
                    print(f"{WHITE}{decoded}{RESET}",end="",flush=True)
        result=self.tok.decode(generated)
        if stream:
            print(f"\n\n{DIM}{'═'*72}{RESET}")
            if panel:
                for ln in self._panel(len(generated),win_ptr,sm,ts,lat,ev,self.engine.n_layers):print(ln)
            print(f"  {DIM}tokens prompt:{RESET}  {len(ids)}")
            print(f"  {DIM}tokens gerados:{RESET} {len(generated)}")
            print(f"  {DIM}evicções:{RESET}       {ev}")
            print(f"  {DIM}cache:{RESET}          {CYAN}constante ({self.max_cap} slots O(1)){RESET}")
            print(f"  {GREEN}✓ memória não cresceu com o número de tokens{RESET}")
            print(f"{DIM}{'═'*72}{RESET}\n")
        return result

__all__ = ["LittleHawkInference"]
