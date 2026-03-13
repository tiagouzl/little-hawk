#!/usr/bin/env python3
"""
little_hawk_cli.py  (v2 — multi-layer Atenção + MLP SwiGLU)
══════════════════════════════════════════════════════════════════════════════
Motor: BPE Tokenizer + StreamingKVCache O(1) + RoPE + SwiGLU empilhado

Modo real:
  python little_hawk_transplant.py --layers 4
  python little_hawk_cli.py --weights little_hawk_weights.npz --prompt "..."
Modo demo:
  python little_hawk_cli.py --prompt "atenção e memória"
══════════════════════════════════════════════════════════════════════════════
"""
import re, sys, time, math, json, argparse, os
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np

try:
    import colorama; colorama.init()
except ImportError:
    pass

def _c(code): return f"\033[{code}m"
RESET=_c(0);BOLD=_c(1);DIM=_c(2);CYAN=_c(36);GREEN=_c(32)
YELLOW=_c(33);MAGENTA=_c(35);RED=_c(31);WHITE=_c(97)

# ══════════════════════════════════════════════════════════════════════════════
# BPE TOKENIZER
# ══════════════════════════════════════════════════════════════════════════════
CORPUS = """
atenção e memória são os pilares da inteligência
o modelo aprende a focar no que importa
tokens fluem pelo cache como água por uma calha
o sink ancora a corrente de informação
cada palavra carrega contexto do passado
o gradiente descobre o caminho de menor resistência
a janela desliza mas o fio terra permanece
entropia mede a surpresa de cada símbolo
compressão é encontrar padrões no caos
silêncio também é informação quando o canal é conhecido
a rede neural transforma texto em pensamento
pesos são a memória destilada do treino
atenção causal olha apenas para o passado
memória constante permite geração infinita
o ponteiro circular avança sem crescer
inferência na borda requer eficiência máxima
linguagem emerge da composição de padrões simples
bits e tokens são faces da mesma moeda
""".strip()

class BPETokenizer:
    PAD="<pad>";UNK="<unk>";BOS="<bos>";EOS="<eos>";SPACE="▁"
    def __init__(self):
        self.vocab:Dict[str,int]={};self.id_to_token:Dict[int,str]={}
        self.merges:List[Tuple]=[];self._trained=False
        self._donor_mode=False;self.bos_id=1;self.eos_id=2
    def _pretokenize(self,text):
        return [[self.SPACE]+list(w) for w in re.findall(r"\w+|[^\w\s]",text.lower(),re.UNICODE)]
    @staticmethod
    def _count_pairs(vw):
        p=Counter()
        for ws,f in vw.items():
            s=ws.split()
            for i in range(len(s)-1):p[(s[i],s[i+1])]+=f
        return p
    @staticmethod
    def _merge_pair(pair,vw):
        a,b=pair;bg=re.escape(f"{a} {b}");mg=f"{a}{b}"
        return {re.sub(bg,mg,w):f for w,f in vw.items()}
    def train(self,corpus,vocab_size=512,verbose=True):
        wl=self._pretokenize(corpus);wf=Counter()
        for w in wl:wf[" ".join(w)]+=1
        base=set()
        for w in wf:base.update(w.split())
        sp=[self.PAD,self.UNK,self.BOS,self.EOS];at=sp+sorted(base);vw=dict(wf)
        if verbose:
            _hdr("BPE Tokenizer — Treino")
            print(f"  Corpus: {len(corpus)} chars · {len(wf)} palavras · meta: {vocab_size}\n")
        nm=max(0,vocab_size-len(at));self.merges=[]
        for step in range(nm):
            pairs=self._count_pairs(vw)
            if not pairs:break
            best=pairs.most_common(1)[0][0];vw=self._merge_pair(best,vw)
            self.merges.append(best);at.append(f"{best[0]}{best[1]}")
            if verbose and step%50==0:
                d=step*20//nm;bar="█"*d+"░"*(20-d)
                print(f"\r  [{bar}] {step}/{nm}  {CYAN}{best[0]!r}+{best[1]!r}{RESET}",end="",flush=True)
        if verbose:print(f"\r  [{'█'*20}] {len(self.merges)}/{nm}  {GREEN}✓{RESET} vocab={len(at)}{' '*20}\n")
        self.vocab={t:i for i,t in enumerate(at)};self.id_to_token={i:t for t,i in self.vocab.items()}
        self.bos_id=self.vocab.get(self.BOS,1);self.eos_id=self.vocab.get(self.EOS,2);self._trained=True
    def _tok_word(self,word):
        syms=list(word)
        for a,b in self.merges:
            i,new=0,[]
            while i<len(syms):
                if i<len(syms)-1 and syms[i]==a and syms[i+1]==b:new.append(f"{a}{b}");i+=2
                else:new.append(syms[i]);i+=1
            syms=new
        return syms
    @staticmethod
    def _bytes_to_unicode():
        """Tabela GPT-2: byte → caractere Unicode para byte-level BPE."""
        bs=(list(range(ord("!"),ord("~")+1))+
            list(range(ord("¡"),ord("¬")+1))+
            list(range(ord("®"),ord("ÿ")+1)))
        cs=bs[:]
        n=0
        for b in range(256):
            if b not in bs:bs.append(b);cs.append(256+n);n+=1
        return {b:chr(c) for b,c in zip(bs,cs)}

    def _encode_donor(self,text):
        """
        Encode correto para SmolLM/GPT-2 byte-level BPE:
        1. UTF-8 bytes do texto
        2. Cada byte → caractere Unicode via tabela GPT-2
        3. Espaços → prefixo Ġ  (byte 32 → U+0120)
        4. Greedy longest-match no vocabulário
        """
        b2u=self._bytes_to_unicode()
        unk=self.vocab.get(self.UNK,0)
        ids=[self.bos_id]
        # Marca espaços como Ġ (exatamente como o GPT-2 pretokenizer)
        words=re.findall(r"\s?\S+|\s",text)
        for word in words:
            # Converte cada byte da palavra para o char unicode correspondente
            encoded="".join(b2u[b] for b in word.encode("utf-8"))
            # Greedy match mais longo no vocab
            i=0
            while i<len(encoded):
                # Tenta do maior para o menor
                matched=False
                for end in range(len(encoded),i,-1):
                    tok=encoded[i:end]
                    if tok in self.vocab:
                        ids.append(self.vocab[tok]);i=end;matched=True;break
                if not matched:
                    ids.append(unk);i+=1
        return ids
    def encode(self,text,add_bos=True):
        assert self._trained
        if self._donor_mode:
            if getattr(self,"_hf_tok",None):
                # SmolLM é GPT-2 style: sem BOS explícito
                # <|endoftext|> é separador de documento, não âncora
                return self._hf_tok.encode(text).ids
            return self._encode_donor(text)
        unk=self.vocab.get(self.UNK,0);ids=[self.vocab[self.BOS]] if add_bos else []
        for w in self._pretokenize(text):ids.extend(self.vocab.get(t,unk) for t in self._tok_word(w))
        return ids
    def decode(self,ids):
        if self._donor_mode and getattr(self,"_hf_tok",None):
            return self._hf_tok.decode(ids)
        text="".join(self.id_to_token.get(i,self.UNK) for i in ids)
        for sp in [self.PAD,self.UNK,self.BOS,self.EOS]:text=text.replace(sp,"")
        text=text.replace("Ċ","\n").replace("Ġ"," ").replace(self.SPACE," ")
        return text.strip()
    def load_donor_vocab(self,meta_path):
        # Tenta carregar o tokenizer.json do cache do HuggingFace (tokenizers Rust)
        self._hf_tok=None
        try:
            from tokenizers import Tokenizer as HFTok
            import glob
            from pathlib import Path
            with open(meta_path,encoding="utf-8") as _f2:_meta2=json.load(_f2)
            _donor=_meta2.get("donor","HuggingFaceTB/SmolLM-135M").replace("/","--")
            # Tenta path exato primeiro, depois busca ampla
            pattern=str(Path.home())+f"/.cache/huggingface/hub/models--{_donor}/**/tokenizer.json"
            hits=glob.glob(pattern,recursive=True)
            if not hits:
                # Busca ampla — qualquer tokenizer.json no cache HF
                pattern2=str(Path.home())+"/.cache/huggingface/hub/**/tokenizer.json"
                all_hits=glob.glob(pattern2,recursive=True)
                # Filtra pelo nome do modelo (parte após o --)
                model_name=_donor.split("--")[-1].lower()
                hits=[h for h in all_hits if model_name in h.lower()]
            if hits:
                self._hf_tok=HFTok.from_file(hits[0])
                # Popula id_to_token para o painel de telemetria
                hf_vocab=self._hf_tok.get_vocab()
                self.id_to_token={int(i):str(t) for t,i in hf_vocab.items()}
                self.vocab=hf_vocab
        except ImportError:
            pass
        with open(meta_path,encoding="utf-8") as f:meta=json.load(f)
        self.bos_id=int(meta.get("bos_id",1));self.eos_id=int(meta.get("eos_id",2))
        # Só usa vocab do meta se o HF tokenizer não foi carregado
        if not self._hf_tok:
            raw=meta.get("vocab",{})
            self.vocab={str(t):int(i) for t,i in raw.items()}
            self.id_to_token={int(i):str(t) for t,i in self.vocab.items()}
        self._trained=True;self._donor_mode=True
    def save(self,path):
        with open(path,"w",encoding="utf-8") as f:json.dump({"vocab":self.vocab,"merges":self.merges},f,ensure_ascii=False,indent=2)
    def load(self,path):
        with open(path,encoding="utf-8") as f:d=json.load(f)
        self.vocab=d["vocab"];self.id_to_token={int(i):t for t,i in self.vocab.items()}
        self.merges=[tuple(m) for m in d["merges"]];self.bos_id=self.vocab.get(self.BOS,1)
        self.eos_id=self.vocab.get(self.EOS,2);self._trained=True

# ══════════════════════════════════════════════════════════════════════════════
# LLAMA LAYER — Atenção + MLP SwiGLU
# ══════════════════════════════════════════════════════════════════════════════
class LlamaLayer:
    def __init__(self,W_q,W_k,W_v,W_o,rms_attn,gate,up,down,rms_ffn,n_heads,d_k,b_q=None,b_k=None,b_v=None):
        self.n_heads=n_heads;self.d_k=d_k;self.d_model=W_q.shape[0]
        # HF Linear: [out,in] → transpomos para x@W
        self.W_q=W_q.T;self.W_k=W_k.T;self.W_v=W_v.T;self.W_o=W_o.T
        self.rms_attn=rms_attn
        self.gate=gate.T;self.up=up.T;self.down=down.T;self.rms_ffn=rms_ffn
        # Bias opcional (Qwen2)
        self.b_q=b_q;self.b_k=b_k;self.b_v=b_v
    @staticmethod
    def _rms_norm(x,w):
        return (x/np.sqrt(np.mean(x**2,axis=-1,keepdims=True)+1e-6))*w
    @staticmethod
    def _silu(x):return x/(1.0+np.exp(-x))

    def attn_step(self,x_t,k_cache,v_cache,win_ptr,inv_freq,S,W,max_cap,wbi,si,n_ctx):
        x_n=self._rms_norm(x_t,self.rms_attn);B=1
        _q=x_n@self.W_q;_k=x_n@self.W_k;_v=x_n@self.W_v
        if self.b_q is not None:_q=_q+self.b_q
        if self.b_k is not None:_k=_k+self.b_k
        if self.b_v is not None:_v=_v+self.b_v
        q=_q.reshape(B,1,self.n_heads,self.d_k).transpose(0,2,1,3)
        k=_k.reshape(B,1,self.n_heads,self.d_k).transpose(0,2,1,3)
        v=_v.reshape(B,1,self.n_heads,self.d_k).transpose(0,2,1,3)
        # ── Escrita ────────────────────────────────────────────────────────────
        # Primeiros S tokens → slots sink (0..S-1), imutáveis depois
        # Tokens seguintes  → janela circular (S + win_ptr)
        if n_ctx<=S: slot=n_ctx-1
        else:        slot=S+win_ptr
        k_cache[:,:,slot:slot+1,:]=k;v_cache[:,:,slot:slot+1,:]=v
        # ── Contexto: apenas slots preenchidos ─────────────────────────────────
        n_sink=min(n_ctx,S);n_win=max(0,min(n_ctx-S,W))
        if n_win<W: win_ctx=np.arange(S,S+n_win,dtype=np.int64)  # crescendo
        else:       win_ctx=(wbi+win_ptr+1)%W+S                   # estado estacionário
        ctx=np.concatenate([si[:n_sink],win_ctx])
        # ── Posições StreamingLLM — position freeze ────────────────────────────
        # Fase de enchimento (n_ctx ≤ max_cap): posições reais crescendo
        # Fase estacionária (n_ctx > max_cap):  posições CONGELADAS
        #   sink  → 0 .. S-1          (sempre)
        #   janela → S .. S+W-1       (fixo mesmo com evicções)
        #   Q      → max_cap          (fixo: modelo "acha" que está sempre no fim)
        if n_ctx <= max_cap:
            pos_sink = np.arange(n_sink, dtype=np.int64)
            pos_win  = np.arange(S, S + len(win_ctx), dtype=np.int64)
            # Consulta deve alinhar com o slot recém-escrito (n_ctx-1)
            pos_q    = np.array([n_ctx-1], dtype=np.int64)
        else:
            pos_sink = np.arange(n_sink, dtype=np.int64)
            pos_win  = np.arange(S, S + len(win_ctx), dtype=np.int64)
            # Fase estacionária: Q fica no último índice válido do cache
            pos_q    = np.array([max_cap-1], dtype=np.int64)
        pos_ctx = np.concatenate([pos_sink, pos_win])
        kc=k_cache[:,:,ctx,:];vc=v_cache[:,:,ctx,:]
        def rope(x,pos):
            ang=np.outer(pos.astype(np.float32),inv_freq)[np.newaxis,np.newaxis]
            s,c=np.sin(ang),np.cos(ang);x0,x1=x[...,0::2],x[...,1::2]
            return np.stack([x0*c-x1*s,x0*s+x1*c],axis=-1).reshape(x.shape)
        qr=rope(q,pos_q);kr=rope(kc,pos_ctx)
        sc=(qr@kr.transpose(0,1,3,2))/math.sqrt(self.d_k)
        sc=sc-sc.max(axis=-1,keepdims=True);at=np.exp(sc);at/=at.sum(axis=-1,keepdims=True)
        out=(at@vc).transpose(0,2,1,3).reshape(B,1,self.d_model)@self.W_o
        return out,k_cache,v_cache,float(at[:,:,0,0].mean()*100)

    def ffn(self,x):
        x_n=self._rms_norm(x,self.rms_ffn)
        return (self._silu(x_n@self.gate)*(x_n@self.up))@self.down

# ══════════════════════════════════════════════════════════════════════════════
# MOTOR MULTI-CAMADA
# ══════════════════════════════════════════════════════════════════════════════
class MultiLayerEngine:
    def __init__(self,d_model=128,n_heads=4,n_layers=2,sink_size=4,window_size=28,vocab_size=512,rope_base=10000.0,seed=42):
        self.d_model=d_model;self.n_heads=n_heads;self.d_k=d_model//n_heads
        self.n_layers=n_layers;self.S=sink_size;self.W=window_size
        self.max_cap=sink_size+window_size;self.V=vocab_size;self.bos_id=1;self.eos_id=2
        rng=np.random.default_rng(seed);s=0.02
        self.embed=rng.normal(0,s,(vocab_size,d_model)).astype(np.float32)
        self.W_lm=rng.normal(0,s,(d_model,vocab_size)).astype(np.float32)
        self.norm_w=np.ones(d_model,dtype=np.float32)
        self.layers=[]
        for _ in range(n_layers):
            inter=d_model*4
            self.layers.append(LlamaLayer(
                W_q=rng.normal(0,s,(d_model,d_model)).astype(np.float32),
                W_k=rng.normal(0,s,(d_model,d_model)).astype(np.float32),
                W_v=rng.normal(0,s,(d_model,d_model)).astype(np.float32),
                W_o=rng.normal(0,s,(d_model,d_model)).astype(np.float32),
                rms_attn=np.ones(d_model,np.float32),
                gate=rng.normal(0,s,(inter,d_model)).astype(np.float32),
                up=rng.normal(0,s,(inter,d_model)).astype(np.float32),
                down=rng.normal(0,s,(d_model,inter)).astype(np.float32),
                rms_ffn=np.ones(d_model,np.float32),n_heads=n_heads,d_k=d_model//n_heads))
        self._init_rope(rope_base);self._init_idx()
    def _init_rope(self,base):
        i=np.arange(0,self.d_k,2,dtype=np.float32);self.inv_freq=1.0/(base**(i/self.d_k))
    def _init_idx(self):
        self.wbi=np.arange(self.W,dtype=np.int64);self.si=np.arange(self.S,dtype=np.int64)
    def load_weights(self,path):
        data=np.load(path,allow_pickle=False)
        self.d_model=int(data["_meta_d_model"]);self.n_heads=int(data["_meta_n_heads"])
        self.d_k=self.d_model//self.n_heads;self.n_layers=int(data["_meta_n_layers"])
        self.V=int(data["_meta_vocab_size"]);self.bos_id=int(data["_meta_bos_id"])
        self.eos_id=int(data["_meta_eos_id"]);rope_base=float(data["_meta_rope_base"])
        self.embed=data["embed"].astype(np.float32)
        self.W_lm=data["lm_head"].astype(np.float32).T
        self.norm_w=data["norm_w"].astype(np.float32)
        self.layers=[]
        keys=set(data.keys())
        for i in range(self.n_layers):
            bq=data[f"L{i}_b_q"].astype(np.float32) if f"L{i}_b_q" in keys else None
            bk=data[f"L{i}_b_k"].astype(np.float32) if f"L{i}_b_k" in keys else None
            bv=data[f"L{i}_b_v"].astype(np.float32) if f"L{i}_b_v" in keys else None
            self.layers.append(LlamaLayer(
                W_q=data[f"L{i}_W_q"].astype(np.float32),W_k=data[f"L{i}_W_k"].astype(np.float32),
                W_v=data[f"L{i}_W_v"].astype(np.float32),W_o=data[f"L{i}_W_o"].astype(np.float32),
                rms_attn=data[f"L{i}_rms_attn"].astype(np.float32),
                gate=data[f"L{i}_gate"].astype(np.float32),up=data[f"L{i}_up"].astype(np.float32),
                down=data[f"L{i}_down"].astype(np.float32),rms_ffn=data[f"L{i}_rms_ffn"].astype(np.float32),
                n_heads=self.n_heads,d_k=self.d_k,b_q=bq,b_k=bk,b_v=bv))
        self._init_rope(rope_base);self._init_idx()
        print(f"  {GREEN}✓{RESET} {self.n_layers} camadas  d_model={self.d_model}  "
              f"n_heads={self.n_heads}  d_k={self.d_k}  vocab={self.V:,}")
    def init_cache(self):
        sh=(1,self.n_heads,self.max_cap,self.d_k)
        return [(np.zeros(sh,np.float32),np.zeros(sh,np.float32)) for _ in range(self.n_layers)]
    @staticmethod
    def _rms_norm(x,w):return (x/np.sqrt(np.mean(x**2,axis=-1,keepdims=True)+1e-6))*w
    def step(self,token_id,caches,win_ptr,n_ctx):
        x=self.embed[token_id][np.newaxis,np.newaxis,:]
        sm0=0.0;new_caches=[]
        for li,layer in enumerate(self.layers):
            kc,vc=caches[li]
            ao,kc,vc,sm=layer.attn_step(x,kc,vc,win_ptr,self.inv_freq,
                                          self.S,self.W,self.max_cap,
                                          self.wbi,self.si,n_ctx)
            x=x+ao;x=x+layer.ffn(x);new_caches.append((kc,vc))
            if li==0:sm0=sm
        xn=self._rms_norm(x[:,0,:],self.norm_w)
        # win_ptr só avança quando estamos na fase de janela
        new_win_ptr=(win_ptr+1)%self.W if n_ctx>self.S else win_ptr
        return xn@self.W_lm,new_caches,new_win_ptr,sm0

# ══════════════════════════════════════════════════════════════════════════════
# LOOP DE INFERÊNCIA
# ══════════════════════════════════════════════════════════════════════════════
class LittleHawkInference:
    def __init__(self,tokenizer,engine):
        self.tok=tokenizer;self.engine=engine
        self.S=engine.S;self.W=engine.W;self.max_cap=engine.max_cap
    @staticmethod
    def _softmax(x):e=np.exp(x-x.max());return e/e.sum()
    def _sample(self,logits,temperature=1.0,top_k=40,top_p=0.92,rep_penalty=1.3,generated=None):
        logits=logits.astype(np.float64)
        # Penalidade de repetição: divide logits de tokens já gerados
        if rep_penalty!=1.0 and generated:
            for tid in set(generated[-64:]):   # só últimos 64 tokens
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
        # Inclui tokens do prompt (exceto BOS/EOS) na penalidade de repetição
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

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def _hdr(title):
    w=60;print(f"\n{DIM}╔{'═'*w}╗{RESET}\n{DIM}║{RESET}{BOLD}{title:^{w}}{RESET}{DIM}║{RESET}\n{DIM}╚{'═'*w}╝{RESET}\n")

DEMO_PROMPTS=["atenção e memória","o modelo aprende"]

def build(args):
    tok=BPETokenizer()
    if args.weights:
        meta=args.weights.replace(".npz","_meta.json")
        for p,l in [(args.weights,"weights"),(meta,"_meta.json")]:
            if not os.path.exists(p):
                print(f"  {RED}✗{RESET} Não encontrado: {p}")
                print(f"  Execute: {CYAN}python little_hawk_transplant.py{RESET}");sys.exit(1)
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

    # Sanitize parâmetros básicos para evitar crashes por entrada inválida
    if args.max_tokens<1:
        print(f"{YELLOW}⚠ max_tokens<1, ajustando para 1{RESET}")
        args.max_tokens=1
    if args.temperature<=0:
        print(f"{YELLOW}⚠ temperature<=0, ajustando para 1e-3{RESET}")
        args.temperature=1e-3
    if not (0 < args.top_p <= 1.0):
        print(f"{YELLOW}⚠ top_p fora de (0,1], ajustando para 0.92{RESET}")
        args.top_p=0.92
    if args.rep_penalty<=0:
        print(f"{YELLOW}⚠ rep_penalty<=0, ajustando para 1.0 (desativado){RESET}")
        args.rep_penalty=1.0

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
    # Valida top_k com base no vocabulário carregado
    max_k=len(tok.vocab)
    if args.top_k<1:
        print(f"{YELLOW}⚠ top_k<1, ajustando para 1{RESET}")
        args.top_k=1
    if args.top_k>max_k:
        print(f"{YELLOW}⚠ top_k={args.top_k} > vocab={max_k:,}, ajustando para {max_k}{RESET}")
        args.top_k=max_k

    hawk=LittleHawkInference(tokenizer=tok,engine=engine)
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

if __name__=="__main__":main()
