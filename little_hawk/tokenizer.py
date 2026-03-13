#!/usr/bin/env python3
"""Tokenizer BPE minimal usado pelo Little Hawk."""
import re
import json
from collections import Counter
from typing import Dict, List, Tuple

from .utils import BOLD, CYAN, GREEN, RESET, DIM

CORPUS = """
atenção e memória são os pilares da inteligência
o modelo aprende a focar no que importa
tokens fluem pelo cache como água por uma calhao sink ancora a corrente de informaçãocada palavra carrega contexto do passadoo gradiente descobre o caminho de menor resistênciaa janela desliza mas o fio terra permaneceentropia mede a surpresa de cada símbolocompressão é encontrar padrões no caossilêncio também é informação quando o canal é conhecid o a rede neural transforma texto em pensamentopesos são a memória destilada do treinoatenção causal olha apenas para o passadomemória constante permite geração infinitao ponteiro circular avança sem crescernferência na borda requer eficiência máximalinguagem emerge da composição de padrões simplesbits e tokens são faces da mesma moeda
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
            print(f"\n{DIM}╔{'═'*60}╗{RESET}\n{DIM}║{RESET}{BOLD}{'BPE Tokenizer — Treino':^60}{RESET}{DIM}║{RESET}\n{DIM}╚{'═'*60}╝{RESET}\n")
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
            encoded="".join(b2u[b] for b in word.encode("utf-8"))
            i=0
            while i<len(encoded):
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
        self._hf_tok=None
        try:
            from tokenizers import Tokenizer as HFTok
            import glob
            from pathlib import Path
            with open(meta_path,encoding="utf-8") as _f2:_meta2=json.load(_f2)
            _donor=_meta2.get("donor","HuggingFaceTB/SmolLM-135M").replace("/","--")
            pattern=str(Path.home())+f"/.cache/huggingface/hub/models--{_donor}/**/tokenizer.json"
            hits=glob.glob(pattern,recursive=True)
            if not hits:
                pattern2=str(Path.home())+"/.cache/huggingface/hub/**/tokenizer.json"
                all_hits=glob.glob(pattern2,recursive=True)
                model_name=_donor.split("--")[-1].lower()
                hits=[h for h in all_hits if model_name in h.lower()]
            if hits:
                self._hf_tok=HFTok.from_file(hits[0])
                hf_vocab=self._hf_tok.get_vocab()
                self.id_to_token={int(i):str(t) for t,i in hf_vocab.items()}
                self.vocab=hf_vocab
        except ImportError:
            pass
        with open(meta_path,encoding="utf-8") as f:meta=json.load(f)
        self.bos_id=int(meta.get("bos_id",1));self.eos_id=int(meta.get("eos_id",2))
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

__all__ = ["BPETokenizer", "CORPUS"]
