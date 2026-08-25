"""
engine/engine.py — MultiLayerEngine para Little Hawk
"""
import math
import numpy as np
from .transformer import LlamaLayer
from .jit_kernels import _rope_numpy as _rope

# Cores para output (temporário, até mover para utils)


class MultiLayerEngine:
    def __init__(self,d_model=128,n_heads=4,n_layers=2,sink_size=4,window_size=28,vocab_size=512,rope_base=10000.0,seed=42):
        self.d_model=d_model;self.n_heads=n_heads;self.d_k=d_model//n_heads
        self.n_layers=n_layers;self.S=sink_size;self.W=window_size
        self.max_cap=sink_size+window_size;self.V=vocab_size;self.bos_id=1;self.eos_id=2
        rng=np.random.default_rng(seed);s=0.02
        self.embed=rng.normal(0,s,(vocab_size,d_model)).astype(np.float32)
        self.W_lm=rng.normal(0,s,(d_model,vocab_size)).astype(np.float32)
        self.W_lm_t=np.ascontiguousarray(self.W_lm.T)  # [V,d] p/ sgemv rápido no step
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
        lm=data["lm_head"].astype(np.float32)   # [V,d] como salvo pelo transplant
        self.W_lm=lm.T                          # [d,V] mantido p/ compatibilidade
        self.W_lm_t=np.ascontiguousarray(lm)    # já contíguo do npz — sgemv rápido
        self.norm_w=data["norm_w"].astype(np.float32)
        # ── Validação de integridade: chaves presentes + shapes coerentes ────
        expected={"embed":(self.V,self.d_model),"lm_head":(self.V,self.d_model),
                  "norm_w":(self.d_model,)}
        for i in range(self.n_layers):
            expected.update({f"L{i}_W_{n}":(self.d_model,self.d_model) for n in "qkvo"})
            expected.update({f"L{i}_rms_{n}":(self.d_model,) for n in ("attn","ffn")})
            expected.update({f"L{i}_{n}":None for n in ("gate","up","down")})
            for n in ("b_q","b_k","b_v"):
                if f"L{i}_{n}" in data:expected[f"L{i}_{n}"]=(self.d_model,)
        errs=[]
        for key,shape in expected.items():
            if key not in data:errs.append(f"ausente: {key}");continue
            if shape is None:continue
            got=tuple(data[key].shape)
            if got!=shape:errs.append(f"{key}: shape {got} ≠ esperado {shape}")
        inter=None
        for i in range(self.n_layers):
            g=tuple(data[f"L{i}_gate"].shape);d=tuple(data[f"L{i}_down"].shape)
            if g[1]!=self.d_model or d[0]!=self.d_model or g[0]!=d[1]:
                errs.append(f"L{i}: gate{g}/down{d} incoerentes com d_model={self.d_model}")
            if inter is None:inter=g[0]
            elif g[0]!=inter:errs.append(f"L{i}: intermediate {g[0]} ≠ {inter} (L0)")
        if errs:
            raise ValueError(f"Pesos inválidos em '{path}':\n  "+"\n  ".join(errs[:12])
                             +(f"\n  … +{len(errs)-12} erros" if len(errs)>12 else ""))
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
          # Removido print direto para manter núcleo limpo
    def init_cache(self):
        sh=(1,self.n_heads,self.max_cap,self.d_k)
        return [(np.zeros(sh,np.float32),np.zeros(sh,np.float32)) for _ in range(self.n_layers)]
    _rms_norm = staticmethod(LlamaLayer._rms_norm)
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
        # lm_head via [V,d] contígua: sgemv orientado a linhas (~8ms vs ~14ms)
        logits=(self.W_lm_t @ xn[0].reshape(-1,1)).T
        # win_ptr só avança quando estamos na fase de janela
        new_win_ptr=(win_ptr+1)%self.W if n_ctx>self.S else win_ptr
        return logits,new_caches,new_win_ptr,sm0
    def prefill(self,tokens,caches=None):
        """Forward batched do prompt.

        T ≤ max_cap: um único forward GEMM com máscara causal (TTFT 10-20× menor).
        T > max_cap: chunked — primeiro max_cap batched (fill) + restante sequencial
        via step() com cache circular/position freeze. Estado final bit-a-bit
        equivalente ao loop de steps. Retorna (logits_do_último_token, caches, win_ptr, sm0).
        """
        ids=np.asarray(tokens,dtype=np.int64);T=int(ids.size)
        if T==0:
            caches=caches or self.init_cache()
            return np.zeros((1,self.V),np.float32),caches,0,0.0
        # Chunked para prompts maiores que a janela
        if T>self.max_cap:
            # Primeiro chunk batched (fill)
            logits,caches,win_ptr,sm=self.prefill(ids[:self.max_cap],caches)
            n_ctx=self.max_cap
            for tid in ids[self.max_cap:]:
                n_ctx+=1
                logits,caches,win_ptr,sm=self.step(int(tid),caches,win_ptr,n_ctx)
            return logits,caches,win_ptr,sm
        caches=caches or self.init_cache()
        x=self.embed[ids][np.newaxis]                      # [1,T,d]
        sm0=0.0;new_caches=[]
        pos=np.arange(T,dtype=np.int64)
        causal=np.tril(np.ones((T,T),dtype=bool))
        for li,layer in enumerate(self.layers):
            kc,vc=caches[li]
            x_n=self._rms_norm(x,layer.rms_attn)
            q=(x_n@layer.W_q).reshape(1,T,self.n_heads,self.d_k).transpose(0,2,1,3)
            k=(x_n@layer.W_k).reshape(1,T,self.n_heads,self.d_k).transpose(0,2,1,3)
            v=(x_n@layer.W_v).reshape(1,T,self.n_heads,self.d_k).transpose(0,2,1,3)
            if layer.b_q is not None:
                q=q+layer.b_q;k=k+layer.b_k;v=v+layer.b_v
            # fase fill: slots 0..T-1 (sequenciais), imutáveis adiante
            kc[0,:,:T,:]=k[0];vc[0,:,:T,:]=v[0]
            qr=_rope(q,pos,self.inv_freq);kr=_rope(kc[:,:,:T,:],pos,self.inv_freq)
            sc=(qr@kr.transpose(0,1,3,2))/math.sqrt(self.d_k)
            sc=np.where(causal,sc,np.float32(-np.inf))
            sc=sc-sc.max(axis=-1,keepdims=True);at=np.exp(sc);at/=at.sum(axis=-1,keepdims=True)
            out=(at@v).transpose(0,2,1,3).reshape(1,T,self.d_model)@layer.W_o
            x=x+out;x=x+layer.ffn(x)
            if li==0:sm0=float(at[:,:,0,:].mean()*100)
            new_caches.append((kc,vc))
        xn=self._rms_norm(x[0,-1],self.norm_w)
        logits=(self.W_lm_t@xn.reshape(-1,1)).T
        new_win_ptr=(T-self.S)%self.W if T>self.S else 0
        return logits,new_caches,new_win_ptr,sm0