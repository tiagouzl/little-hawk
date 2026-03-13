#!/usr/bin/env python3
"""Camadas LLaMA + motor multi-camada."""
import math
import numpy as np

class LlamaLayer:
    def __init__(self,W_q,W_k,W_v,W_o,rms_attn,gate,up,down,rms_ffn,n_heads,d_k,b_q=None,b_k=None,b_v=None):
        self.n_heads=n_heads;self.d_k=d_k;self.d_model=W_q.shape[0]
        self.W_q=W_q.T;self.W_k=W_k.T;self.W_v=W_v.T;self.W_o=W_o.T
        self.rms_attn=rms_attn
        self.gate=gate.T;self.up=up.T;self.down=down.T;self.rms_ffn=rms_ffn
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
        if n_ctx<=S: slot=n_ctx-1
        else:        slot=S+win_ptr
        k_cache[:,:,slot:slot+1,:]=k;v_cache[:,:,slot:slot+1,:]=v
        n_sink=min(n_ctx,S);n_win=max(0,min(n_ctx-S,W))
        if n_win<W: win_ctx=np.arange(S,S+n_win,dtype=np.int64)
        else:       win_ctx=(wbi+win_ptr+1)%W+S
        ctx=np.concatenate([si[:n_sink],win_ctx])
        if n_ctx <= max_cap:
            pos_sink = np.arange(n_sink, dtype=np.int64)
            pos_win  = np.arange(S, S + len(win_ctx), dtype=np.int64)
            pos_q    = np.array([n_ctx-1], dtype=np.int64)
        else:
            pos_sink = np.arange(n_sink, dtype=np.int64)
            pos_win  = np.arange(S, S + len(win_ctx), dtype=np.int64)
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
        self.layers=[];keys=set(data.keys())
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
        new_win_ptr=(win_ptr+1)%self.W if n_ctx>self.S else win_ptr
        return xn@self.W_lm,new_caches,new_win_ptr,sm0

__all__ = ["LlamaLayer", "MultiLayerEngine"]
