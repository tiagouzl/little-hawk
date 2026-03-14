#!/usr/bin/env python3
"""
engine/transformer.py — LlamaLayer para Little Hawk
"""
import math
import numpy as np

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
        # ── Posições StreamingLLM — position freeze ───────────────────────────
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