"""
engine/transformer.py — LlamaLayer para Little Hawk
"""
import math
import numpy as np
from .jit_kernels import _jit_rms_norm, _jit_silu_mul, _rope_numpy

class LlamaLayer:
    def __init__(self,W_q,W_k,W_v,W_o,rms_attn,gate,up,down,rms_ffn,n_heads,d_k,b_q=None,b_k=None,b_v=None):
        self.n_heads=n_heads;self.d_k=d_k;self.d_model=W_q.shape[0]
        # HF Linear: [out,in] → transpomos para x@W (contiguous para BLAS)
        self.W_q=np.ascontiguousarray(W_q.T);self.W_k=np.ascontiguousarray(W_k.T)
        self.W_v=np.ascontiguousarray(W_v.T);self.W_o=np.ascontiguousarray(W_o.T)
        self.rms_attn=rms_attn
        self.gate=np.ascontiguousarray(gate.T);self.up=np.ascontiguousarray(up.T)
        self.down=np.ascontiguousarray(down.T);self.rms_ffn=rms_ffn
        # Bias opcional (Qwen2)
        self.b_q=b_q;self.b_k=b_k;self.b_v=b_v
    @staticmethod
    def _rms_norm(x,w):
        return _jit_rms_norm(x, w)

    def attn_step(self,x_t,k_cache,v_cache,win_ptr,inv_freq,S,W,max_cap,wbi,si,n_ctx,slot_override=None,ctx_override=None):
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
        # Tokens seguintes  → janela circular (S + win_ptr) ou slot externo (Nexus)
        if slot_override is not None:
            slot = int(slot_override)
        elif n_ctx<=S: slot=n_ctx-1
        else:        slot=S+win_ptr
        k_cache[:,:,slot:slot+1,:]=k;v_cache[:,:,slot:slot+1,:]=v
        # ── Contexto: apenas slots preenchidos ─────────────────────────────────
        n_sink=min(n_ctx,S);n_win=max(0,min(n_ctx-S,W))
        # ── Posições StreamingLLM — position freeze ───────────────────────────
        # Fase de enchimento (n_ctx ≤ max_cap): posições reais crescendo
        # Fase estacionária (n_ctx > max_cap):  posições CONGELADAS
        #   sink  → 0 .. S-1          (sempre)
        #   janela → S .. S+W-1       (fixo mesmo com evicções)
        #   Q      → max_cap          (fixo: modelo "acha" que está sempre no fim)
        if ctx_override is not None:
            # Nexus: política fornece os slots vivos reais (sinks + reservoir + recentes).
            # Posições por rank de recência (position freeze): sinks 0..S-1, janela S..S+n-1.
            ctx=np.asarray(ctx_override,dtype=np.int64)
            pos_sink = np.arange(n_sink, dtype=np.int64)
            pos_win  = np.arange(S, S + len(ctx) - n_sink, dtype=np.int64)
            pos_q    = np.array([max_cap-1], dtype=np.int64)
        else:
            if n_win<W: win_ctx=np.arange(S,S+n_win,dtype=np.int64)  # crescendo
            else:       win_ctx=(wbi+win_ptr+1)%W+S                   # estado estacionário
            ctx=np.concatenate([si[:n_sink],win_ctx])
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
        qr=_rope_numpy(q,pos_q,inv_freq);kr=_rope_numpy(kc,pos_ctx,inv_freq)
        sc=(qr@kr.transpose(0,1,3,2))/math.sqrt(self.d_k)
        sc=sc-sc.max(axis=-1,keepdims=True);at=np.exp(sc);at/=at.sum(axis=-1,keepdims=True)
        out=(at@vc).transpose(0,2,1,3).reshape(B,1,self.d_model)@self.W_o
        # Retorna at para políticas de evicção (Nexus); caller pode ignorar o 5º valor
        return out,k_cache,v_cache,float(at[:,:,0,0].mean()*100),at

    def ffn(self,x):
        x_n=self._rms_norm(x,self.rms_ffn)
        return _jit_silu_mul(x_n@self.gate, x_n@self.up)@self.down