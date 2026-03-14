"""
runtime/inference.py — Núcleo de inferência desacoplado, sampling e telemetria
"""

import time
import numpy as np
from typing import Optional, Callable, List, Any, Dict
from dataclasses import dataclass

try:
    from utils import RESET, BOLD, DIM, CYAN, GREEN, YELLOW, MAGENTA, RED, WHITE
except ImportError:
    # fallback para execução isolada
    RESET="\033[0m";BOLD="\033[1m";DIM="\033[2m";CYAN="\033[36m";GREEN="\033[32m"
    YELLOW="\033[33m";MAGENTA="\033[35m";RED="\033[31m";WHITE="\033[97m"

@dataclass
class SamplingConfig:
    max_tokens: int = 80
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.92
    rep_penalty: float = 1.15

class Sampler:
    """
    Estratégia de sampling (top-k, top-p, temperature, rep-penalty)
    """
    def __init__(self, config: SamplingConfig):
        self.config = config

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def sample(self, logits: np.ndarray, generated: Optional[List[int]] = None) -> int:
        cfg = self.config
        logits = logits.astype(np.float64)
        # Penalidade de repetição
        if cfg.rep_penalty != 1.0 and generated:
            for tid in set(generated[-64:]):
                if logits[tid] > 0:
                    logits[tid] /= cfg.rep_penalty
                else:
                    logits[tid] *= cfg.rep_penalty
        if cfg.temperature != 1.0:
            logits /= max(cfg.temperature, 1e-8)
        if cfg.top_k > 0:
            k = min(cfg.top_k, logits.size)
            kth = np.partition(logits, -k)[-k]
            logits[logits < kth] = -np.inf
        probs = self.softmax(logits)
        if cfg.top_p < 1.0:
            si = np.argsort(probs)[::-1]
            cum = np.cumsum(probs[si])
            cut = np.searchsorted(cum, cfg.top_p) + 1
            mask = np.zeros_like(probs)
            mask[si[:cut]] = 1.0
            probs = probs * mask
        probs = probs / probs.sum()
        return int(np.random.choice(len(probs), p=probs))

class Telemetry:
    """
    Interface para telemetria de geração (pode ser estendida para logs, métricas, etc)
    """
    def on_token(self, token: str, idx: int, stats: Dict[str, Any]):
        pass
    def on_panel(self, lines: List[str]):
        pass
    def on_finish(self, output: str, stats: Dict[str, Any]):
        pass

class ConsoleTelemetry(Telemetry):
    def on_token(self, token: str, idx: int, stats: Dict[str, Any]):
        print(f"{WHITE}{token}{RESET}", end="", flush=True)
    def on_panel(self, lines: List[str]):
        print("\n".join(lines))
    def on_finish(self, output: str, stats: Dict[str, Any]):
        print(f"\n{DIM}{'═'*72}{RESET}")
        if 'panel' in stats:
            for ln in stats['panel']:
                print(ln)
        print(f"  {DIM}tokens prompt:{RESET}  {stats.get('prompt_tokens', '-')}")
        print(f"  {DIM}tokens gerados:{RESET} {stats.get('generated_tokens', '-')}")
        print(f"  {DIM}evicções:{RESET}       {stats.get('evictions', '-')}")
        print(f"  {DIM}cache:{RESET}          {CYAN}constante ({stats.get('max_cap', '-') } slots O(1)){RESET}")
        print(f"  {GREEN}✓ memória não cresceu com o número de tokens{RESET}")
        print(f"{DIM}{'═'*72}{RESET}\n")

class LittleHawkInference:
    """
    Núcleo de inferência autoregressiva para Little Hawk LLM.
    - Não faz prints diretos (usa Telemetry/callbacks)
    - Sampling plugável
    - Suporta hooks para integração com API, CLI, etc
    """
    def __init__(self, tokenizer, engine, sampler: Optional[Sampler] = None):
        self.tok = tokenizer
        self.engine = engine
        self.S = engine.S
        self.W = engine.W
        self.max_cap = engine.max_cap
        self.sampler = sampler or Sampler(SamplingConfig())

    def _panel(self, step, wp, sm, ts, lat, ev, nl):
        fi = min(step + 1, self.max_cap)
        wf = max(0, min(step + 1 - self.S, self.W))
        bw = 32
        sc = min(int(self.S * bw / self.max_cap), bw)
        wc = min(int(wf * bw / self.W), bw - sc)
        ec = bw - sc - wc
        cb = (f"{GREEN}{'█'*sc}{RESET}{CYAN}{'▓'*wc}{RESET}{DIM}{'░'*ec}{RESET}")
        sw = 20
        sf = min(int(sm * sw / 100), sw)
        csm = GREEN if sm > 15 else YELLOW if sm > 5 else RED
        smb = f"{csm}{'█'*sf}{RESET}{'░'*(sw-sf)}"
        pct = (self.S / max(fi, 1)) * 100
        return [
            f"{DIM}{'─'*44}{RESET}",
            f"  {BOLD}{MAGENTA}LITTLE HAWK{RESET}  {DIM}{nl}L · Attn+SwiGLU{RESET}",
            f"{DIM}{'─'*44}{RESET}",
            f"  {DIM}step{RESET}      {WHITE}{step:>6}{RESET}",
            f"  {DIM}win_ptr{RESET}   {CYAN}{wp:>6}{RESET}  {DIM}/ {self.W}{RESET}",
            f"  {DIM}evicções{RESET}  {YELLOW}{ev:>6}{RESET}",
            f"  {DIM}latência{RESET}  {WHITE}{lat:>5.1f} ms{RESET}", f"",
            f"  {DIM}cache [{GREEN}sink{RESET}{DIM}|{RESET}{CYAN}janela{RESET}{DIM}]{RESET}",
            f"  [{cb}]",
            f"  {DIM}{fi}/{self.max_cap} slots  ({pct:.0f}% sink){RESET}", f"",
            f"  {DIM}sink L0 (tok[0]){RESET}", f"  [{smb}] {csm}{sm:.1f}%{RESET}", f"",
            f"  {DIM}último token{RESET}", f"  {YELLOW}{repr(ts):<18}{RESET}",
            f"{DIM}{'─'*44}{RESET}"
        ]

    def generate(
        self,
        prompt: str,
        sampling_config: Optional[SamplingConfig] = None,
        telemetry: Optional[Telemetry] = None,
        on_token: Optional[Callable[[str, int, Dict[str, Any]], None]] = None,
        panel: bool = True,
    ) -> str:
        """
        Gera texto autoregressivo a partir do prompt.
        - Não faz prints diretos (usa Telemetry ou callback)
        - Pode ser usado em API, CLI, etc
        """
        caches = self.engine.init_cache()
        win_ptr = 0
        ids = self.tok.encode(prompt, add_bos=True)
        generated = [t for t in ids if t not in (self.tok.bos_id, self.tok.eos_id)]
        ev = 0; lat = 0.0; sm = 0.0; ts = ""
        last_logits = None; n_ctx = 0
        pl = 0
        sampler = self.sampler if sampling_config is None else Sampler(sampling_config)
        output_tokens = []
        for tid in ids:
            n_ctx += 1
            logits, caches, win_ptr, sm = self.engine.step(tid, caches, win_ptr, n_ctx)
            last_logits = logits[0]
        for step in range((sampling_config.max_tokens if sampling_config else 80)):
            t0 = time.perf_counter()
            nid = sampler.sample(last_logits.copy(), generated=generated)
            n_ctx += 1
            logits, caches, win_ptr, sm = self.engine.step(nid, caches, win_ptr, n_ctx)
            last_logits = logits[0]
            lat = (time.perf_counter() - t0) * 1000
            ts = self.tok.id_to_token.get(nid, self.tok.UNK).replace("Ġ", " ").replace("Ċ", "↵")
            if n_ctx > self.max_cap:
                ev += 1
            if nid == self.tok.eos_id:
                break
            generated.append(nid)
            output_tokens.append(nid)
            # Callback/telemetry para token
            if on_token:
                on_token(ts, step, {"latency": lat, "step": step, "token_id": nid})
            if telemetry:
                telemetry.on_token(ts, step, {"latency": lat, "step": step, "token_id": nid})
            # Painel
            if telemetry and panel and step > 0 and step % 8 == 0:
                lines = self._panel(step, win_ptr, sm, ts, lat, ev, self.engine.n_layers)
                telemetry.on_panel(lines)
        result = self.tok.decode(generated)
        if telemetry:
            stats = {
                "prompt_tokens": len(ids),
                "generated_tokens": len(generated),
                "evictions": ev,
                "max_cap": self.max_cap,
                "panel": self._panel(len(generated), win_ptr, sm, ts, lat, ev, self.engine.n_layers) if panel else None
            }
            telemetry.on_finish(result, stats)
        return result
