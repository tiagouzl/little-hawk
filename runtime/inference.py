"""
runtime/inference.py — Núcleo de inferência desacoplado, sampling e telemetria
"""

import time
import numpy as np
from typing import Optional, Callable, List, Any, Dict
from dataclasses import dataclass
from runtime.tokenizer import StreamDecoder
from engine.speculative import can_verify, verify_chunk

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
    min_p: float = 0.0  # 0 desativa; mantém apenas tokens com p >= min_p · p_max

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

    def sample(self, logits: np.ndarray, generated: list[int] | None = None, rng=None) -> int:
        cfg = self.config
        rng = rng if rng is not None else np.random
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
        if cfg.min_p > 0.0:
            # min_p: descarta tokens cuja probabilidade < min_p · p_max
            # (mais robusto que top_k em distribuições achatadas — mitiga drift longo)
            probs = np.where(probs >= cfg.min_p * probs.max(), probs, 0.0)
            s = probs.sum()
            if s <= 0.0:  # degenereu (não deve ocorrer: p_max sempre passa)
                probs = self.softmax(logits)
            else:
                probs = probs / s
        if cfg.top_p < 1.0:
            si = np.argsort(probs)[::-1]
            cum = np.cumsum(probs[si])
            cut = np.searchsorted(cum, cfg.top_p) + 1
            mask = np.zeros_like(probs)
            mask[si[:cut]] = 1.0
            probs = probs * mask
        probs = probs / probs.sum()
        return int(rng.choice(len(probs), p=probs))

class Telemetry:
    """
    Interface para telemetria de geração (pode ser estendida para logs, métricas, etc)
    """
    def on_token(self, token: str, idx: int, stats: dict[str, Any]):
        pass
    def on_panel(self, lines: list[str]):
        pass
    def on_finish(self, output: str, stats: dict[str, Any]):
        pass

class ConsoleTelemetry(Telemetry):
    def on_token(self, token: str, idx: int, stats: dict[str, Any]):
        print(f"{WHITE}{token}{RESET}", end="", flush=True)
    def on_panel(self, lines: list[str]):
        print("\n".join(lines))
    def on_finish(self, output: str, stats: dict[str, Any]):
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
    def __init__(self, tokenizer, engine, sampler: Sampler | None = None):
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
            f"  {DIM}último token{RESET}", f"  {YELLOW}{ts!r:<18}{RESET}",
            f"{DIM}{'─'*44}{RESET}"
        ]

    def generate(
        self,
        prompt: str,
        sampling_config: SamplingConfig | None = None,
        telemetry: Telemetry | None = None,
        on_token: Callable[[str, int, dict[str, Any]], None] | None = None,
        panel: bool = True,
        speculative_k: int = 0,
    ) -> str:
        """
        Gera texto autoregressivo a partir do prompt.
        - Não faz prints diretos (usa Telemetry ou callback)
        - Pode ser usado em API, CLI, etc
        - speculative_k > 0 ativa N-gram speculative decoding greedy (Fase B)
        """
        if speculative_k and hasattr(self.engine, "verify_chunk"):
            from engine.speculative import can_verify
            if can_verify(self.engine, self.max_cap + 1, speculative_k):
                return self._generate_speculative(
                    prompt, sampling_config, on_token, speculative_k
                )
        caches = self.engine.init_cache()
        win_ptr = 0
        sdec = StreamDecoder(self.tok)
        ids = self.tok.encode(prompt, add_bos=True)
        generated = [t for t in ids if t not in (self.tok.bos_id, self.tok.eos_id)]
        ev = 0; lat = 0.0; sm = 0.0; ts = ""
        last_logits = None; n_ctx = 0
        sampler = self.sampler if sampling_config is None else Sampler(sampling_config)
        output_tokens = []
        # Prefill batched do prompt (TTFT ~10-20× menor que step sequencial);
        # engine.prefill lida com chunked >max_cap e expõe mesma interface para ONNX
        if len(ids) > 0 and hasattr(self.engine, "prefill"):
            lg_pre, caches, win_ptr, sm = self.engine.prefill(ids, caches)
            last_logits, n_ctx = lg_pre[0], len(ids)
        else:
            for tid in ids:
                n_ctx += 1
                logits, caches, win_ptr, sm = self.engine.step(tid, caches, win_ptr, n_ctx)
                last_logits = logits[0]
        for step in range(sampling_config.max_tokens if sampling_config else 80):
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
            # Callback/telemetry para token — texto decodificado byte-safe
            decoded = sdec.push(nid)
            if on_token:
                on_token(decoded, step, {"latency": lat, "step": step, "token_id": nid})
            if telemetry:
                telemetry.on_token(decoded, step, {"latency": lat, "step": step, "token_id": nid})
            # Painel
            if telemetry and panel and step > 0 and step % 8 == 0:
                lines = self._panel(step, win_ptr, sm, ts, lat, ev, self.engine.n_layers)
                telemetry.on_panel(lines)
        sdec.flush()
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

    def _generate_speculative(self, prompt, cfg, on_token, k: int) -> str:
        """Fase B — greedy N-gram speculative decoding (ANALISE §21.2).

        Contrato: 2 forwards do alvo por rodada (verify + step bônus), m+1
        tokens emitidos. Rollback por restauração dos slots rejeitados.
        Sampler consultado apenas para o token bônus — rascunhos são
        aceitos/rejeitados por argmax do alvo.
        """
        from runtime.speculative import NGramSpeculator, SpeculativeStats, restore_slot

        eng = self.engine
        stats = SpeculativeStats()
        caches = eng.init_cache()
        win_ptr = 0
        sdec = StreamDecoder(self.tok)
        ids = self.tok.encode(prompt, add_bos=True)
        all_ids = list(ids)
        generated = [t for t in ids if t not in (self.tok.bos_id, self.tok.eos_id)]
        spec = NGramSpeculator(n=3)
        # semeia a tabela com todos os trigramas do prompt — sem isso a
        # primeira rodada nunca teria hits (bug pego por teste fraco que
        # aceitava rounds=0 silenciosamente)
        for i in range(len(ids) - spec.n + 1):
            spec.observe(ids[i:i + spec.n])
        ev = 0; lat = 0.0; sm = 0.0; ts = ""
        sampler = self.sampler
        max_tokens = cfg.max_tokens if cfg else 80
        n_ctx = 0
        t_start = time.perf_counter()

        def feed_and_emit(nid):
            nonlocal lat, ts, ev
            decoded = sdec.push(nid)
            if on_token:
                on_token(decoded, stats.emitted, {"latency": lat, "step": stats.emitted, "token_id": nid})

        # prefill compartilhado com caminho normal
        lg, caches, win_ptr, sm = eng.prefill(ids, caches)
        prev_logits = lg[0]
        n_ctx = len(ids)
        stats.forwards += 1

        speculative_mode = True
        emitted_count = 0
        stopped = False
        while emitted_count < max_tokens and not stopped:
            if not can_verify(eng, n_ctx, k):
                speculative_mode = False
                break
            drafts = spec.propose(all_ids, k)
            if not drafts:
                speculative_mode = False
                break
            kd = len(drafts)
            stats.rounds += 1
            stats.proposed += kd
            cand_slots = (win_ptr + np.arange(kd, dtype=np.int64)) % eng.W + eng.S
            snap = [
                (caches[li][0][:, :, cand_slots, :].copy(),
                 caches[li][1][:, :, cand_slots, :].copy())
                for li in range(eng.n_layers)
            ]
            t0 = time.perf_counter()
            lg_all, caches, _, _ = verify_chunk(eng, drafts, caches, win_ptr, n_ctx)
            stats.forwards += 1
            lat = (time.perf_counter() - t0) * 1000

            # aceitação greedy contra logits prévios e intermediários
            m = 0
            cur = prev_logits
            for j, d in enumerate(drafts):
                if int(np.argmax(cur)) != d:
                    break
                m += 1
                cur = lg_all[j]
            stats.accepted += m

            for j in range(m, kd):                       # rollback sufixo rejeitado
                for li in range(eng.n_layers):
                    kc_snap, vc_snap = snap[li]
                    caches[li][0][:, :, cand_slots[j]:cand_slots[j] + 1, :] = \
                        kc_snap[:, :, j:j + 1, :]
                    caches[li][1][:, :, cand_slots[j]:cand_slots[j] + 1, :] = \
                        vc_snap[:, :, j:j + 1, :]
                stats.rollback_restores += 1
            win_ptr = (win_ptr + m) % eng.W
            n_ctx += m
            for j in range(m):
                nid = drafts[j]
                generated.append(nid); all_ids.append(nid)
                spec.observe(all_ids[-spec.n:])
                emit_fn = sdec.push(nid)
                if on_token:
                    on_token(emit_fn, stats.emitted,
                             {"latency": lat, "step": stats.emitted, "token_id": nid})
                stats.emitted += 1; emitted_count += 1
                if n_ctx > eng.max_cap:
                    ev += 1
                if nid == self.tok.eos_id:
                    stopped = True
                    break

            if stopped or emitted_count >= max_tokens:
                break
            # token bônus: o alvo assume no ponto de parada
            t0 = time.perf_counter()
            nid = sampler.sample(cur.copy(), generated=generated)
            n_ctx += 1
            lg_b, caches, win_ptr, sm = eng.step(nid, caches, win_ptr, n_ctx)
            prev_logits = lg_b[0]
            stats.forwards += 1
            lat = (time.perf_counter() - t0) * 1000
            ts = self.tok.id_to_token.get(nid, self.tok.UNK).replace("Ġ", " ").replace("Ċ", "↵")
            if n_ctx > eng.max_cap:
                ev += 1
            if nid == self.tok.eos_id:
                stopped = True
                break
            generated.append(nid); all_ids.append(nid)
            spec.observe(all_ids[-spec.n:])
            dec = sdec.push(nid)
            if on_token:
                on_token(dec, stats.emitted,
                         {"latency": lat, "step": stats.emitted, "token_id": nid})
            stats.emitted += 1; emitted_count += 1

        # cauda sequencial (fallback ou complemento até max_tokens)
        while emitted_count < max_tokens and not stopped:
            stats.fallback_steps += 1
            t0 = time.perf_counter()
            nid = sampler.sample(prev_logits.copy(), generated=generated)
            n_ctx += 1
            lg, caches, win_ptr, sm = eng.step(nid, caches, win_ptr, n_ctx)
            prev_logits = lg[0]
            stats.forwards += 1
            lat = (time.perf_counter() - t0) * 1000
            ts = self.tok.id_to_token.get(nid, self.tok.UNK).replace("Ġ", " ").replace("Ċ", "↵")
            if n_ctx > eng.max_cap:
                ev += 1
            if nid == self.tok.eos_id:
                break
            generated.append(nid); all_ids.append(nid)
            spec.observe(all_ids[-spec.n:])
            dec = sdec.push(nid)
            if on_token:
                on_token(dec, stats.emitted,
                         {"latency": lat, "step": stats.emitted, "token_id": nid})
            stats.emitted += 1; emitted_count += 1

        sdec.flush()
        stats.wall_s = time.perf_counter() - t_start
        return self.tok.decode(generated), stats
