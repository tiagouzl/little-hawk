#!/usr/bin/env python3
"""
bench_ruler_eviction.py — Little Hawk

Benchmark estilo RULER (needle-in-haystack) para comparar as estratégias
de evicção Nexus vs FIFO em contextos que ultrapassam a janela W=508
(cache total S+W=512).

Ideia do teste: esconde uma "agulha" (um número mágico único) numa
posição X% da profundidade de um contexto longo, gera texto de
preenchimento (filler) até atingir N tokens totais, e no final pergunta
ao modelo qual era o número.

IMPORTANTE — dois modos:

  --baseline-only
      Roda SÓ dentro da janela (context_length bem abaixo de 512, sem
      nenhuma evicção acontecendo — FIFO e Nexus são idênticos nesse
      regime). Mede a capacidade crua de retrieval do checkpoint em si,
      sem a variável de evicção contaminando o resultado. RODE ISSO
      PRIMEIRO. Se a acurácia aqui já for baixa, o sweep completo abaixo
      não vai te dizer nada útil sobre Nexus vs FIFO — o gargalo é o
      modelo, não o gerenciador de cache.

  (default, sem a flag)
      Sweep completo comparando fifo vs nexus em contextos que estouram
      a janela — só faz sentido rodar depois de confirmar um baseline
      aceitável.

Chama a CLI já existente do projeto via subprocess:
    python -m cli.main infer --prompt "..." --eviction {nexus,fifo} \
        --min-p ... --max-tokens ... --no-panel
(cli/main.py:20, conforme relatório — --no-panel evita que o
banner/painel do Rich polua o stdout que o parser lê.)

Uso:
    # 1) Confirme o baseline antes de qualquer coisa (rápido, cabe num café)
    python bench_ruler_eviction.py --weights pesos.npz --baseline-only \
        --context-lengths 256 400 --depths 0.1 0.5 0.9 --repeats 3 --max-tokens 8

    # 2) Só depois, o sweep completo fifo vs nexus além da janela
    python bench_ruler_eviction.py --weights pesos.npz \
        --context-lengths 600 800 1200 1500 --depths 0.1 0.5 0.9 --repeats 3

    # Modo mock, só pra validar a lógica do harness sem pesos/modelo
    python bench_ruler_eviction.py --mock --context-lengths 600 1500 --depths 0.1 0.9 --repeats 2

Saída: tabela de accuracy, mais latência (wall time) e pico de RSS por
execução, salvos em JSON.
"""

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

try:
    import psutil
except ImportError:
    psutil = None


FILLER_SENTENCES = [
    "O céu estava nublado naquela tarde em Mossoró.",
    "A biblioteca municipal recebeu novos livros de história.",
    "O trânsito na avenida principal ficou intenso às seis.",
    "Um grupo de estudantes discutia projetos de robótica.",
    "A padaria do bairro vendeu todo o pão até o meio-dia.",
    "O time local venceu o campeonato regional de futebol.",
    "As chuvas de março ajudaram a plantação de milho.",
    "O mercado de tecnologia cresceu no último trimestre.",
    "Uma nova ciclovia foi inaugurada na zona leste.",
    "O museu de arte contemporânea abriu uma nova ala.",
]

DEFAULT_CLI_TIMEOUT_S = 120


@dataclass
class TrialResult:
    context_length: int
    depth: float
    eviction: str
    correct: bool
    wall_s: float
    peak_rss_mb: float | None
    timed_out: bool = False
    raw_output: str = field(repr=False, default="")
    trial_id: int = 0  # identifica prompts idênticos entre modos de evicção (desenho pareado)


def build_prompt(context_length: int, depth: float, rng: random.Random, style: str = "continuation") -> tuple[str, str]:
    """
    Monta um prompt de ~context_length tokens (aprox. por palavras) com
    uma agulha (número mágico) inserida na profundidade `depth` (0.0 a 1.0).
    Retorna (prompt, magic_number_esperado).

    style="continuation" (default, recomendado para checkpoints BASE, sem
    instruction-tuning): a agulha aparece uma vez no meio do texto, e o
    prompt termina com o MESMO prefixo da frase-agulha, cortado antes do
    número — um checkpoint base tende a completar naturalmente pela
    estatística de repetição de padrão, sem precisar entender uma
    instrução tipo "responda a pergunta".

    style="instruct": formato pergunta/resposta explícito (funciona bem
    com checkpoints *-Instruct*; em checkpoints base tende a gerar
    continuação de texto solta em vez de responder — ver diagnóstico do
    raw_output).
    """
    magic_number = str(rng.randint(100000, 999999))
    n_words_target = int(context_length / 1.3)  # aprox. 1.3 tokens/palavra em pt-BR
    needle_word_pos = int(n_words_target * depth)

    words: list[str] = []
    while len(words) < n_words_target:
        words.extend(rng.choice(FILLER_SENTENCES).split())

    needle_prefix = "O número mágico secreto para esta tarefa é"
    needle_sentence = f"{needle_prefix} {magic_number}. Lembre-se dele.".split()

    insertion_point = min(needle_word_pos, len(words))
    full = words[:insertion_point] + needle_sentence + words[insertion_point:n_words_target]

    if style == "continuation":
        # Termina com o mesmo prefixo, sem o número — o modelo completa.
        prompt = " ".join(full) + f"\n\n{needle_prefix}"
    else:
        prompt = " ".join(full) + "\n\nPergunta: qual é o número mágico secreto mencionado acima? Responda apenas com o número."
    return prompt, magic_number


def run_cli(
    cli_module: str,
    weights: str,
    prompt: str,
    eviction: str,
    min_p: float,
    max_tokens: int,
    timeout_s: int,
    extra_args: list[str],
) -> tuple[str, float, float | None, bool]:
    """
    Executa a CLI real via subprocess, medindo wall time e pico de RSS.

    Usa communicate(timeout=...) em vez de ler stdout manualmente — ler
    com Popen + amostrar RSS em loop na mesma thread trava se o buffer
    de stdout/stderr (64KB) enche antes do processo terminar. A
    amostragem de RSS roda numa thread separada; o processo em si é
    sempre lido via communicate().
    """
    cmd = [
        sys.executable, "-m", cli_module, "infer",
        "--prompt", prompt,
        "--eviction", eviction,
        "--min-p", str(min_p),
        "--max-tokens", str(max_tokens),
        "--weights", weights,
        "--no-panel",
        *extra_args,
    ]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    rss_samples: list[float] = []
    stop_flag = threading.Event()

    def sample_rss():
        if psutil is None:
            return
        try:
            ps_proc = psutil.Process(proc.pid)
        except psutil.NoSuchProcess:
            return
        while not stop_flag.is_set():
            try:
                rss_samples.append(ps_proc.memory_info().rss / (1024 * 1024))
            except psutil.NoSuchProcess:
                break
            time.sleep(0.1)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()

    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        timed_out = True
    finally:
        stop_flag.set()
        sampler.join(timeout=2)

    wall_s = time.perf_counter() - t0
    peak_rss = max(rss_samples) if rss_samples else None

    if stderr and ("Traceback" in stderr or "Error" in stderr[-500:]):
        print(f"  [aviso] stderr não vazio para eviction={eviction}: {stderr[-300:]}", file=sys.stderr)
    if timed_out:
        print(f"  [aviso] TIMEOUT ({timeout_s}s) para eviction={eviction}, ctx maior que o previsto?", file=sys.stderr)

    return stdout, wall_s, peak_rss, timed_out


def run_mock(prompt: str, eviction: str, magic_number: str, context_length: int, rng: random.Random) -> str:
    """Simula respostas plausíveis pra validar a lógica do harness sem pesos reais."""
    overflow = max(0, context_length - 512)
    if eviction == "fifo":
        p_correct = max(0.05, 1.0 - overflow / 800)
    else:  # nexus
        p_correct = max(0.15, 1.0 - overflow / 1800)
    return magic_number if rng.random() < p_correct else str(rng.randint(100000, 999999))


def extract_number(text: str) -> str | None:
    """
    Pega o ÚLTIMO número de 6 dígitos na saída, não o primeiro — se a CLI
    ecoar o prompt antes da resposta (mesmo com --no-panel), o primeiro
    match pode ser lixo de outro lugar (ex: win_ptr/step nos logs). A
    resposta do modelo é o que vem por último.
    """
    matches = re.findall(r"\b(\d{6})\b", text)
    return matches[-1] if matches else None


def run_sweep(args, eviction_modes: list[str]) -> list[TrialResult]:
    """
    Desenho PAREADO: o mesmo prompt (mesma agulha, mesmo filler) é gerado
    UMA vez por (context_length, depth, rep) e testado sob TODOS os modos
    de evicção — condição necessária pro teste de McNemar comparar
    acerto/erro no mesmo caso, não em amostras independentes.
    """
    rng = random.Random(args.seed)
    results: list[TrialResult] = []
    total = len(args.context_lengths) * len(args.depths) * len(eviction_modes) * args.repeats
    done = 0
    trial_id = 0

    print(f"{'ctx':>6} {'depth':>6} {'evict':>6} {'rep':>4}  resultado")
    print("-" * 50)

    for ctx_len in args.context_lengths:
        for depth in args.depths:
            for rep in range(args.repeats):
                # Prompt gerado UMA vez, reaplicado a cada modo de evicção.
                prompt, magic_number = build_prompt(ctx_len, depth, rng, style=args.prompt_style)
                trial_id += 1

                for eviction in eviction_modes:
                    if args.mock:
                        t0 = time.perf_counter()
                        output = run_mock(prompt, eviction, magic_number, ctx_len, rng)
                        wall_s = time.perf_counter() - t0
                        peak_rss, timed_out = None, False
                    else:
                        output, wall_s, peak_rss, timed_out = run_cli(
                            args.cli_module, args.weights, prompt, eviction,
                            args.min_p, args.max_tokens, args.timeout, [],
                        )

                    answered = extract_number(output)
                    correct = (answered == magic_number) and not timed_out
                    results.append(TrialResult(ctx_len, depth, eviction, correct, wall_s, peak_rss, timed_out, output, trial_id))

                    done += 1
                    tag = "TIMEOUT" if timed_out else ("OK " if correct else "ERR")
                    print(f"{ctx_len:6d} {depth:6.2f} {eviction:>6} {rep+1:4d}  [{tag}] ({done}/{total})  {wall_s:.1f}s")

    return results


def mcnemar_exact_p(b: int, c: int) -> float:
    """
    Teste de McNemar exato (binomial), sem depender de scipy: testa se
    b (só o primeiro modo acerta) e c (só o segundo acerta) são
    igualmente prováveis, condicionado a b+c discordâncias.
    p bicaudal = 2 * P(X <= min(b,c)) sob Binomial(n=b+c, p=0.5),
    truncado em 1.0. Recomendado sobre a versão com correção de
    continuidade quando b+c < 25 (caso típico aqui).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)

    def binom_cdf(k: int, n: int, p: float = 0.5) -> float:
        from math import comb
        return sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k + 1))

    p_value = 2 * binom_cdf(k, n)
    return min(p_value, 1.0)


def paired_breakdown(results: list[TrialResult], mode_a: str, mode_b: str, group_key) -> dict:
    """
    Agrupa por group_key (ex: lambda r: r.depth) e calcula, para cada
    grupo, quantos trial_ids têm resultado pareado disponível para
    mode_a e mode_b, junto com a contagem de discordâncias (b, c) para
    o McNemar e a acurácia de cada modo no grupo.
    """
    by_trial: dict[int, dict[str, TrialResult]] = {}
    for r in results:
        by_trial.setdefault(r.trial_id, {})[r.eviction] = r

    groups: dict = {}
    for per_mode in by_trial.values():
        if mode_a not in per_mode or mode_b not in per_mode:
            continue
        key = group_key(per_mode[mode_a])
        g = groups.setdefault(key, {"n": 0, "a_correct": 0, "b_correct": 0, "b_only": 0, "c_only": 0})
        a_ok, b_ok = per_mode[mode_a].correct, per_mode[mode_b].correct
        g["n"] += 1
        g["a_correct"] += int(a_ok)
        g["b_correct"] += int(b_ok)
        if a_ok and not b_ok:
            g["b_only"] += 1  # só mode_a acertou
        elif b_ok and not a_ok:
            g["c_only"] += 1  # só mode_b acertou

    for g in groups.values():
        g["acc_a"] = g["a_correct"] / g["n"] if g["n"] else float("nan")
        g["acc_b"] = g["b_correct"] / g["n"] if g["n"] else float("nan")
        g["mcnemar_p"] = mcnemar_exact_p(g["b_only"], g["c_only"])

    return groups


def print_summary(results: list[TrialResult], eviction_modes: list[str], context_lengths: list[int], depths: list[float]):
    print("\n" + "=" * 72)
    print("Accuracy agregada (fração de acertos por combinação)")
    print("=" * 72)
    cols = "".join(f"{m:>8}" for m in eviction_modes)
    header = f"{'ctx':>6} {'depth':>6}{cols}"
    if len(eviction_modes) == 2:
        header += f"{'delta':>8}"
    print(header)
    print("-" * len(header))

    summary = []
    for ctx_len in context_lengths:
        for depth in depths:
            row = {}
            for eviction in eviction_modes:
                subset = [r.correct for r in results if r.context_length == ctx_len and r.depth == depth and r.eviction == eviction]
                row[eviction] = sum(subset) / len(subset) if subset else float("nan")
            line = f"{ctx_len:6d} {depth:6.2f}" + "".join(f"{row[m]:8.2f}" for m in eviction_modes)
            if len(eviction_modes) == 2:
                delta = row[eviction_modes[1]] - row[eviction_modes[0]]
                line += f"{delta:+8.2f}"
                row["delta"] = delta
            print(line)
            summary.append({"context_length": ctx_len, "depth": depth, **row})

    overall = {}
    for eviction in eviction_modes:
        vals = [r.correct for r in results if r.eviction == eviction]
        overall[eviction] = statistics.mean(vals) if vals else float("nan")
    total_line = f"{'TOTAL':>6} {'':>6}" + "".join(f"{overall[m]:8.2f}" for m in eviction_modes)
    if len(eviction_modes) == 2:
        total_line += f"{overall[eviction_modes[1]] - overall[eviction_modes[0]]:+8.2f}"
    print("-" * len(header))
    print(total_line)

    return summary, overall


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=str, default=None, help="Caminho para o .npz de pesos (ignorado em --mock)")
    ap.add_argument("--cli-module", type=str, default="cli.main", help="Módulo da CLI (default: cli.main)")
    ap.add_argument("--context-lengths", type=int, nargs="+", default=[600, 800, 1200, 1500])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9], help="Profundidade da agulha (0.0-1.0)")
    ap.add_argument("--repeats", type=int, default=3, help="Repetições por combinação")
    ap.add_argument("--min-p", type=float, default=0.05)
    ap.add_argument("--max-tokens", type=int, default=8, help="Tokens gerados na resposta (a agulha é curta)")
    ap.add_argument("--timeout", type=int, default=DEFAULT_CLI_TIMEOUT_S, help="Timeout por chamada da CLI, em segundos")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--mock", action="store_true", help="Roda sem chamar a CLI real, só valida o harness")
    ap.add_argument(
        "--baseline-only", action="store_true",
        help="Roda só dentro da janela (sem evicção variável) pra medir retrieval cru do modelo antes do sweep completo",
    )
    ap.add_argument("--json", type=str, default=None, help="Caminho do JSON de saída (default automático por modo)")
    ap.add_argument(
        "--prompt-style", choices=["continuation", "instruct"], default="continuation",
        help="'continuation' (default) para checkpoints base; 'instruct' para checkpoints *-Instruct*",
    )
    ap.add_argument(
        "--modes", type=str, default=None,
        help="Políticas de evicção separadas por vírgula (ex: 'fifo,nexus,nexus-salience'). "
             "Default: fifo,nexus. Desenho pareado: cada prompt roda em TODAS as modas.",
    )
    args = ap.parse_args()

    if not args.mock and not args.weights:
        ap.error("--weights é obrigatório fora do modo --mock")

    if args.baseline_only:
        over_window = [c for c in args.context_lengths if c >= 480]
        if over_window:
            print(
                f"[aviso] --baseline-only espera context_lengths bem abaixo de 512 "
                f"(janela S+W); {over_window} está perto ou acima do limite e pode já sofrer evicção.",
                file=sys.stderr,
            )
        eviction_modes = ["fifo"]  # dentro da janela, fifo == nexus (nada é evictado ainda)
        json_path = args.json or "ruler_baseline_results.json"
    else:
        eviction_modes = (
            [m.strip() for m in args.modes.split(",")] if args.modes else ["fifo", "nexus"]
        )
        json_path = args.json or ("ruler_eviction_results.json" if len(eviction_modes) == 2 else "ruler_multimode_results.json")

    results = run_sweep(args, eviction_modes)
    summary, overall = print_summary(results, eviction_modes, args.context_lengths, args.depths)

    if not args.baseline_only and len(eviction_modes) >= 2:
        from itertools import combinations
        print("\n" + "=" * 72)
        print("McNemar pareado por profundidade (mesmos prompts, todos os pares)")
        print("=" * 72)
        paired_all = {}
        for mode_a, mode_b in combinations(eviction_modes, 2):
            print(f"\n--- {mode_a} vs {mode_b} ---")
            by_depth = paired_breakdown(results, mode_a, mode_b, group_key=lambda r: r.depth)
            header = f"{'depth':>6} {'n':>4} {f'acc_{mode_a[:12]}':>14} {f'acc_{mode_b[:12]}':>14} {'delta':>8} {'b/c':>7} {'p':>8}"
            print(header)
            print("-" * len(header))
            for depth in sorted(by_depth):
                g = by_depth[depth]
                delta = g["acc_b"] - g["acc_a"]
                sig = " *" if g["mcnemar_p"] < 0.05 else ("  " if (g["b_only"] + g["c_only"]) >= 10 else " (n baixo)")
                print(
                    f"{depth:6.2f} {g['n']:4d} {g['acc_a']:14.2f} {g['acc_b']:14.2f} {delta:+8.2f} "
                    f"{g['b_only']:>3d}/{g['c_only']:<3d} {g['mcnemar_p']:8.3f}{sig}"
                )
            all_group = paired_breakdown(results, mode_a, mode_b, group_key=lambda r: "all")
            g = all_group.get("all", {"n": 0})
            print("-" * len(header))
            print(
                f"{'TODOS':>6} {g['n']:4d} {g['acc_a']:14.2f} {g['acc_b']:14.2f} "
                f"{g['acc_b'] - g['acc_a']:+8.2f} {g['b_only']:>3d}/{g['c_only']:<3d} {g['mcnemar_p']:8.3f}"
            )
            paired_all[f"{mode_a}_vs_{mode_b}"] = {"by_depth": by_depth, "overall": g}
        print(
            "\n(* = p<0.05; 'n baixo' sinaliza menos de ~10 discordâncias pareadas no par, "
            "onde mesmo um delta grande pode não ser conclusivo)"
        )

    if args.baseline_only:
        acc = overall["fifo"]
        print(f"\nRetrieval cru dentro da janela: {acc:.2f}")
        if acc < 0.7:
            print(
                "=> Acurácia baixa mesmo sem evicção envolvida: o gargalo é a capacidade do "
                "checkpoint, não o gerenciador de cache. Considere um modelo maior (smollm2-360m "
                "ou 1.7b) antes de investir no sweep fifo-vs-nexus."
            )
        else:
            print("=> Baseline aceitável — agora o sweep completo (sem --baseline-only) é informativo.")
    else:
        delta = overall["nexus"] - overall["fifo"]
        if delta > 0:
            print(f"\n=> Nexus supera FIFO em {delta * 100:.1f} p.p. de accuracy geral.")
        elif delta < 0:
            print(f"\n=> FIFO supera Nexus em {-delta * 100:.1f} p.p. — revisar alpha/R do Nexus.")
        else:
            print("\n=> Empate — sem diferença mensurável neste conjunto de testes.")

    n_timeouts = sum(1 for r in results if r.timed_out)
    if n_timeouts:
        print(f"\n[aviso] {n_timeouts}/{len(results)} trials bateram timeout ({args.timeout}s) — considere aumentar --timeout.")

    paired_json = None
    if not args.baseline_only and len(eviction_modes) >= 2:
        from itertools import combinations
        paired_json = {}
        for mode_a, mode_b in combinations(eviction_modes, 2):
            by_depth = paired_breakdown(results, mode_a, mode_b, group_key=lambda r: r.depth)
            all_group = paired_breakdown(results, mode_a, mode_b, group_key=lambda r: "all")
            paired_json[f"{mode_a}_vs_{mode_b}"] = {"by_depth": by_depth, "overall": all_group.get("all")}

    with open(json_path, "w") as f:
        json.dump(
            {
                "mode": "baseline_only" if args.baseline_only else "full_sweep",
                "config": {k: v for k, v in vars(args).items()},
                "summary": summary,
                "overall": overall,
                "paired_mcnemar": paired_json,
                "raw": [
                    {
                        "context_length": r.context_length, "depth": r.depth, "eviction": r.eviction,
                        "correct": r.correct, "wall_s": round(r.wall_s, 3), "peak_rss_mb": r.peak_rss_mb,
                        "timed_out": r.timed_out, "trial_id": r.trial_id,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )
    print(f"\nResultados completos salvos em {json_path}")


if __name__ == "__main__":
    main()
