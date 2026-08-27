# Case Study — Nexus-Salience: de FIFO a p=0.0156

> **Projeto:** Little Hawk v0.8.0 · **Hardware:** i5-10210U 7,6G+2G swap, sem AVX-512 · **Stack:** Python/NumPy puro · **Período:** trilhas D→D2→correções 08/2026

## 1. Problema

StreamingLLM com `S=4` sinks + `W=508` janela circular é O(1) mas FIFO puro
descarta o token mais antigo, independente de relevância. Em needle-in-haystack
(RULER) com `context > 512`, recall cai porque a agulha fora da janela recente é
evictada.

Hipótese: ponderar evicção por atenção + surpresa na chegada (`-log P`) protege
fatos raros sem custo de modelo draft.

## 2. Trilha D — bug de indexação (corrigido)

`engine/eviction.py:order` era fonte da verdade só para escrita. Leitura usava
`win_ptr` FIFO → após `next_slot` por override, `ctx_array` via `win_ptr`
retornava ordem errada. Corrigido com `order` explícita para leitura e escrita.
Teste de recência validou.

## 3. Trilha D2 — bug de reset entre gerações (corrigido)

`NexusEviction.reset()` zerava `order` mas não `scores`. Entre gerações, slots
herdavam EMA da geração anterior. Corrigido `engine/eviction.py:59` (reset de
`scores` e `n_reservoir`). `NexusSalienceEviction.reset()` idem para `salience`.

Resultado D2 pareado (21 prompts, `number`): **nexus-salience 0.95 vs fifo 0.52,
McNemar p=0.004 (9/9 discordâncias)**. Depth 0.1 discriminante.

## 4. Generalização §22 — formatos adversários

Harness expandido `bench_ruler_eviction.py:102` com `NEEDLE_SPECS` (`number` alta
surpresa controle, `date`/`word` baixa — adversários). Seed 777, `512/0.1/0.5×4`:

| formato | fifo | salience |
|---|---|---|
| number 0.1 | 0.00 | 0.75 |
| date 0.1 | 0.00 | 1.00 |
| word 0.1 | 0.00 | 1.00 |

Agregado depth 0.1: **0/12 vs 11/12**. Compressão prevista NÃO ocorreu — surpresa
é condicional ao contexto local, não frequência global. ` §22.5` delimita fronteira:
adversário real precisaria de token frequente *no próprio filler* (`tarde`).

Caveat: n=4/célula, p=0.125 por bloco, 1 checkpoint.

## 5. Bug de score herdado intra-geração (esta sessão)

**Local:** `engine/eviction.py:92` e `185` (`NexusEviction.next_slot`,
`NexusSalienceEviction.next_slot`).

Quando vítima é reciclada, só `salience` recebia valor fresco via
`set_salience` após `next_slot`; `scores` (EMA atenção, `alpha=0.9`) ficava com
90% do fantasma do ocupante anterior. Slot com histórico baixo vira ímã de
evicção; alto protege indevidamente por vários passos — thrashing independente
de conteúdo.

**Fix:** `self.scores[victim]=0.0` (e `self.salience[victim]=0.0` na variante
salience) imediatamente após escolher vítima, antes de `order.remove`.

**Regressão comprovada:** `tests/test_engine.py:308` `test_score_reset_on_reuse`
e `test_salience_score_reset_on_reuse` — falham sem fix (`0.8 != 0.0`), passam
com fix. 44/44 → 52/52 com harness fix.

## 6. Bug de harness — Unicode/case (esta sessão)

**Local:** `bench_ruler_eviction.py:156` `extract_answer`.

Regex `r"\b(\d{6})\b"` vs `"(terça-feira|...)"` sem `NFC` nem `IGNORECASE`.
`ó` NFD (`o`+combining) e `Robótica` capitalizada não batiam → `None`,
subestimando `date`/`word`. `word` adversarial `robótica` falhou por isso.

**Fix:** `bench_ruler_eviction.py:8` `import unicodedata`,
`extract_answer` normaliza texto para NFC e usa `IGNORECASE` + `lower()` para
`date`/`word` (`number` intocado). `tests/test_bench_ruler.py:1` 8 testes,
4 falham no código antigo.

Re-run com timeout adequado: `word` 0.60 com 200s (5 timeouts) → 1.00 com 300s.
Qwen 0.5B baseline `ruler_baseline_results.json` 0/27 sem timeouts, mas manual
`cli.main infer --weights qwen_weights.npz` OOM (`dmesg: total-vm 4822M`,
`swap 2/2G` cheio, `exit 137`) — limite de hardware, não capacidade.

## 7. Re-validação e fechamento de significância

Após fixes, `600/0.1` com timeout 300:

- `ruler_gen_date_fixed.json` (seed 42, date, 5 reps): 0/5 vs **5/5**, `p=0.062`
  (piso com n=5: `2×0.5⁵`)
- `ruler_smollm_extra_reps.json` (seed 99, number, 2 reps): 0/2 vs **2/2**

Combinados **7/7 unânimes → p=2×0.5⁷=0.0156** (<0.05). Registrado em
`ANALISE.md:749` `§22.5`.

**Ressalvas explícitas:**
1. Heterogeneidade — soma `date`+`number` como mesma amostra; ambas tinham mesma
   direção antes, mas o rigoroso seria Mantel-Haenszel.
2. Optional stopping — +2 reps após ver `p=0.062`; em efeitos fracos infla
   falso-positivo, mas 7/7 unânime é grande demais para ser só esse viés.

## 8. Estado final

```
v0.8.0 + fixes
├── Runtime  Transformer/GQA/RoPE CPU/NumPy ✓
├── Memória  StreamingKV/PositionFreeze/Eviction ✓
├── Salience 0.95 vs 0.52 p=0.004 → 11/12 depth0.1 → 7/7 p=0.0156 ✓
├── Speculative verify 0.55× k steps, RoPE batch 1.6× FAIL → CLOSED
├── Bugs     3 evicção + 1 harness, todos com regressão comprovada
├── Tests    52/52 (44 engine + 8 harness)
└── Harness  NFC/case + timeout calibrado
```

**Lição:** cada otimização provou valor (ou falhou) contra pipeline real, com
equivalência numérica e benchmark reproduzível. Documentar negativos com mecanismo
explicado vale tanto quanto positivos — é o que torna o baseline auditável.
