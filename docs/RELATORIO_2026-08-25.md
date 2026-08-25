# Relatório Geral — Little Hawk 🦅
**LLM Streaming Engine em Python/NumPy puro**

> **Data:** 25/08/2026 · **Versão:** `v0.7.0` + trilhas `A/B/C/D` (commits `8cacad4` → `418b082`)  
> **Autor:** tiagouzl · **Repositório:** `github.com/tiagouzl/little-hawk` · **Licença:** MIT  
> **Hardware de referência:** Acer Aspire A515-54 (i5-10210U, 8 GB, sem GPU, Mint 21)

---

## Sumário
- [1. Objetivo do projeto](#1-objetivo)
- [2. Arquitetura do núcleo](#2-arquitetura)
- [3. Cronologia completa](#3-cronologia)
- [4. O que foi feito — detalhamento por fase](#4-detalhamento)
- [5. Métricas e benchmarks consolidados](#5-métricas)
- [6. Estado atual do código](#6-estado)
- [7. Modelos suportados](#7-modelos)
- [8. Qualidade, testes e CI](#8-qualidade)
- [9. Interface (CLI/API/Docker)](#9-interface)
- [10. Débitos tratados e pesquisa 08/2026](#10-débitos)
- [11. Lições aprendidas](#11-lições)
- [12. Próximos passos](#12-próximos)
- [Anexos — arquivos chave e referências](#anexos)

---

## 1. Objetivo

Reconstruir **do zero** a pilha de inferência autoregressiva de LLMs família LLaMA/Qwen **sem PyTorch/CUDA/frameworks**, apenas `numpy` + matemática explícita, para fins didáticos e de engenharia. Cada GEMV, RoPE, RMSNorm e evicção é visível no código — sem `autograd` ou `device` escondendo custo.

```
~4.077 linhas Python · 6 pacotes (api, cli, engine, runtime, transplants, utils)
· 30 testes verdes · 0 alocação dinâmica no cache
```

---

## 2. Arquitetura

**`engine/engine.py:12` `MultiLayerEngine` + `engine/transformer.py:8` `LlamaLayer`** — espelha `LLaMA2` (Touvron 2023):

| Componente | Implementação | Arquivo |
|---|---|---|
| **StreamingKVCache O(1)** | `S=4` sinks imutáveis + `W=508` janela circular `win_ptr % W`, sempre `512` slots, zero `append` | `transformer.py:39` |
| **Position Freeze** | `n_ctx≤512` → pos `0..n-1`; `>512` → sink `0..3`, janela `4..511`, `Q=511` (sem drift RoPE) | `transformer.py:43` |
| **RoPE `rotate_half`** | `x*c + [-x2,x1]*s` com `inv_freq=1/10000^(i/d_k)` (HF, não GPT-J interleaved) | `jit_kernels.py:72` |
| **GQA→MHA** | `np.repeat` por `ratio = n_heads/n_kv` | `transplants/smollm.py:72` |
| **SwiGLU** | `silu(gate)*up @ down` com `RMSNorm` | `transformer.py:67` |
| **Prefill batched** | GEMM `[T,d]@[d,m]` + `tril` causal, `~10–20×` TTFT | `engine.py:111` |
| **Evicção Nexus** | `engine/eviction.py:40` reservoir ponderado `R=64`, `alpha 0.9` | `eviction.py:1` |

```
Cache tradicional: token N → KV[1..N]  O(N) RAM
Little Hawk:       [sink 4 | janela 508 circular]  sempre 512, win_ptr avança
```

**Tokenizer** `runtime/tokenizer.py:30` — BPE demo + modo doador byte-level GPT-2 (`Ġ` prefix, `StreamDecoder` incremental `codecs` para `’=E2 8099` não quebrar em `âĢĻ`).

**Sampling** `runtime/inference.py:27` — `temperature/top-k/top-p/min_p/rep_penalty` com `min_p` para estabilizar `>512` toks.

---

## 3. Cronologia completa

```
76145ec 2026-03-13 feat: Little Hawk v2 — base NumPy (~1.210 linhas, cache O(1) correto)
42f58ef 2026-08-23 fix: RoPE rotate_half + StreamDecoder UTF-8 + validação pesos + API concorrente
bfb10c7 2026-08-23 test: contexto longo 1.500 toks — streaming ≡ contexto completo (ΔNLL +0.08)
c6043d1 2026-08-23 build: pyproject.toml (3.10+, deps sincronizadas)
9bb6312 2026-08-23 feat: min_p + bench automatizado (RSS/latência/NLL)
a2ade15 2026-08-23 perf: int8 rejeitado (3–4× mais lento), [V,d] contígua adotada
afa1765 2026-08-24 fix: RoPE benchmark re-sincronizado
c555bb3 2026-08-24 P3: fast_step 1.13× vs Cython 0.70× (rejeitado)
a854329 2026-08-24 ONNX 30L 6.62× teto FFN (318 MB)
3254293 2026-08-24 v0.6.0 — ONNX fp32 1.21× validado 600 steps (diff 1.5e-04, top-5 5/5)
9679b0c 2026-08-24 docs: int8/int4 quant rejeitada (top-5 1.83/5)
3ba4458 2026-08-24 v0.7.0 — prefill batched TTFT 14s→0.2s (129 toks)
428614c 2026-08-24 docs: encerra trilha perf — veredito consolidado
8cacad4 2026-08-24 fix: débitos (Docker api.server:app, prefill chunked >512, ruff, cython, API lock)
32b1a3e 2026-08-24 feat C: SmolLM2 135M/360M/1.7B + BF16 fix
418b082 2026-08-25 feat A/B/D: Nexus + ONNX contrib investigado + PESQUISA_2026-08.md
```

`git log --oneline | wc -l` → **~38** commits em 5 meses; `git ls-files *.py | wc -l` → **4.077** linhas.

---

## 4. Detalhamento por fase

### 4.1 Fundação e correções P0–P1 (até `42f58ef`)
* Cache circular + sinks + freeze validados contra HF Transformers — divergência RoPE interleaved→`rotate_half` corrigida (top-5 `1/5→5/5`).
* `StreamDecoder` byte-safe elimina `âĢĻ`.
* `load_weights()` valida shapes (`engine.py:51`) e `BPETokenizer.load_donor_vocab()` falha alto se vocab vazio.
* API: `lifespan`, `Semaphore(2)`, cancelamento cooperativo `threading.Event`, RNG por requisição.

### 4.2 P2 — Qualidade (`867956b`)
* `utils/helpers.py` 64→30 linhas, `ruff.toml` `utils/` promovido a lint-clean.

### 4.3 Performance P3 — encerrada `ANALISE.md:329` (`428614c`)
| Marco | Ganho | Decisão |
|---|---|---|
| `fast_step` Python inlined | `1.13×` (66→58ms) | mantido |
| Numba RMSNorm/SwiGLU | `~0×` (2% do passo) | rejeitado (opt-in) |
| Cython `cython_fast.pyx` 718 KB | `0.70×` (mais lento) | rejeitado |
| `scipy sgemv` | `0.17×` | rejeitado |
| int8 NumPy lm_head | mais lento | rejeitado |
| **ONNX fp32** | **1.21× decode / 6.62× FFN** | mantido (`LITTLE_HAWK_ONNX=1`) |
| ONNX int8/int4 | `1.83/5` top-5 | rejeitado |
| **Prefill batched** | **TTFT 10–20×** | mantido |

Conclusão: gargalo é `GEMV batch-1` em `OpenBLAS` — ganho veio de **mudar a forma** (prompt como GEMM) + **fusão de grafo**.

### 4.4 `v0.6.0`–`v0.7.0` (`3254293`, `3ba4458`)
* ONNX `engine/torch_model.py:102` `LittleHawkTorch` com cache circular exportável `win_ctx = where(n_win<W, arange, (wbi+win_ptr+1)%W+S)` e `pos_q = where(n_ctx≤512, n_ctx-1, 511)`. Validação `600` steps `diff 1.5e-04`.
* Prefill `engine.py:111` com `causal tril` + `RoPE` por posição; `runtime/inference.py:172` delega `prefill(ids)`; chunked `>512` via loop `step` (fix `8cacad4`).

### 4.5 Débitos restantes (`8cacad4`)
* `Dockerfile:16` `api:app→api.server:app`.
* `ruff.toml:3` `RUF034/PLR0402/F811` em `engine/onnx_engine.py:130`, `torch_model.py:15` corrigidos; `engine/runtime` mantidos fora do `format` (semi-colons) mas lint-clean verificável.
* `pyproject.toml:27` `cython` removido (POC `0.70×`), artefatos `.c/.so` limpos.
* `requirements.txt:1` sincronizado com `pyproject` (`ruff` adicionado) + header fonte da verdade.
* `api/server.py:47` `_load_lock` + doc `single-process educacional`.
* `tests/test_engine.py:236` `test_prefill_chunked_beyond_max_cap` (+1 → `30` testes).

### 4.6 Trilha C — SmolLM2 (`32b1a3e`)
* `utils/config.py:32` `smollm2-135m` (`100k`), `360m` (`960d/15h/5kv/32L/2560, 100k`), `1.7b` (`2048d/32h/32L/8192, 130k, MHA`).
* `transplants/smollm.py:40` `MODEL_CONFIGS` + `--model`, `load_safetensors()` BF16 `<<16` (fix `bfloat16 not understood`), `cli/main.py:54` choices.
* Validado `smollm2-135m 1L` `242 MB` e `infer` OK.

### 4.7 Pesquisa `docs/PESQUISA_2026-08.md:1` e trilhas A/B/D (`418b082`)
| Trilha | Descrição | Esforço | Status |
|---|---|---|---|
| **A** | Knobs `ORT 1.29` `ORT_INTRA_OP_NUM_THREADS` + `NumPy 2.4.3` (ufuncs `6×`) | horas | **Concluída** — `2.4.3`+`1.29.0` já instalados, bench `1` vs `4` threads `484→156ms` |
| **B** | Grafo ONNX `RotaryEmbedding`/`RMSNorm`/`Attention` `com.microsoft` | dias | **Investigada e cancelada** — pip wheel `1.29` não registra ops (`Fail: is not a registered function/op`), exigiria build custom; mantido `optimizer` |
| **D** | **Nexus** reservoir (`2606.23961`) | médio | **Concluída** — `engine/eviction.py:1` |

**D detalhe:** `NexusEviction` (`S=4, W=508, R=64 adaptativo`, `alpha 0.9` EMA) protege sinks + `R` recentes, anel `W-R` escolhe vítima entre `32` menores scores (reservoir). `engine/engine.py:50` `eviction="nexus"` via `LITTLE_HAWK_EVICTION`, `transformer.py:23` `slot_override` + `at` 5º retorno, `api/server.py:40` `EVICTION` em `/health`, `cli/main.py:54` `--eviction nexus`. `prefill` chunked compatível, `O(1)` preservado.

---

## 5. Métricas consolidadas

**Hardware:** `A515-54` `OMP=1` `SmolLM-135M 30L` `min_p=0.05` `600` toks pós `afa1765`:

| Métrica | Valor | Fonte |
|---|---|---|
| Pico RSS | `1.28 GB` (cache `70.8 MB` constante) | `scripts/benchmark.py` |
| p50 fill / estacionária | `134 / 249 ms` | `README.md:143` |
| ONNX decode | `91 ms` vs NumPy `110 ms` (`1.21×`) | `ANALISE.md:329` |
| Teto FFN ONNX | `6.62×` (`294→44ms` 30L) | `CHANGELOG:0.5.0` |
| Prefill `129` toks | `14s → 0.2s` | `CHANGELOG:0.7.0` |
| `W_lm_t [V,d]` | `-1.4ms` (`2e-7` erro) | `ANALISE.md:11` |
| Long-CTX `1500` | `ΔNLL +0.08`, `win_ptr` `5` wraps, `0` realocações | `ANALISE.md:8` |

`thermal throttling` ±60% — comparar sempre `A/B` intercalado.

---

## 6. Estado atual

```
little-hawk/
├── api/server.py        # FastAPI SSE, lifespan, Semaphore(2), timeout 300s, EVICTION
├── cli/main.py          # infer/transplant/api + --eviction/--min-p
├── engine/
│   ├── engine.py        # MultiLayerEngine + prefill chunked + eviction
│   ├── transformer.py   # LlamaLayer + slot_override
│   ├── eviction.py      # FIFO/Nexus (novo)
│   ├── jit_kernels.py   # _rope_numpy rotate_half
│   ├── torch_model.py   # LittleHawkTorch (onnx export)
│   └── onnx_engine.py   # OnnxEngine (prefill sequencial)
├── runtime/             # tokenizer + inference (StreamDecoder, Sampler min_p)
├── transplants/         # smollm.py (4 modelos) + qwen.py (BF16)
├── utils/               # helpers limpos
├── tests/               # 30 testes (prefill 3, position freeze 2)
├── scripts/             # benchmark/profile/onnx
└── docs/                # ANALISE.md 19 seções + PESQUISA_2026-08.md
```

`pyproject.toml:6` `0.7.0` `requires-python>=3.10` + extras `dev/jit/onnx/equiv`.

---

## 7. Modelos suportados

| Modelo | HF ID | d | heads (Q/KV) | L | intermediate | rope | .npz (4L) |
|---|---|---|---|---|---|---|---|
| `smollm-135m` | `HuggingFaceTB/SmolLM-135M` | 576 | 9/3 | 30 | 1536 | 10k | `~64 MB` |
| `smollm2-135m` | `HuggingFaceTB/SmolLM2-135M` | 576 | 9/3 | 30 | 1536 | 100k | `~64 MB` |
| `smollm2-360m` | `HuggingFaceTB/SmolLM2-360M` | 960 | 15/5 | 32 | 2560 | 100k | `~177 MB` |
| `smollm2-1.7b` | `HuggingFaceTB/SmolLM2-1.7B` | 2048 | 32/32 | 24 | 8192 | 130k | `~1074 MB` |
| `qwen2.5-0.5b` | `Qwen/Qwen2.5-0.5B` | 896 | 14/2 | 24 | 4864 | 1M | `~900 MB` |

Todos com `tokenizer.json` embutido no `_meta.json`, `vocab 49152` (Qwen `151936`), `GQA` expandido.

---

## 8. Qualidade

* `30` testes: `StreamingCache` (shape/ptr/finite), `WeightValidation` (roundtrip), `Sampling` (determinístico/rep), `MinP` (tail), `LMHead` (`[V,d]`), `PositionFreeze`, `Prefill` (3, inclui chunked).
* `ruff check` ✅ (`RUF034/PLR0402/F811` corrigidos), `format --check` `20` files ✅ (engine/runtime excluídos do format por estilo `;`).
* `py_compile` ✅, `pytest -q` `30 passed` `~43s` (inclui equivalência HF opcional).
* `CI .github/workflows/ci.yml:7` `check→test→lint→fmt-check` em `3.11`.

---

## 9. Interface

**CLI** `cli/main.py:20`:
```bash
python -m cli.main infer --prompt "..." --eviction nexus --min-p 0.05
python -m cli.main transplant --model smollm2-360m --layers 32
python -m cli.main api --weights x.npz
# legada: python little_hawk_cli.py --weights x.npz --prompt "..." → infer
```

**API** `api/server.py:1` `lifespan` + `Semaphore` + `LITTLE_HAWK_TIMEOUT_SECS=300` + `EVICTION`:
```bash
curl -N -X POST localhost:8000/generate -d '{"prompt":"...","max_tokens":32,"min_p":0.05}'
# /health → {status, mode, max_concurrency, timeout_secs, eviction}
```

**Docker** `Dockerfile:6` `python:3.11-slim` `CMD ["uvicorn","api.server:app","--host","0.0.0.0","--port","8000"]` (fix `api:app`), pesos via volume + `LITTLE_HAWK_WEIGHTS`.

---

## 10. Débitos e pesquisa

* Todos os débitos de `ANALISE.md:3` (P0–P7) fechados; `§19` trilha perf encerrada com veredito honesto.
* Pesquisa `PESQUISA_2026-08.md` mapeou `ORT 1.29` contrib, `NumPy 2.4/2.5`, `SmolLM2/3`, `Qwen3.5 DeltaNet`, e comparativo `H2O/SnapKV/Quest/Nexus` — validou `StreamingLLM` como baseline `2026`.

---

## 11. Lições

1. **GEMV batch-1 é teto em CPU** — `OpenBLAS` já ótimo; jittar GEMV = `6×` mais lento.
2. **Mudar a forma > acelerar o passo** — `prefill` GEMM e `ONNX` fusão são os únicos ganhos reais.
3. **Quantização sem calibração destrói `top-k`** — `DynamicQuantizeLinear` outlier `200×` → erro `11.7`.
4. **`diff` de logits engana** — métrica correta é `top-k` ao longo de `600` steps.
5. **Position freeze é frágil** — reusar `ONNX` stale de `/tmp` divergiu em `step 513`.

---

## 12. Próximos passos (fora do escopo atual)

* `SmolLM2-1.7b` full `24L` bench e `RULER` com `Nexus` vs `FIFO` em `>600` passos.
* `ONNX` batched `prefill` + `past` dinâmico (requer `torch.export` novo).
* `int8` estático com `calibration dataset` + `QDQ` + `VNNI` (fora do `NumPy`).
* `free-threaded` `3.13t` para API SSE multi-cliente.

---

## Anexos

**Commits chave:** `76145ec→418b082` (`38` commits) — ver `git log --oneline`.

**Arquivos chave:**
* `engine/engine.py:111` `prefill`, `engine/transformer.py:23` `attn_step`, `engine/jit_kernels.py:72` `RoPE`, `engine/eviction.py:40` `Nexus`, `runtime/inference.py:104` `generate`, `transplants/smollm.py:40`, `api/server.py:71` `load_model`.

**Referências:**
* `StreamingLLM` Xiao 2023, `LLaMA2` Touvron 2023, `RoPE` Su 2021, `Nexus 2606.23961`, `Quest 2406.10774`, `SmolLM2` HuggingFaceTB, `ORT` `github.com/microsoft/onnxruntime/releases`, `NumPy` `numpy.org/doc/stable/release/`.

---

*Relatório gerado automaticamente a partir de `ANALISE.md`, `CHANGELOG.md`, `README.md`, `PESQUISA_2026-08.md` e `git log` em `25/08/2026`. Para reproduzir métricas: `OMP_NUM_THREADS=1 venv/bin/python scripts/benchmark.py --gen-tokens 600 --json bench.json`.*
