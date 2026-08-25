# Análise Técnica — Little Hawk 🦅

> ⚠️ **Documento histórico**: as seções 1–6 descrevem o estado do projeto ANTES das correções
> (snapshot de 22/08/2026). Os problemas nelas listados que já foram resolvidos estão marcados
> na seção 7 "Correções aplicadas". Consulte o §8 e o §9 para o estado atual validado.

> Auditoria do código verificada contra execução real (modo demo + pesos SmolLM-135M).
> Data: 22/08/2026 · Commit analisado: `76145ec` (`feat: Little Hawk v2`)

---

## 1. Visão geral

Motor experimental de inferência LLM autoregressivo escrito **do zero em Python/NumPy** (~1.210 linhas), sem PyTorch, implementando a arquitetura da família LLaMA/Qwen:

- inferência autoregressiva streaming;
- StreamingKVCache O(1) — cache circular de tamanho constante (4 slots *sink* + janela de 508);
- RoPE com *position freeze* na fase estacionária;
- GQA expandido para MHA por repetição de grupos;
- MLP SwiGLU + RMSNorm;
- tokenizer BPE próprio + integração com `tokenizer.json` nativo (Rust);
- transplante de pesos reais: SmolLM-135M e Qwen2.5-0.5B → `.npz`;
- CLI com telemetria em tempo real, API FastAPI (SSE), Docker, Makefile e CI.

**Estado atual verificado:**

| Verificação | Resultado |
|---|---|
| `py_compile` (4 scripts) | ✅ passou |
| Modo demo (pesos aleatórios) | ✅ executou e gerou tokens |
| Pesos reais SmolLM (30 camadas, CPU) | ✅ gerou inglês coerente, cache constante em 512 slots |
| `ruff` | ⚠️ não instalado no ambiente |
| Git | ✅ `.npz` fora do versionamento; 2 `_meta.json` trackeados; ZIP não rastreado |

O projeto funciona, mas está mais próximo de um **protótipo técnico/educacional** do que de uma implementação pronta para produção.

---

## 2. Pontos fortes

- **Proposta educacional excelente**: toda a matemática da inferência exposta diretamente em `little_hawk_cli.py`.
- **Core do cache bem implementado**: slots sink imutáveis, janela circular modular, position freeze — fiel ao paper StreamingLLM (Xiao et al., 2023).
- **Documentação de alto nível** (`README.md`): diagramas da arquitetura do cache, benchmarks honestos em hardware modesto, referências acadêmicas.
- **Modo demo sem download** valida o pipeline completo imediatamente.
- **Transplantes separados por modelo** (SmolLM e Qwen), com tratamento correto das diferenças arquiteturais (bias Q/K/V do Qwen, rope_base YaRN, lm_head não atado).
- **API move o cálculo CPU-bound para thread**, sem bloquear o event loop (api.py:93).
- Tooling completo: Dockerfile com volume para pesos, Makefile, CI, ruff configurado, demo HTML de SSE.
- Fallbacks robustos (busca ampla do tokenizer no cache HF, modo demo).

---

## 3. Riscos técnicos

### 3.1 Ausência de testes numéricos 🔴 (crítico)

O CI valida apenas se os scripts compilam. Não há teste comparando logits contra PyTorch/Transformers ou implementação de referência. Divergências subtis em RoPE, RMSNorm, posições congeladas, expansão GQA, ordem dos resíduos, tokenizer ou transposição de pesos produzem texto incorreto **sem qualquer erro visível**.

### 3.2 Decode incremental corrompe UTF-8 🔴 (reproduzido)

Execução real com pesos SmolLM gerou `âĢĻ` no lugar de `'`: decodificar token a token com `decode([nid])` quebra caracteres multibyte divididos entre tokens.

- Locais: `little_hawk_cli.py:429`, `api.py:86`
- Fix: buffer de ids + re-decode do sufixo desde a última fronteira válida (como `TextIteratorStreamer` do HF).

### 3.3 RoPE interleaved vs rotate_half 🟠

O motor pareia dimensões `(0,1),(2,3)…` (estilo GPT-J, `little_hawk_cli.py:260`); LLaMA/Qwen2/SmolLM usam metades (`rotate_half`). Os transplantes **não permutam** os pesos Q/K → rotação divergente do treino. O modelo gera texto plausível porque o conteúdo (V) domina, mas a qualidade provavelmente está degradada. O teste de equivalência numérica (3.1) detectaria isto imediatamente.

### 3.4 Transplante Qwen incompleto para ambientes novos 🟠

`little_hawk_transplant_qwen.py` não baixa nem empacota o `tokenizer.json`. Pior: `qwen_weights_meta.json` (253 bytes) **não embute vocabulário** — ao contrário do meta do SmolLM (1 MB, que embute). Em máquina limpa, `load_donor_vocab` termina com vocab vazio e `_encode_donor` mapeia tudo para UNK **silenciosamente**, gerando lixo sem crash (`little_hawk_cli.py:139,189-192`).

Fix: baixar/packar tokenizer no transplant Qwen + embutir vocab no meta (padrão já usado pelo SmolLM).

### 3.5 API sem controle de concorrência 🟠

Cada requisição tem cache próprio, mas todas partilham modelo e CPU:

- saturação de CPU e latência imprevisível;
- threads produtoras são daemon e continuam inferindo após desconexão do cliente (api.py:108) — sem cancelamento cooperativo;
- sampling usa RNG global do NumPy → resultados não-determinísticos sob concorrência;
- sem autenticação, rate limit ou timeout.

### 3.6 Validação de pesos insuficiente 🟡

`np.load(..., allow_pickle=False)` está correto, mas `engine.load_weights` confia cegamente nas dimensões. O `validate()` dos transplants só checa presença de chaves, nunca shapes. Arquivo parcialmente corrompido falha apenas durante a inferência.

### 3.7 Inconsistências menores 🟡

- README diz 30 camadas; default do transplant SmolLM é 4 (`DEFAULT_N_LAYERS=4`, little_hawk_transplant.py:40) — reduzido e não sinalizado.
- README diz que `_meta.json` não é versionado, mas ambos estão trackeados no git.
- CI: `make fmt --dry-run` (ci.yml:34) apenas simula o make — não valida formatação. Deveria ser `make fmt-check`.
- Depreciações FastAPI/asyncio: `@app.on_event("startup")` (api.py:120) → `lifespan`; `asyncio.get_event_loop()` (api.py:96) → `get_running_loop()`.
- Estrutura do README desatualizada (não menciona `api.py`, Dockerfile, Makefile).
- Lixo local: `.git_backup_before_squash/`, `Repositório-clonado-github/`, diretórios órfãos `little_hawk/` e `transplants/` (só `__pycache__`).

### 3.8 Tokenizer demo limitado ℹ️

O BPE próprio usa regex simplificada e lowercase — adequado para demonstração, não equivalente a um tokenizer de produção.

---

## 4. Observação: versão modular no ZIP

O arquivo não rastreado `Repositório-clonado-github/little-hawk-main.zip` contém uma versão com arquitetura de diretórios mais modular:

```
little-hawk-main/
├── api/server.py
├── cli/main.py
├── engine/engine.py
├── engine/transformer.py
├── examples/demo.py
└── docs/
```

Recomendação: usar essa estrutura como **alvo da refatoração** em vez de reescrever do zero.

---

## 5. Prioridades recomendadas

| # | Ação | Resolve |
|---|---|---|
| 1 | Testes de equivalência numérica vs HF Transformers | §3.1, §3.3 (de uma vez) |
| 2 | Empacotar tokenizer no fluxo Qwen + vocab embutido no meta | §3.4 |
| 3 | Fix decode streaming UTF-8 (buffer de ids) | §3.2 |
| 4 | Validar shapes no `load_weights` | §3.6 |
| 5 | Concurrency limiter + cancelamento de stream na API | §3.5 |
| 6 | Corrigir CI (`fmt-check`) + testes unitários básicos (roundtrip tokenizer, shapes do cache, position freeze, sampling com seed) | §3.7 |
| 7 | Refatorar para estrutura modular (engine/cli/api) usando o ZIP como referência | manutenibilidade |

---

## 6. Veredito

A base é sólida, executável e didaticamente valiosa — o núcleo (cache O(1) + attention sinks + position freeze) está corretamente implementado e validado em execução real. As lacunas principais são **confiança numérica** (sem comparação com referência, bugs como o de RoPE passam invisíveis) e **robustez de empaquetamento** (tokenizer Qwen). Ambas são endereçáveis com esforço moderado; os itens 1–3 da tabela acima têm o melhor retorno por hora investida.

---

## 7. Correções aplicadas (22/08/2026)

| Prioridade | Correção | Resultado |
|---|---|---|
| P1 | `tests/test_equivalence.py` — logits vs SmolLM-135M real | ✅ mediana rel < 1e-3 |
| — | **RoPE corrigido**: interleaved (GPT-J) → `rotate_half` (HF) | ✅ top-5 overlap 1/5 → **5/5**; geração visivelmente mais coerente em EN e PT |
| P2 | Transplant Qwen baixa tokenizer + embute vocab no meta; loader falha alto se vocab vazio | ✅ Qwen funciona sem cache HF |
| P3 | `StreamDecoder` byte-safe no CLI e na API | ✅ zero mojibake (`âĢĻ`) |
| P4 | Validação de chaves/shapes/consistência no `load_weights` | ✅ npz corrompido rejeitado com mensagem clara |
| P5 | Lifespan, semáforo (`LITTLE_HAWK_MAX_CONCURRENCY`), cancelamento cooperativo por desconexão, RNG por requisição | ✅ smoke test ok |
| P6 | CI: `fmt-check` real + step `make test`; suíte pytest com 22 testes | ✅ verde local |
| P7 | Refatoração modular: pacote `little_hawk/` (`tokenizer`, `engine`, `inference`); CLI vira entry point fino com re-exports; API e testes importam do pacote | ✅ suíte + equivalência verdes pós-move |

Pendências: ~~sincronizar benchmarks do README com o RoPE corrigido~~ → ✅ em 24/08/2026 (`afa1765`, `bench_600.json`): p50 enchimento 134 ms / estacionária 249 ms (1 thread, SmolLM-135M, 600 tokens), cache 70.8 MB, perfil 13 ms/camada.

---

## 8. Validação de contexto longo (23/08/2026)

Teste de 1.500 tokens com pesos reais SmolLM-135M (`scripts/test_long_context.py`) + sondas contra o HF Transformers:

| Verificação | Resultado |
|---|---|
| Buffers do cache idênticos após 994 evicções | ✅ zero realocação (O(1) real) |
| `win_ptr` wraps módulo 508 | ✅ 5 wraps corretos |
| PPL streaming vs **contexto completo HF** (texto único, 1663 tokens) | ✅ Δ ≤ +0.08 NLL até 1200 de profundidade |
| Fase de enchimento (logits vs HF, passos 100/300/500) | ✅ mediana ~1e-4 |
| Geração livre >500 tokens | ⚠ drift semântico → colapso para tokens raros |

Conclusões:
- A matemática da fase estacionária está **correta** — perplexidade acompanha o contexto completo dentro de +0.003 a +0.076 NLL, exatamente o comportamento documentado no paper StreamingLLM. A divergência de logits vs janela recalculada é esperada: K/V no streaming é computado uma vez com o contexto da época (definição do método).
- O colapso semântico em geração livre longa também ocorre no HF full-context (controle), só que em palavras genéricas em vez de bytes. É fraqueza do modelo de 135M + compounding de amostragem, não bug do motor.
- Mitigação recomendada para gerações longas: `--temperature 0.5` ou menor; futuro: sampler `min_p`.

---

## 9. Empacotamento e controles operacionais (23/08/2026)

| Item | Correção |
|---|---|
| `setup.py` defasado (`python_requires>=3.8` incompatível com PEP 604; deps incompletas) | Substituído por `pyproject.toml`: `requires-python>=3.10`, deps sincronizadas, extras `dev`/`equiv`, `pip install -e .` validado |
| Erro bruto de traceback ao carregar `.npz` inválido no CLI | `cli/main.py` captura `ValueError` do `load_weights` e sai com mensagem clara |
| API sem limite de prompt/timeout | `prompt` limitado a 8000 chars; timeout global `LITTLE_HAWK_TIMEOUT_SECS` (padrão 300s) emite evento de erro SSE e cancela a inferência |

Observações aceitas como escopo do projeto: estado global do modelo (processo único), ausência
de auth/rate-limiting (demo/educacional), tokenizer demo simplificado. Documentados no README.

---

## 10. Sampler min_p e benchmarks automatizados (24/08/2026)

| Item | Resultado |
|---|---|
| `min_p` no `Sampler` (config, CLI `--min-p`, API `"min_p"`) | ✅ mitiga o drift de gerações longas: 700 tokens com `min_p=0.05` mantêm ~99.9% ASCII (sem min_p: colapso para ~17% na segunda metade) |
| `scripts/benchmark.py` | ✅ memória (pico RSS, cache 70.8 MB constante, buffers reutilizados), latência p50/p95 por fase, NLL teacher-forced + baseline opcional vs contexto completo (`--compare-hf`), saída `--json` |
| Testes | ✅ 23/23 (2 novos para min_p) |

Numba foi adicionado como extra opcional em `[jit]`, com fallback gracioso para NumPy puro. Os benchmarks mostraram que RMSNorm/SwiGLU representam apenas uma pequena parte do decode batch-1; o GEMV do `lm_head` continua sendo o principal gargalo e é o próximo alvo de otimização.

---

## 11. Investigação lm_head int8 vs fp32 (24/08/2026)

Motivação: profiling anterior indicava ~75% do tempo no lm_head. **Medição controlada refutou**: com pesos reais, as 30 camadas consomem ~251 ms (96%) e o lm_head ~11 ms (4%) por token. O gargalo real é o dispatch Python/BLAS distribuído pelas camadas, não um único GEMV.

Microbenchmark (D=576, V=49152, p50 aquecido):

| Variante | p50 | med_rel erro | top-5 |
|---|---|---|---|
| fp32 `x@W` (atual, view não-contígua) | 13.8 ms | 0 | 5/5 |
| fp32 `[V,D]` contígua, 2 threads BLAS | **8.8 ms** | 2e-7 | 5/5 |
| int8 por coluna, upcast bloco 8192 | 33.5 ms | 7.4e-3 | 5/5 |
| int8 bloqueado 2048 | 59.4 ms | 7.4e-3 | 5/5 |

Decisões pelo critério acordado ("só entregar se ganho superar claramente a perda"):
- ❌ **int8 rejeitado**: 3–4× mais lento que fp32 em NumPy puro (upcast por token elimina o ganho de banda; sem VNNI via NumPy) e ainda perde precisão.
- ✅ **Orientação `[V,d]` contígua adotada** (`W_lm_t`, A/B intercalado: −1.4 ms/token, ~14% do head, custo numérico 2e-7). Ganho total do step: <1% — honestidade acima de marketing.
- ⚠️ Medições absolutas neste hardware oscilam ±60% por thermal throttling — comparar sempre A/B intercalado no mesmo processo.

Caminho real para aceleração expressiva (fase futura): fundir/executar as 30 camadas em menos despachos (kernel único por camada ou grafo compilado), atacando os ~250 ms de overhead estrutural.

---

## 12. P2 — Qualidade: utils promovido e dead code removido (24/08/2026 — 867956b)

| Item | Antes | Depois |
|---|---|---|
| `utils/helpers.py` | 64 linhas, 4 funções mortas (`find_file_in_cache` chamava `HfApi.list_repo_files` com rede, duplicava `transplants/qwen.py:84`; `load/save_json_safe` nunca usadas) | 30 linhas, mantém `ensure_dir`/`format_bytes`/`validate_weights_file` com `tuple[bool,str]` |
| `utils/__init__.py` | re-exportava `find_file_in_cache` não usado | re-exporta apenas símbolos usados |
| `ruff.toml` `extend-exclude` | 7 entradas incluindo `utils/` | 6 entradas — `utils/` agora lintado/formatado |
| `utils/colors.py` `config.py` | shebang `EXE001`, `Dict`/`Optional` legados | `dict`/`X|None`, sem shebang |

Verificação: `ruff check` ✅, `ruff format --check` ✅, `27 passed`.

## 13. P3 — Investigação fusão de kernels por camada (24/08/2026)

Hipótese: njit de `attn_step` inteiro eliminaria dispatch Python (≈15 calls ×30L = 450 calls/token, ~250-400ms).

Microbenchmark (d=576, inter=1536, 1 thread, N=200):
- BLAS `x@W_gate` [1,576]@[576,1536]: **401 µs**
- njit loop manual equivalente: **2.448 µs** → **6.1× mais lento**

Conclusão idêntica à de `ANALISE.md:11` (int8): GEMVs são kernel dominante e OpenBLAS é ótimo; jittar GEMVs perde. RMSNorm/SwiGLU já jittados são ~2% do passo (`engine/jit_kernels.py`). Ganho real só viria de eliminar overhead Python sem perder BLAS (inlining manual, Cython, ou grafo compilado) — fora do escopo NumPy puro. Mantido como opt-in apenas para partes não-GEMV.


---

## 14. P3 — Cython compilado e ONNX Runtime (24/08/2026 — c555bb3+)

**Cython:**
- `engine/cython_fast.pyx` compilado via `python setup.py build_ext --inplace` (Cython 3.3, gcc -O3, 718 KB .so)
- Bench 30L SmolLM 20 tok, 1 thread: `orig 66.7 ms` → `fast_step 58.5 ms (1.13×)` → `cython_fast 83.5 ms (0.70×)` — Cython puro com objetos Python é **mais lento** (overhead de boxing). Ganho real vem do inlining Python (`fast_step`), não da compilação sem BLAS C-API. Próximo passo: `cimport numpy` + `scipy.linalg.blas.sgemv` via C-API.

**ONNX Runtime:**
- `scripts/onnx_export.py` exporta stack Torch espelhado (RMSNorm+FFN) via `torch.onnx.export`
- Bench 1L: NumPy 9.56 ms → ONNX 3.14 ms **(3.04×)**
- Bench 30L (318 MB): NumPy 294 ms → ONNX 44.4 ms **(6.62×)** — kernels C++ fusion + MKL superam OpenBLAS dispatch.
- Limitação POC: sem KV cache/RoPE/attention (só FFN), mas prova que ONNX é caminho para 2-4× real em inferência completa. Próximo: `past_key_values` dinâmicos + RoPE.

**Decisão:** manter `fast_step` como opt-in Python (1.13×) e documentar ONNX como trilha de aceleração futura; Cython sem BLAS C-API rejeitado (ganho negativo).

## 15. P3 — Cython BLAS e ONNX full (24/08/2026 — continuação)

**Cython BLAS C-API:**
- `scipy.linalg.blas.sgemv` bench: NumPy `x@W` 613µs vs scipy `sgemv` 3561µs **(5.8× mais lento)** — mesmo padrão de `numba` (6.1×). OpenBLAS via NumPy já é ótimo para batch-1; C-API não ganha.
- `cython_fast.pyx` compilado 83ms vs `fast_step` 58ms (0.70×) — overhead de boxing Python. **Mantido `fast_step` Python inlined como melhor custo/benefício (1.13×)**.

**ONNX full:**
- `scripts/onnx_full.py` esboça export com KV cache + RoPE via `torch` (requer `torch.export` + `onnx>=1.18`). POC FFN já provou teto **6.6×** (30L 294→44ms). Próximo: `past_key_values` dinâmicos.

**Conclusão P3:** NumPy puro atingiu teto; aceleração real (>2×) só via ONNX Runtime com graph fusion — trilha recomendada para `v0.5.0`.

## 16. ONNX full implementado e validado — v0.6.0 (24/08/2026)

**Grafo completo exportável (`engine/torch_model.py`):**
- `LittleHawkTorch` espelha o motor NumPy 1:1 (embed → 30×[RMSNorm→QKV+RoPE→SDPA→Wo→residual→RMSNorm→SwiGLU] → norm → lm_head) com pesos embutidos (705 MB opset 17).
- Cache circular + position freeze **exportáveis**: slot `where(n_ctx≤4, n_ctx-1, 4+win_ptr)`; janela `win_ctx = where(n_win<508, arange(4,512), (wbi+win_ptr+1)%508+4)[:n_win]` — slice dinâmico por tensor funciona no ONNX (opset 17); `pos_q` congelado em 511.
- Exportador: **legado obrigatório** (`dynamo=False`) — o novo `torch.export` falha no `arange` dependente de input.

**Validação end-to-end (600 steps, stream argmax compartilhado vs NumPy):**
- diff máx logits **1.5e-04** · top-1 **12/12** checkpoints · top-5 **5.00/5**
- Step 513 (1ª sobrescrita da janela circular): diff **1.8e-05** ✅ — prova o caminho estacionário.
- 🐛 Pegadinha encontrada: reusar export antigo de `/tmp` (grafo com janela linear pré-fix) reproduz divergência exata a partir do step 513 — sempre re-exportar após mudar `torch_model.py`.

**Integração:** `LITTLE_HAWK_ONNX=1` ativa `OnnxEngine` em `get_engine()` (CLI/API), fallback NumPy automático.

**Bench (OMP=1):** ONNX 91 ms/step vs NumPy 110 ms/step (**1.21×**). Ganho modesto porque o grafo inclui embed/lm_head GEMV [49152×576] que domina fora do FFN; o teto de fusão continua sendo os 6.62× do bloco FFN puro.

**Próximos passos possíveis:** quantização int8 do grafo (onnxruntime dynamic quantization), batch>1, ou GraMA-style kernels custom.

## 17. Quantização int8/4 do grafo ONNX — avaliada e REJEITADA (24/08/2026)

**Protocolo:** loop 600 tokens vs NumPy (stream argmax compartilhado), OMP=1, checkpoints {1..600}.

| Variante | Tamanho | ms/step | Speedup | top-1 | top-5 médio |
|---|---|---|---|---|---|
| fp32 (v0.6) | 706 MB | 91 | 1.21× | 12/12 | 5.00/5 |
| `quantize_dynamic` u8s8 full (per-channel) | 179 MB | 72 | 1.52× | 11/12 | 1.83/5 ❌ |
| idem, exceto lm_head | 264 MB | 76 | 1.43× | 11/12 | 1.83/5 ❌ |
| idem, só FFN | 468 MB | 83 | 1.32× | 11/12 | 3.08/5 ❌ |
| `MatMulNBits` int8 weight-only block128 | 362 MB | 210 | **0.50×** ❌ | 12/12 | 4.58/5 |
| `MatMulNBits` int4 weight-only block128 | 200 MB | 74 | 1.42× | 11/12 | 2.75/5 ❌ |

**Causa raiz da degradação da quantização dinâmica (provado em micro-grafo):**
- `DynamicQuantizeLinear` usa escala min/max por tensor em **ativações u8** — um único outlier 200× na ativação explode o erro p99 de `4e-03` para `11.7` (resolução dos valores normais vira range/255).
- No modelo real: p50 |diff| nos logits ~9 (48080/49152 dims com diff>1) — a distribuição inteira se desloca; top-1 sobrevive porque o topo é destacado, mas top-k/top-p ficam inutilizados.
- `per_channel=True` no toolchain testado nem chegou a ajudar (erro relativo 22% até em MatMul isolado com X~N(0,1)).

**Por que o weight-only não salva:** qualidade boa (top-5 4.58), mas o kernel `MatMulNBits` paga dequant/expansão por step e fica **2× mais lento que NumPy OpenBLAS fp32** em GEMV batch-1 [1,576]@[576,N] — o formato brilha em batch/prefill, não em decode single-token.

**Decisão:** manter **fp32 como único backend ONNX** (v0.6). Rejeitado pelo mesmo critério da v0.4.0 (int8 lm_head em NumPy). Para ganho real futuro seria preciso: static calibration com dataset representativo + QDQ + CPU VNNI, ou kernels estilo GGUF/llama.cpp — fora de escopo.

**Lição metodológica:** diff absoluto de logits engana (topo destacado resiste); a métrica correta para aceitar/rejeitar quantização em pipeline de geração é top-k match contra o referencial numérico ao longo de um stream longo.

## 18. Prefill batched — TTFT 10–20× menor (24/08/2026 — v0.7.0)

**Problema:** o TTFT (time-to-first-token) era `T × ~110 ms` porque o prompt entrava token a token pelo mesmo hot path do decode (`step()` otimizado para GEMV batch-1). Prompt de 129 tokens ≈ 14 s antes da primeira palavra gerada.

**Solução (`MultiLayerEngine.prefill`):** um único forward batched do prompt na fase fill (T ≤ max_cap = 512):
- RMSNorm/SwiGLU/QKV/Wo viram GEMMs `[T,d]@[d,m]` que o OpenBLAS explora muito melhor que GEMVs;
- atenção com máscara causal `tril(T,T)` + RoPE por posição absoluta (`_rope_numpy` já suportava T>1);
- escrita no cache direto em `kc[0,:,:T,:]` (slots fill são sequenciais) e `win_ptr=(T-S)%W` idêntico ao loop;
- lm_head aplicado só ao último token.

**Validação:** prefill == steps sequenciais para T ∈ {1,2,10,100,300,512}: diff logits/cache ~1e-05, win_ptr igual; coberto por `TestPrefill` (2 testes). E2E CLI: 129 tokens de prompt + 12 gerados em **1.6 s total** (~14 s antes).

**Limites:** acima de max_cap o excedente segue sequencial (chunks estacionários exigiriam máscara circular por rank de recência — possível trabalho futuro). OnnxEngine ainda não tem grafo batched; usa fallback automático.

## 19. Trilha performance — ENCERRADA (24/08/2026)

**Veredito final do motor Little Hawk 135M, CPU desktop, batch-1:**

| Marco | Ganho | Status |
|---|---|---|
| `fast_step` Python inlined | 1.13× | mantido |
| Numba JIT RMSNorm/SwiGLU | ~0× (2% do passo) | rejeitado |
| Cython compilado | 0.70× | rejeitado |
| BLAS C-API (scipy sgemv) | 0.17× | rejeitado |
| int8 NumPy (lm_head) | mais lento | rejeitado |
| **ONNX Runtime fp32 (opt-in)** | **1.21× decode / teto FFN 6.62×** | **mantido** |
| ONNX int8/int4 quantizado | rápido mas destrói top-k; weight-only lento em GEMV | rejeitado |
| **Prefill batched** | **TTFT 10–20× menor** | **mantido** |

**Conclusão:** o ganho real não veio de acelerar o decode single-token (NumPy/OpenBLAS já opera no teto prático para GEMV batch-1 em CPU), e sim de (1) um segundo backend com graph fusion para quem puder pagar os 705 MB e as dependências `[onnx]`, e (2) mudar a *forma* do problema — processar o prompt como GEMM batched em vez de GEMV sequencial. Trilha encerrada sem dívida técnica conhecida: tudo opt-in, fallback automático, validação numérica documentada.

Trilhas futuras possíveis (fora do escopo atual): prefill estacionário por chunks (>512 tokens), grafo ONNX batched/prefill, kernels estilo GGUF, ou execução em GPU.

## 20. Trilha D — Evicção Nexus: bug de leitura encontrado e corrigido (25/08/2026)

**Bug:** a implementação inicial escrevia novos tokens nos slots-vítima do anel `[4..447]`
mas a LEITURA (`attn_step`) continuava computando `win_ctx=(wbi+win_ptr+1)%W+S` como FIFO.
Consequências: zona protegida `[448..511]` congelava tokens velhos (nunca mais escrita),
tokens novos churnavam no anel, e a atenção lia uma janela rotativa que não correspondia
à recência real. Resultado: RULER 512/0.5 → fifo 0.50-0.67, nexus **0.00**.

**Fix (`engine/eviction.py`, `engine/engine.py:104`, `engine/transformer.py:23`):**
`NexusEviction.order` mantém a lista explícita de slots vivos em ordem de recência
(fonte única para escrita E leitura); `sync_after_fill(T)` sincroniza após o prefill
batched; `ctx_array()` alimenta `ctx_override` no `attn_step` (posições por rank,
position freeze preservado); vítima sai só do anel `order[:W-R]`, token novo entra
como mais recente na cauda.

**Validação:** 30/30 testes verdes; invariantes `len(order)==min(n_ctx-S,W)` sem
duplicatas através de fill+estacionária; prefill chunked FIFO diff 3.6e-07.

**Comparação powered pareada** (8 prompts idênticos, in-process, 135M, ctx nominal 512 ≈ 934 toks reais, agulha depth 0.5):

| Política | Acertos | Observação |
|---|---|---|
| FIFO | 5/8 (0.62) | garante janela dos últimos 508 |
| **Nexus pós-fix** | **8/8 (1.00)** | reservoir ponderado por atenção |

Delta pareado +0.38 (3 discordâncias, todas a favor do Nexus; McNemar exato p=0.25 —
direção consistente, n ainda pequeno para significância). Pré-fix o mesmo Nexus fazia
0/3 nos draws do bench — a variância entre prompts é grande; comparações com n≤3 por
política não separam sinal de ruído neste setup.

**Lição metodológica:** evicção baseada em atenção passada (EMA α=0.9) tem horizonte
efetivo ~10 steps; retém bem tokens que atraem atenção repetida, e o reservoir com
proteção de cauda recente manteve agulhas single-mention melhor que FIFO neste teste.
Para veredito definitivo: ≥20 prompts pareados e depths {0.1, 0.9}.

**Status trilha D:** mecânica validada, resultado preliminar favorável ao Nexus.
Mantido opt-in (`--eviction nexus` / `LITTLE_HAWK_EVICTION=nexus`).

### 20.1 Sweep powered pareado — veredito final da trilha D (25/08/2026)

Correção ao resultado preliminar do §20. O sweep pareado completo (`bench_ruler_eviction.py`,
desenho pareado real: mesmo prompt sob ambas as evicções + McNemar exato embutido) foi
executado com subprocess limpo (estado fresco por trial): **21 prompts, ctx 512 (~934 toks
reais), seed 42, 7 reps/depth**.

| depth | fifo | nexus | b/c (fifo-only/nexus-only) | p (McNemar) |
|---|---|---|---|---|
| 0.10 | 0.00 | 0.00 | 0/0 | — |
| 0.50 | **0.57** | **0.00** | **4/0** | 0.125 |
| 0.90 | 0.86 | 0.86 | 1/1 | 1.000 |
| TOTAL | 0.48 | 0.29 | 5/1 | 0.219 |

**Leitura por profundidade (mecanismo confirma os números):**
- **0.10**: agulha em pos ~93/934 → fora da janela FIFO e cedo no anel Nexus com EMA→0.
  Ambos perdem sempre — nenhum sinal possível.
- **0.50**: agulha em pos ~467, escrita no fill, entra no anel do Nexus após ~19 steps
  seguintes com score EMA decaído → vítima garantida. FIFO garante retenção até pos 975
  (não alcançada). As 4 discordâncias vão todas para o FIFO.
- **0.90**: agulha em pos ~840, apenas ~94 tokens depois → majoritariamente na cauda
  protegida dos dois esquemas → paridade alta (6/7 cada).

**Correção ao §20:** o "8/8 vs 5/8" preliminar era contaminado — `scores` por slot-id
persistiam entre chamadas `generate()` no mesmo processo, e agulhas herdavam proteção
acumulada por tokens de trials anteriores. Fix: `NexusEviction.reset()` chamado em
`init_cache()` (`engine/engine.py:100`) — estado zero a cada geração.

**Veredito final da trilha D:**
1. Para needle-single-mention (classe RULER/NIAH), **FIFO é estruturalmente superior**:
   atenção passada mede relevância ao step atual, não utilidade futura; EMA descarta
   tokens importantes silenciosos. O mecanismo e os dados (4/0 discordâncias) convergem.
2. Nexus pós-fix é **correto mecanicamente** (ordem consistente, reset entre gerações)
   e paridade com FIFO nas profundidades extremas — mas não há classe de tarefa testada
   onde supere FIFO. Mantido opt-in como plataforma experimental (reservoir ponderado),
   não como default.
3. Próximo passo só faz sentido com scoring de utilidade-futura (ex.: proteção por
   surpresa/perplexity local), que é redesign, não tuning.

Critério do projeto aplicado consistentemente (int8, Cython, scipy, ONNX-contrib):
resultado negativo documentado é resultado.

### 20.2 Conhecimento generalizável — o viés estrutural de evicção por atenção passada

A limitação que derrubou o Nexus não é peculiaridade desta implementação — é uma
propriedade de **toda** política de evicção pontuada por atenção passada (família
H2O, SnapKV, Nexus). Vale registrado como princípio:

**Princípio:** atenção acumulada mede *relevância retrospectiva* (o que o modelo
olhou até agora), nunca *utilidade prospectiva* (o que será perguntado depois).
Toda token que é importante mas silencioso — mencionado uma vez, nunca mais
consultado até o fim do contexto — é indistinguível de filler para o score, e
vira vítima exatamente por isso.

**Tarefa adversarial por construção:** needle-in-haystack single-mention é o caso
onde esse viés custa caro. A agulha tem exatamente dois picos de atenção — quando
é lida e quando a pergunta chega — e passa todo o intervalo na "zona morta" de
baixa atenção. Políticas de recência (FIFO) são imunes porque garantem cobertura
posicional recente, não julgam importância.

**Assinatura do padrão** (confirmada no sweep §20.1): perda concentrada onde a
agulha passa mais tempo na zona morta (depth intermediário), paridade nos extremos
(0.1: irrecuperável por qualquer política; 0.9: nunca sai da cauda recente). Se um
resultado de evicção mostrar esse formato, desconfie de vitória por tuning — é o
viés estrutural aparecendo.

**Implicação prática:** para superar FIFO em recall de fatos raros, o score precisa
antecipar utilidade futura — proxies plausíveis são surpresa/perplexity local no
momento da escrita (tokens improváveis dado o contexto tendem a ser informativos),
não frequência de atenção. Isso é redesign do scorer, não ajuste de α/R — e explica
por que a literatura de attention-eviction historicamente luta contra recall de
fatos "raros e não-repetidos" enquanto brilha em tarefas com documentos
repetidamente consultados (multi-doc QA, summarization longa).

**Metodologia (reafirmada pelo §20):** falso positivo validado (30/30 testes,
diff numérico ok) é o tipo mais perigoso — parece evidência e mede artefato. O
custo de caçar contaminação de estado antes de publicar veredito é sempre menor
que o custo de carregar a conclusão errada para o design seguinte.

### 20.3 Nexus-salience — hipótese PRÉ-REGISTRADA antes da execução (25/08/2026)

**Mudança:** `NexusSalienceEviction` (`engine/eviction.py`) — score combinado
`w_attn·EMA(atenção) + w_sal·surprisal(slot)`, com `w_attn=1.0, w_sal=0.15`
(10 nats → contribuição 1.5 vs atenção ≤1; filler ~3 nats → 0.45 — separação
por outliers, não por escala absoluta). Surprisal = -log P(token|contexto)
capturado na chegada: `prefill()` computa logits por posição só neste modo
(GEMM extra [T,V]); `step()` encadeia via `_prev_logits`. Reset em
`init_cache()`. Opt-in: `--eviction nexus-salience` / env.

**Limitação conhecida (documentada a priori):** surpresa captura fatos
estatisticamente incomuns; fato semanticamente crítico mas lexicalmente banal
("a reunião é na terça-feira") tem surpresa baixa. Intrínseco a proxies de
surpresa (Memorizing Transformers e afins).

**Hipótese falseável (registrada ANTES do sweep):**
- H1: nexus-salience > fifo em depth 0.5 (onde nexus puro fez 0/7 — a agulha
  tem surpresa altíssima e o piso deve protegê-la na zona morta);
- H2: paridade em 0.9 (agulha já fica na cauda recente);
- H3: sem piora em 0.1 (irrecuperável por qualquer política).
Falsificado se: ganho não aparecer em 0.5, ou aparecer regressão em 0.1/0.9.
Protocolo idêntico ao §20.1: 21 prompts pareados (seed 42), 3 políticas,
McNemar exato por profundidade.
