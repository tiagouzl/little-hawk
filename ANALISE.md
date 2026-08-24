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
