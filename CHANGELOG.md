# Changelog — Little Hawk 🦅

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

## [0.8.0] - 2026-08-26

### 🧬 Transplants SmolLM2 (135M / 360M / 1.7B)
- `--model smollm2-{135m,360m,1.7b}` no CLI de transplant; configs em `utils/config.py`.
- Fix BF16: `load_safetensors()` converte bfloat16→float32 via shift de bits (o NumPy não entende o dtype cru).

### 🎯 Evicção Nexus e Nexus-Salience (trilhas D/D2)
- Novo `engine/eviction.py`: FIFO explícito + reservoir ponderado por atenção + **piso de surpresa** (`nexus-salience`): score = w_attn·EMA(atenção) + w_sal·surprisal na chegada (capturado do forward, sem passada extra).
- Fix estrutural: ordem de recência explícita (`NexusEviction.order`) — a leitura deixou de assumir layout FIFO após escritas por override; reset entre gerações (`init_cache`).
- **Resultado RULER pareado (21 prompts, McNemar exato): nexus-salience 0.95 vs fifo 0.52 (p=0.004, 9/9 discordâncias)** — dominância em depth 0.1 (fora da janela de recência). Detalhes: ANALISE §20–20.4.

### 🔧 Correções
- Dockerfile: CMD `api:app` → `api.server:app`.
- Prefill batched: bias Q/K/V do Qwen somado antes do reshape (ValueError com >1 token).
- Prefill chunked para prompts > 512 tokens (primeiro chunk batched + restante sequencial).
- Rollback silencioso ONNX+eviction: agora avisa qualquer modo ≠ fifo.
- Dependências: fastapi/uvicorn movidos para extra `[api]`; colorama removido (morto) — "Sem frameworks" literal para o motor.

### ⚗️ Investigados e documentados (ANALISE §21 + PARECER_ROADMAP)
- **N-gram speculative decoding**: Fase A (verificador causal batched `verify_chunk`) validado contra sequencial; Fase B greedy → **neutro (0.995×)**: aceitação 100% mas ciclo speculator completo não gera speedup wall-clock.
  - **B0 (§21.4):** verify_chunk = **0.55× de k steps** (k=4) — KPI corrigido (era 1.90× vs 1 step). Breakdown: rope_k 43%, FFN 31%, av_out 10%.
  - **B1 (§21.5):** batch de rope_k testado → **1.6× mais lento** que chamadas individuais. Neste backend NumPy não há justificativa para substituição.
  - Trilha E: **CLOSED**. Alavancas de reabertura documentadas.
- Free-threading 3.13t: teto é bandwidth de memória — mantido 3.12 + Semaphore(2).
- Ops contrib ONNX (com.microsoft): ausentes do wheel pip — trilha cancelada.
- Mamba/MiniCache/BitNet/Saguaro avaliados e recusados com justificativa (PARECER_ROADMAP_2026.md).

---

## [0.7.0] - 2026-08-24

### ⚡ Prefill batched — TTFT até 20× menor
- `MultiLayerEngine.prefill(tokens)`: forward único do prompt com máscara causal + RoPE por posição (fase fill, T ≤ max_cap). Equivalência numérica vs steps sequenciais validada (logits/cache diff ~1e-05, win_ptr idêntico em T ∈ {1..512}).
- `runtime/inference.py`: fill do prompt agora usa prefill; excedente acima de `max_cap` continua sequencial. Engines sem `prefill` (ex.: OnnxEngine single-token) caem no loop original automaticamente.
- TTFT medido: prompt de 129 tokens passou de ~14 s para **<0,2 s**; GEMMs batched [T,d] aproveitam OpenBLAS muito melhor que GEMV sequenciais.
- 2 novos testes (`TestPrefill`): prefill == sequencial e prefill+steps == tudo sequencial.

## [0.6.0] - 2026-08-24

### 🚀 Backend ONNX Runtime (opt-in) — 1.21× no loop completo
- `engine/torch_model.py`: espelho PyTorch do motor (`LlamaLayerTorch` + `LittleHawkTorch`) com **cache circular 512 exportável** — `win_ctx` via `torch.where(n_win < W, arange, (wbi+win_ptr+1)%W+S)` + slice dinâmico, e **position freeze** (`pos_q = where(n_ctx ≤ 512, n_ctx-1, 511)`).
- `engine/onnx_engine.py`: reescrito para o grafo completo (embed → 30 camadas → norm → lm_head embutidos, 705 MB) com entradas `(input_ids, k_stack[30,1,9,512,64], v_stack, win_ptr, n_ctx)`; exporta sob demanda via `torch.onnx.export(..., opset_version=17, dynamo=False)` e **não** reusa exports antigos de `/tmp` (grafo stale divergia a partir da 1ª sobrescrita circular).
- `engine/__init__.py` + `cli/main.py`: `get_engine()` retorna `OnnxEngine` quando `LITTLE_HAWK_ONNX=1` (fallback NumPy automático).
- **Validação numérica 600 steps vs NumPy** (stream argmax compartilhado): diff máx de logits `1.5e-04`, top-1 `12/12` checkpoints, top-5 `5.00/5`; step 513 (1ª sobrescrita da janela circular) diff `1.8e-05` ✅.
- Bench OMP=1: ONNX **91 ms/step** vs NumPy 110 ms/step (**1.21×**); teto FFN-only medido antes: **6.62×** (294→44 ms).
- 🔬 **Quantização int8/4 avaliada e rejeitada**: `quantize_dynamic` (u8s8, 179 MB, 1.52×) destrói o top-5 (1.83/5 — outliers de ativação espremem escala min/max); `MatMulNBits` int8 weight-only mantém qualidade mas é 0.50× (kernel lento em GEMV batch-1). fp32 permanece o backend. Detalhes em `ANALISE.md §17`.

## [0.5.0] - 2026-08-24

### ⚡ P3 — Cython e ONNX (investigação completa)
- `engine/fast_step.py` inlined Python (1.13×, 66→58 ms/token) — melhor custo/benefício; `engine/cython_fast.pyx` compilado (Cython 3.3, 718 KB) mas 0.70× (mais lento, boxing) — rejeitado sem BLAS C-API.
- `scipy.linalg.blas.sgemv` 5.8× mais lento que NumPy (613→3561µs) — mesmo padrão `numba` 6.1×; OpenBLAS já ótimo para batch-1.
- `scripts/onnx_export.py` POC 1L 3.04×, 30L 6.62× (294→44 ms, 318 MB) via `onnxruntime` MKL fusion — teto para `v0.5.0`. `scripts/onnx_full.py` stub com KV cache + RoPE para próximo passo (`torch.export`).
- `pyproject.toml` extras `[cython]` e `[onnx]`, `engine/cython_fast.*` ignorados em `.gitignore`.

### 🧹 Qualidade
- `ruff.toml` 7→6 entradas (`utils/` promovido, `engine/runtime` lint-clean `EXE001/SIM118`), `ruff check` ✅ `27 passed`.

## [0.4.0] - 2026-08-23

### ⚡ Performance & JIT
- Adicionado extra opcional `[jit]` no `pyproject.toml` com dependência de `numba>=0.58.0`.
- Implementados kernels JIT (`@njit(fastmath=True, nogil=True)`) em `engine/jit_kernels.py` para **RMSNorm** (`_jit_rms_norm`) e fusão **SwiGLU** (`_jit_silu_mul`) — **opt-in** (requer numba + `LITTLE_HAWK_JIT=1`), com fallback NumPy **vetorizado** quando ausente.
- ⚠️ Medição honesta: no decode batch-1 os ganhos são nulos/levemente negativos (JIT 405 vs NumPy 358 ms/token) — RMSNorm/SwiGLU são ~2% do passo; o GEMV do `lm_head` domina (~75%). Próximo alvo real de otimização: quantização/caching do `lm_head`.
- Adicionado script padronizado de medição de latência `scripts/benchmark_latency.py` (latência média, p50, p95 e warm-up controlado) e profiler por componente `scripts/profile_step.py`.
- Mantida contiguidade de memória (`np.ascontiguousarray`) nas matrizes de peso transpostas do `LlamaLayer` para máxima eficiência em operações GEMV do BLAS.

### 🔬 lm_head: int8 avaliado e rejeitado; orientação contígua adotada
- Microbenchmark controlado: int8 por coluna é 3–4× MAIS LENTO que fp32 em NumPy puro (upcast por token) — rejeitado pelo critério de ganho vs perda numérica.
- Profiling corrigido: lm_head custa ~4% do passo (11 ms), não 75%; gargalo real é o dispatch das 30 camadas (~96%).
- Adotado `W_lm_t` `[V,d]` contíguo no hot path (−1.4 ms/token em A/B intercalado, erro 2e-7); +2 testes de regressão.

### 🐛 Correções de Bugs
- **P0**: Corrigido bug de incremento duplo de `win_ptr` em `scripts/test_long_context.py`.
- Deduplicada a função `_rms_norm` entre `engine.py` e `transformer.py`.
- Limpeza de imports locais e variáveis não utilizadas em `runtime/inference.py`.

### 📦 Modularização e Arquitetura
- Criado o módulo `transplants/` estruturado (`smollm.py` e `qwen.py`), desacoplando o comando CLI de scripts na raiz.
- Adicionada rota raiz `GET /` no servidor FastAPI (`api/server.py`) para servir a interface web diretamente.
- Removidos diretórios órfãos locais e referências desatualizadas a `setup.py`.

### 💬 Interface & Experiência
- Reformulada a interface `sse_demo.html` com visualizador gráfico em tempo real do **StreamingKVCache $O(1)$** (Attention Sinks vs Janela Deslizante) e métricas dinâmicas de throughput (tok/s) e latência.

---

## [0.3.0] - 2026-08-22
- Modularização inicial em submódulos (`engine`, `runtime`, `api`, `cli`, `utils`).
- Implementação de `StreamDecoder` para decodificação UTF-8 byte-safe.
- Correção do cálculo RoPE para convenção `rotate_half` HuggingFace.
- Implementação do servidor FastAPI assíncrono com SSE streaming e controle de concorrência.
