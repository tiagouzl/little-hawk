# Changelog — Little Hawk 🦅

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

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
