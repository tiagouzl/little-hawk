# Tiny Conversational Runtime v1 — Design

Data: 2026-09-19
Status: aprovado para spec, aguardando revisão do usuário antes do plano
Escopo: produto tiny-ai local, funcionar primeiro em PT, 100% offline

## 1. Objetivo

Chat PT usável sobre o Little Hawk existente, sem rede, com memória
persistente que parece longa sem contexto longo.

Sucesso v1:
- `chat --weights qwen_weights.npz` responde em PT, lembra fatos
  (ex.: "meu projeto é Little Hawk") após 500+ turnos sem recarregar
  histórico inteiro.
- RSS aceita ~1.3GB (Qwen float32 atual). Otimização <500/<200MB é
  v2, não v1.
- Zero dependência nova além de stdlib + sqlite3.

## 2. Não-objetivos (YAGNI v1)

- Sem vector DB (Qdrant/Chroma), sem embeddings.
- Sem fallback cloud, sem μLM 8-30M, sem BitNet.
- Sem RWKV/Mamba, sem GGUF/quantização nova.
- Sem MCP, sem multi-agente, sem Docker novo.
- Sem mudar `engine/`: FIFO default, `prefill()` existente.

## 3. Arquitetura

```
USER
 ↓
Router (regex) ──┬── COMMAND → Tools FS ──┐
                 ├── MEMORY → SQLite FTS5 ─┤→ Context Builder (≤750 toks) → Qwen engine → RESPONSE
                 └── CHAT ────────────────┘
```

Orçamento de contexto (hard cap 1024):
- system 100 + memórias 100 + últimos turnos 500 + pergunta 50 ≈ 750.

## 4. Componentes

### 4.1 `runtime/router.py`
- `route(text) -> (intent, arg)` com regex PT:
  `abra|liste|mostre|qual meu|lembre que|meu projeto`.
- Intents: `OPEN_FILE, LIST_FILES, SAVE_FACT, ASK_MEMORY, CHAT`.
- Sem LLM no roteamento. Fallback sempre `CHAT`.

### 4.2 `runtime/memory.py`
- SQLite `memory.db`, WAL mode:
  `messages(id, role, text, ts)`, `facts(key, value, importance, updated)`,
  `facts_fts(value)` FTS5.
- API: `save_fact(key, value)`, `recall(query, k=3)`, `log_turn(role, text)`.
- Recall = FTS5 `MATCH` + `ORDER BY rank LIMIT 3`. Sem embeddings.

### 4.3 `runtime/conversation.py`
- `turn(user_text) -> str`:
  1. `route()` 2. `recall()` 3. monta prompt
     `Context: <facts>\nHistórico: <últimos>\nUser: <q>` 4. chama
     `MultiLayerEngine.prefill + generate` existente 5. `log_turn()`.
- Trunca histórico para caber em 500 toks (contagem aproximada
  `len//4`, sem tokenizer extra).

### 4.4 `runtime/tools_fs.py`
- `list_files(path)`, `open_file(path, n=50)` com sandbox:
  resolve `Path.cwd()`, rejeita `..` fora da raiz e arquivos >1MB.
- Chamado só quando `intent == COMMAND`. Retorno vira `Context:`.

### 4.5 Hook CLI
- `cli/main.py chat`: troca chamada direta ao engine por
  `conversation.turn()`. Flags `--memory-db`, `--no-memory` novas.
- Sem mudar sampling, eviction, telemetria.

## 5. Data flow exemplo

```
You: Meu projeto se chama Little Hawk.
→ SAVE_FACT(project_name=Little Hawk) → "Anotado."

... 500 turnos depois ...

You: Qual é meu projeto?
→ ASK_MEMORY → FTS5 retorna "Little Hawk"
→ prompt = "Context: projeto=Little Hawk\nUser: Qual é meu projeto?"
→ Qwen responde local, ≤1024 toks, FIFO.
```

## 6. Erros

- `memory.db` ilegível → modo in-memory + aviso, chat continua.
- Arquivo fora do sandbox → resposta "caminho não permitido", sem throw.
- `.npz` inválido → erro claro já existente em `load_weights`, sem mudar.
- Prompt >8000 chars → rejeita (limite API já existente).

## 7. Testes

- `test_router.py`: 10 frases PT → intent esperado.
- `test_memory.py`: save/recall roundtrip + FTS5 rank + persistência.
- `test_conversation.py`: budget ≤1024 toks, truncamento, mock engine.
- E2E manual: Qwen PT "lembre meu projeto / pergunte 5 turnos depois".
- `make test` deve continuar 42/42 + novos.

## 8. Futuro (fora do v1)

- v1.1: `--eviction nexus-salience` default para recall raro.
- v2: backend GGUF int4 plugável (`backend/`), meta <500MB.
- v3: fallback cloud opcional, μLM router 10-30M.

## Self-review

- Placeholders: nenhum TBD. Paths, tabelas, limites explícitos.
- Consistência: Qwen PT (não SmolLM EN), FIFO (não salience), offline
  total (sem cloud), sem mudança em engine/.
- Escopo: 4 arquivos novos + hook CLI. Cabe num plano único.
- Ambiguidade: contagem de tokens aproximada documentada como
  `len//4`; sandbox definido como `Path.cwd()`; FTS5 sem embeddings.
