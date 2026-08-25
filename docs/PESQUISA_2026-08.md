# Pesquisa de Ecossistema — agosto/2026

> **Contexto:** levantamento de atualidades do ecossistema realizado em 24/08/2026,
> após o fechamento da trilha performance (v0.7.0, `428614c`). Objetivo: mapear
> desenvolvimentos recentes que possam beneficiar o Little Hawk e definir as
> próximas trilhas de trabalho.
>
> **Estado do projeto na data:** ONNX fp32 validado (1.21× vs NumPy), prefill
> batched (TTFT <0.2 s para prompts ~130 tokens), quantização int8/4 rejeitada,
> suíte com 29 testes verdes, working tree sincronizada com `origin/main`.

---

## 1. ONNX Runtime ≥ 1.25 — ops fundidas nativas no CPU EP

O que há de novo nos release notes (`github.com/microsoft/onnxruntime/releases`):

- **Operadores contrib novos no CPU EP**:
  - `RotaryEmbedding` — funde toda a cadeia Mul/Add/Slice do nosso RoPE manual;
  - `RMSNorm` — substitui o bloco `Mul/Mean/Add/Sqrt/Div/Mul/Add`;
  - `Attention` (opset 23+) — atenção completa com GQA, máscara causal e
    semântica de KV-cache (`past`/`present`) integradas ao operador;
  - `LinearAttention` / `CausalConvState` — adicionados para suportar Qwen3.5
    (arquitetura DeltaNet).
- **ORT 1.29**: variável de ambiente `ORT_INTRA_OP_NUM_THREADS` para controlar o
  thread pool sem código.

**Aplicação ao Little Hawk:**
Nosso grafo atual é exportado por tracing do PyTorch → cada RMSNorm/RoPE/atenção
vira dezenas de ops elementares que o ORT funde apenas parcialmente (daí os
apenas 1.21×). Reconstruir o grafo **à mão** com `onnx.helper`, usando esses
operadores contrib nativos, pode fundir blocos inteiros → potencial bem acima
de 1.21×, mantendo o `torch_model.py` como referência numérica.

⚠️ Ops contrib são sensíveis à versão do runtime — fixar `onnxruntime` no
`pyproject.toml` se seguirmos esse caminho.

---

## 2. NumPy 2.4 / 2.5

Release notes (`numpy.org/doc/stable/release/`):

- Ufuncs com escalares ~**6× mais rápidas** (afeta diretamente o glue Python do
  `fast_step`: indexação do cache circular, RoPE posicional, caminho do argmax);
- `np.unique` com hash interno (não crítico para nós);
- Baseline x86 elevada para **V2 (AVX)**;
- Suporte a Python free-threaded (3.13t/3.14t) maduro no NumPy 2.5.

**Aplicação ao Little Hawk:**
1. Checar versão instalada no venv (`venv/bin/python -c "import numpy; print(numpy.__version__)"`);
   se <2.0, upgrade + rodar suíte de testes + re-benchmark (ganho esperado
   pequeno, mas custo é quase zero);
2. Free-threading é relevante apenas para o servidor API multi-cliente (SSE),
   não para tok/s single-stream — item futuro da trilha API.

---

## 3. Cenário de modelos pequenos (candidatos a transplant)

| Modelo | Arquitetura | Compatibilidade | Observações |
|---|---|---|---|
| **SmolLM2 135M/360M/1.7B** | Igual ao nosso transplant (RMSNorm, RoPE, SwiGLU) | **~zero esforço**: novo `.npz` + config | Melhor razão ganho/esforço; salto de qualidade |
| **SmolLM3 3B** | GQA + NoPE + YaRN, 128k ctx | Média: exige GQA nos kernels + NoPE/YaRN | fp32 = 12 GB RAM; inviável sem revisitar quantização |
| **Qwen3.5** | Atenção linear (DeltaNet) | Baixa: kernels novos | Watch-list apenas |

Referência prática: SmolLM3-3B Q4_K_M roda ~13.6 tok/s em CPU desktop via
llama.cpp/GGUF — mostra que a classe de modelo é viável em CPU, mas com
quantização que nós rejeitamos por qualidade (ver §17 do ANALISE.md).

**Validação externa do nosso núcleo:** o paper StreamingLLM
(`arxiv.org/abs/2309.17453`) continua sendo, em 2026, o baseline padrão da
literatura para streaming infinito com janela fixa — valida a escolha central
do projeto.

---

## 4. Eviction / KV-cache — estado da arte 2026

Panorama (comparativo completo: `dreaming.press/posts/kv-cache-eviction-streamingllm-vs-h2o-vs-snapkv-vs-quest.html`):

- **StreamingLLM** (nosso esquema): sinks + janela deslizante. Não estende
  recall, estende tempo de execução estável. Continua válido.
- **H2O / SnapKV**: políticas de eviction baseadas em scores históricos ou voto
  no fim do prefill. SnapKV comprime o *prompt* no fim do prefill — conceito
  aplicável agora que temos `prefill()` batched, mas projetado para prompts de
  16k+; nossa janela é 512 fixa. Prioridade baixa.
- **Quest** (`arxiv.org/abs/2406.10774`): seleção query-aware por páginas
  min/max. **Não se aplica** — economiza banda mantendo o cache cheio; somos
  memory-bound por capacidade fixa.
- **⭐ Nexus Sampling** (`arxiv.org/html/2606.23961`, jun/2026): o achado mais
  relevante. Substitui a seleção determinística top-K/janela por **reservoir
  sampling ponderado** sobre os scores de atenção direta:
  - Em 80% de eviction, fica a ~1 ponto da atenção densa no LongBench;
  - Supera top-K/H2O/MorphKV em tarefas retrieval-heavy (ex.: Llama-3.2-1B
    RULER 59.81 vs H2O 36.49);
  - Cache pós-decode constante (~20% da densidade), 5–10× menos memória;
  - Training-free, opera dentro de budget fixo.

**Aplicação ao Little Hawk — "Eviction Policy v2":**
- Nosso buffer circular sobrescreve estritamente por recência; o slot mais
  antigo da janela morre sem julgamento.
- Nós **já calculamos as probabilidades de atenção a cada step** → os dados de
  scoring são gratuitos.
- Ideia: manter sinks imutáveis + janela recente, mas o anel intermediário
  passa a ser gerido por reservoir sampling ponderado (probabilidade de inclusão
  proporcional à massa de atenção acumulada), em vez de FIFO cego.
- Métrica alvo: coerência/retrieval em streams longos (>600 steps), onde hoje
  tokens importantes fora da janela recente são perdidos.

---

## Trilhas propostas

| # | Trilha | Esforço | Ganho esperado | Reabre perf? |
|---|---|---|---|---|
| A | Knobs rápidos: threads do ORT 1.29 + NumPy 2.x | Horas | Pequeno (~5–15%?) | Parcial |
| B | Grafo ONNX reconstruído com ops contrib fundidas (RMSNorm/RoPE/Attention) | Dias | Médio-alto (>1.21×) | Sim |
| C | Transplant SmolLM2-360M/1.7B | Baixo | Qualidade (não velocidade) | Não |
| D | Eviction v2: Nexus/reservoir no buffer circular | Médio | Coerência em streams longos | Não |

**Recomendação de ordem** (se todas forem aprovadas): C → A → B → D.
C é quase de graça e dá salto de qualidade imediato; A é diagnóstico barato;
B é o grande salto de performance; D é pesquisa com risco científico maior.

> Nota: A e B tocam a trilha performance que foi **encerrada por decisão do
> usuário** — só iniciar com aval explícito.

---

## Como retomar

1. Ler este documento + `ANALISE.md` §§16–19 (contexto técnico das trilhas
   anteriores).
2. Confirmar estado: `git status` limpo, HEAD em `428614c` ou posterior.
3. Escolher trilha e abrir com objetivo claro (padrão das sessões anteriores:
   implementar → validar contra NumPy → bench → release + CHANGELOG).

### Artefatos úteis (sessão de pesquisa)

- Scripts de validação/bench em `/tmp/opencode/`: `val600.py`, `val_variants.py`
  (aceita paths de modelos como argv), `ttft_e2e.py`, `micro8.py`.
- Export ONNX canônico fp32: `/tmp/little_hawk_full_30L.onnx` (706 MB) —
  **regenerar se ausente**; nunca reusar exports stale de `/tmp` (lição da
  divergência step-513).
- Exportador legado obrigatório: `torch.onnx.export(..., opset_version=17, dynamo=False)`.

---

## Fontes

- ONNX Runtime releases: https://github.com/microsoft/onnxruntime/releases
- Nexus Sampling: https://arxiv.org/html/2606.23961
- Comparativo eviction (StreamingLLM vs H2O vs SnapKV vs Quest):
  https://dreaming.press/posts/kv-cache-eviction-streamingllm-vs-h2o-vs-snapkv-vs-quest.html
- Quest: https://arxiv.org/abs/2406.10774 · https://github.com/mit-han-lab/quest
- SnapKV: https://proceedings.neurips.cc/paper_files/paper/2024/file/28ab418242603e0f7323e54185d19bde-Paper-Conference.pdf
- StreamingLLM (base do projeto): https://arxiv.org/abs/2309.17453
- NumPy release notes: https://numpy.org/doc/stable/release/
