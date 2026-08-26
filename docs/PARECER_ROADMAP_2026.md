# Parecer Técnico — Propostas "Estado da Arte 2025/2026" para o Little Hawk
**Análise crítica das 4 frentes propostas (Mamba/SSM, KV Cache 3.0, Speculative Decoding, Integer-Only)**

> **Data:** 26/08/2026 · **Contexto:** resposta à proposta de roadmap pós-trilha D2  
> **Método:** cada proposta confrontada com (a) evidência já documentada no próprio projeto (`ANALISE.md`), (b) o ciclo de valor do motor (transplant → equivalência numérica → inferência de checkpoints reais), (c) hardware de referência (i5-10210U, 8GB, sem AVX-512).  
> **Veredicto em uma linha:** 1 das 4 frentes é ouro alinhado ao projeto; as outras 3 conflitem com evidência interna, exigem checkpoints inexistentes no nosso pipeline, ou otimizam o gargalo errado.

---

## Sumário do parecer

| # | Proposta | Veredicto | Motivo central |
|---|---|---|---|
| 1 | Mamba/SSM em NumPy | **Recusar** | Bloco órfão: nenhum checkpoint do transplant pipeline tem camadas SSM; híbridos reais estouram 8GB |
| 2a | MiniCache (compressão entre camadas) | **Recusar (com número)** | Cache = 70.8MB de 1.28GB RSS (5.5%); metade dele economiza ~35MB — problema que O(1) já resolveu |
| 2b | RocketKV (evicção permanente+ dinâmica) | **Adiar** | Testável, mas `nexus-salience` acabou de dominar o RULER (p=0.004); margem estreita, complexidade alta |
| 2c | SmallKV / FP4 simulado | **Recusar** | Padrão já refutado no §11: sub-byte simulado em NumPy perde para fp32 (upcast mata o ganho) |
| 3a | **N-gram Speculative Decoding** | **CLOSED** | Mecanismo correto (verify 0.55× de k steps), mas ciclo completo não gera speedup wall-clock; B1 rope batch negativo (1.6×) |
| 3b | Saguaro/SSD | **Não verificável** | Claim "5×" não auditável; padrão `fast_step` sugere dispatch Python como gargalo, não verificação |
| 4 | BitNet/XNOR-Popcount/Integer-only | **Recusar** | Exige checkpoints treinados nativamente em b1.58; binarização post-hoc já destruiu qualidade em int4 (§17) |

---

## 1. Mamba / SSM — recusar, e o motivo é estrutural

A matemática é real e atraente em NumPy: estado discreto recursivo
`h_t = Ā·h_{t-1} + B̄·x_t`, saída `y_t = C·h_t` — poucas linhas, O(N) no comprimento,
sem KV cache/RoPE/sinks para essas camadas. Como exercício didático, funcionaria.

O problema não é a equação — é **o que o bloco faria no motor**:

* **Órfão de pesos.** O ciclo de valor do Little Hawk é transplantar checkpoints reais
  (SmolLM, Qwen, SmolLM2) e validá-los numericamente contra o HF Transformers
  (`test_equivalence.py`, diffs ~1e-4, RoPE rotate_half corrigido por equivalência).
  Um `SSMBlock` híbrido não roda nenhum checkpoint do nosso pipeline — SmolLM e Qwen
  não possuem camadas SSM. O bloco seria validável apenas contra si mesmo.
* **Os híbridos reais não cabem.** Jamba: 52B MoE. Zamba2 menor ainda excede os 8GB
  em fp32. Cada arquitetura nova exige reescrever transplant, GQA-expansion e meta —
  semanas de esforço para rodar qual modelo, em qual máquina?
* **Redefine o projeto.** Se o objetivo virar "playground de arquiteturas", isso é um
  novo projeto, não uma evolução deste — cujo escopo documentado é inferência de
  modelos reais em CPU modesta.

## 2. KV Cache 3.0 — três propostas, três problemas distintos

### 2a. MiniCache (fusão entre camadas) — recusada por aritmética

Número decisivo, medido no nosso próprio benchmark (`README.md:143`): o cache O(1)
ocupa **70.8MB de 1.28GB de pico RSS** — **5.5%** do footprint. O gargalo de memória
são os *pesos* fp32 das 30 camadas, não o cache. Fusão perfeita entre camadas adjacentes
economizaria ~35MB num sistema que já é tamanho-constante por design. O StreamingLLM
atacou a dimensão certa *para este motor*: crescimento temporal, não redundância entre
camadas. Custo/benefício desfavorável mesmo antes de medir qualidade.

### 2b. RocketKV — adiar

Evicção permanente (estatísticas do prefill) + dinâmica (atenção corrente) é conceito
testável com o harness RULER pareado que já possuímos. Mas: o `nexus-salience` acabou
de dominar esse benchmark (0.95 vs 0.52 do FIFO, p=0.004, 9/9 discordâncias — §20.4).
A margem remanescente é estreita e o custo em complexidade de estado é alto. Só
revisitar se surgir classe de tarefa onde o salience demonstrar fraqueza específica.

### 2c. SmallKV / FP4 — refutado por precedente interno

O §11 já mediu o padrão: int8 por coluna foi **3–4× mais lento** que fp32 em NumPy
puro porque o upcast por token elimina o ganho de banda. Simulação FP4 via bit-packing
é estritamente pior (mais manipulação de bits por elemento). E XNOR+Popcount exige
instruções que o hardware de referência **não possui** — o README documenta
explicitamente "i5-10210U (4 cores, sem AVX-512)".

## 3. Speculative Decoding — ENCERRADO

### 3a. N-gram speculative decoding — CLOSED

Implementado e medido (§21.1–5 do ANALISE.md). Resultado:

- **verify_chunk** é estruturalmente viável: 0.55× de k steps sequenciais em k=4,
  0.39× em k=8 (§21.4, B0).
- **Ciclo speculator completo** não gera speedup wall-clock (0.995× neutro, §21.3):
  custo fixo do ciclo (lookup + argmax + rollback + token bônus) consome o ganho.
- **B1 (fusão rope_k):** batching proposto foi 1.6× mais lento que chamadas
  individuais (§21.5). Neste backend NumPy, não há justificativa para substituição.
- **Equivalência:** 28/28 testes.
- **Reabertura somente se:** surgir novo backend/kernel capaz de alterar o custo
  fundamental da operação de rotação RoPE.

### 3b. Saguaro/SSD — não verificável

Claim de "até 5×" e "remoção da dependência sequencial" não auditável por mim.
Desconfiança principiol pelo padrão já medido (`fast_step`, §19): em CPU batch-1,
o gargalo dominante é dispatch Python/BLAS (~96% do passo nas 30 camadas), não o
algoritmo de verificação. Qualquer esquema que multiplique trabalho por token rascunhado
paga o dispatch — exceto quando agrupa em GEMM, que é precisamente o caso 3a.

## 4. BitNet / Integer-only — mesma resposta do item 1

Os números de energia (ex.: 0.028J/inferência) são claims de paper não auditáveis aqui,
e o caminho exige modelos **treinados nativamente** em b1.58/2-bit — quantização é parte
do treino, não transformação post-hoc. Nosso próprio §17 provou que até int8/int4
dinâmico destrói o top-k (1.83/5); binarização é ordens de magnitude além. Sem checkpoint
BitNet pequeno compatível com o pipeline de transplant, é matemática órfã igual ao Mamba.

---

## Roadmap recomendado (contraproposta)

**Ciclo atual — ENCERRADO (trilha E fechada, §21.4–5):**

1. ~~`runtime/speculative.py`: tabela n-gram do prompt + janela recente~~ ✓
2. ~~Draft de k≤4 → verificação batched via `verify_chunk()`~~ ✓
3. ~~Flag opt-in `--speculative K`~~ ✓
4. **Resultado:** verify_chunk rápido (0.55× de k steps), mas ciclo completo neutro.
   B1 rope batch negativo (1.6×). Trilha CLOSED.

**Backlog mantido (dependências externas, inalterado):**
ORT `cp313t` wheel → reavaliar migração free-threaded da API;
int8 estático + VNNI → exige hardware/dataset fora do escopo atual;
generalização do salience (seeds, checkpoints, formatos de agulha lexicalmente banais).

**Rejeitados/completados com justificativa documentada (esta análise):**
Mamba/SSM híbrido, MiniCache, FP4 simulado, XNOR/BitNet post-hoc, Saguaro SSD,
N-gram speculative decoding (CLOSED — §21.4–5).

---

## Princípio aplicado

O projeto construiu sua credibilidade documentando resultados negativos com mecanismo
explicado (int8 §11, Cython §13-14, ONNX-contrib, Nexus pré-fix §20.1). Esta análise aplica
o mesmo critério na direção oposta: **recusar propostas atraentes quando conflitam com
evidência interna ou rompem o ciclo transplant→validação→inferência** — e aceitar a única
que se apoia em ativos que o projeto já possui.

*"Só entregar se ganho superar claramente a perda"* — critério acordado no §11, válido
para features tanto quanto para microbenchmarks.

---

## Revisão v1.1 — Emendas da revisão externa (26/08/2026)

Três emendas aceitas à proposta original; nenhuma altera veredictos, todas endurecem execução.

### E1 — Speculator como abstração mínima ANTES do N-gram

Um passo antes de `runtime/speculative.py`: contrato pequeno com quatro operações,
e `NGramSpeculator` como primeira implementação concreta.

```
Speculator          # contrato
├── propose(ctx_ids) -> k candidatos
├── verify(batched_forward) -> logprobs dos k
├── accept()        # reject-sampling corrigido
└── rollback(n)
```

Justificativa: se N-gram funcionar, estratégias futuras (prompt lookup puro,
cache lookup, draft model pequeno) plugam sem reescrever a verificação.
**Restrição explícita:** manter mínimo — sem ABC/formalismos; dataclass +
funções bastam. A abstração existe para servir o experimento, não para
precedê-lo em complexidade.

### E2 — H1 endurecido: speedup sem acceptance rate não conta

Speedup isolado é ambíguo economicamente (1/4 aceitos vs 4/4 aceitos são
"funcionando" com custos opostos). Registro obrigatório por trial:

| Métrica | Definição |
|---|---|
| tokens propostos | total de candidatos gerados pelo speculator |
| tokens aceitos | candidatos que sobreviveram à verificação |
| acceptance rate | aceitos/propostos |
| tokens efetivos/forward | throughput real vs 1 token/forward do baseline |
| ms/token | latência média por token emitido |
| speedup | vs rollout sequencial, mesmo prompt/seed |
| top-k match | equivalência vs sequencial (obrigatório 100% nos aceitos) |
| memória adicional | RSS delta da tabela n-gram |

Falso positivo a evitar: speedup alto com acceptance rate baixo indica overhead
de verificação pagando por nada — nesse caso a conclusão é "mecanismo correto,
política de proposal ruim", não "speculative decoding funciona".

### E3 — Mamba: fora do roadmap imediato ≠ fora do conhecimento estratégico

Distinção preservada no registro: a recusa é sobre **ordem**, não sobre valor
permanente. Se o ecossistema produzir checkpoints híbridos pequenos compatíveis
com transplant (ou o projeto ganhar hardware), a análise do item 1 é reaberta —
a matemática SSM em NumPy permanece conhecimento mapeado, não descartado.
Mesma lógica vale para BitNet: a distinção *quantizar modelo existente* ≠
*modelo que nasceu quantizado* fica registrada como critério permanente.

### Identidade técnica (registrada conforme revisão)

> Um runtime experimental de inferência CPU no qual cada otimização precisa
> provar valor contra um pipeline real, com equivalência numérica e benchmarks
> reproduzíveis.

Não é clone de llama.cpp, não é framework universal de arquiteturas, não é
playground de papers. As emendas E1–E3 e os veredictos §§1–4 derivam dessa definição.

---

## Revisão v1.2 — Correção técnica material (revisão de código, 26/08/2026)

**Correção ao item 3a:** o claim *"verificação paralela já existe via `prefill()`"*
estava **tecnicamente incompleto** e é substituído por:

> O `prefill()` demonstra que o runtime já possui um caminho GEMM-batched eficiente;
> precisamos **generalizar essa ideia para uma verificação causal batched sobre um
> estado de cache já existente**.

Motivo (verificado no código): `prefill()` assume cache vazio — escreve K/V em
slots `0..T-1`, máscara causal do zero. Chamar `prefill(candidatos)` após o prompt
**reiniciaria semanticamente o contexto**. Reaproveitar o *conceito* e os kernels
é correto; reutilizar a função literalmente, não.

**Consequências para o roadmap (fase A antes de tudo):**

1. `engine/speculative.py` (não `runtime/`): `verify_chunk(engine, tokens, caches,
   win_ptr, n_ctx)` — extensão causal batched sobre o estado existente;
   candidatos são sequência única `t0→t1→…` com máscara triangular própria;
2. Equivalência obrigatória ANTES de qualquer N-gram: logits/cache/win_ptr/positions
   vs steps sequenciais (mesmo padrão dos testes TestPrefill);
3. Rollback por cópia de buffers (`k.copy()`) — k≤4 torna isso aceitável na v1;
4. Greedy speculation primeiro (aceita enquanto argmax coincide); reject sampling
   só depois; desacoplado do `Sampler`;
5. v1 FIFO-only: evicção reservoir escolhe vítimas por step com scores que mudam
   a cada camada-0 — batching sob reservoir é problema distinto, cai no fallback
   sequencial documentado.

**Hipótese refraseada conforme revisão:** não "speculative decoding é mais rápido
por definição", e sim — *se 4 verificações virarem GEMMs maiores, há ganho mesmo
executando mais FLOPs, porque o gargalo medido é dispatch/GEMV, não FLOPs.*
