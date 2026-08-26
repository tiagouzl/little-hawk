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
| 3a | **N-gram Speculative Decoding** | **IMPLEMENTAR** | Sem modelo draft, lookup array puro, verificação via `prefill()` batched já existente |
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

## 3. Speculative Decoding — a única frente ouro

### 3a. N-gram speculative decoding — IMPLEMENTAR

Única proposta alinhada em todas as dimensões do projeto:

* **Sem modelo rascunhador** — usa prompt + histórico como tabela de n-gramas;
  filosofia leve preservada, zero dependência nova.
* **Lookup puro em array** — dict hash → candidatos de próximo token; custo ~zero.
* **Verificação paralela já existe**: aceitar/rejeitar k rascunhos é um forward batched
  `[k,d]@[d,m]` — exatamente o `prefill()` do §18, que já provou ser 10–20× mais
  eficiente que steps sequenciais no mesmo hardware.
* **Testável com infraestrutura existente**: `benchmark_latency.py` + verificação
  top-k idêntica à validação do ONNX (§16).

**Expectativa honesta, por regime:**

| Regime | Expectativa |
|---|---|
| Greedy / min_p baixo, texto previsível | speedup ≥1.3× plausível |
| Amostragem criativa (temp ≥0.9) | ≤5% — reject-sampling corrige a distribuição mas a taxa de aceitação despenca |
| Equivalência top-k vs rollout sequencial | obrigatória, 100% |

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

**Ciclo atual — n-gram speculative decoding:**

1. `runtime/speculative.py`: tabela n-gram do prompt + janela recente
   (hash → candidatos next-token, O(1));
2. Draft de k≤4 → verificação num único forward batched reaproveitando `prefill()`;
   aceitar prefixo com correção reject-sampling para temperatura >0;
3. Flag opt-in `--speculative k`, fallback automático ao rollout sequencial;
4. **Pré-registro no `ANALISE.md §21` antes do bench** (protocolo §20.3):
   - H1: speedup ≥1.3× em greedy/min_p baixo sobre texto repetitivo;
   - H2: ≤5% de delta em temp ≥0.9 (nem positivo nem negativo relevante);
   - H3: equivalência top-k 100% vs rollout sequencial nos tokens aceitos.
5. Métricas: ms/token efetivo por bucket de repetibilidade + top-k match.

**Backlog mantido (dependências externas, inalterado):**
ORT `cp313t` wheel → reavaliar migração free-threaded da API;
int8 estático + VNNI → exige hardware/dataset fora do escopo atual;
generalização do salience (seeds, checkpoints, formatos de agulha lexicalmente banais).

**Rejeitados com justificativa documentada (esta análise):**
Mamba/SSM híbrido, MiniCache, FP4 simulado, XNOR/BitNet post-hoc, Saguaro SSD.

---

## Princípio aplicado

O projeto construiu sua credibilidade documentando resultados negativos com mecanismo
explicado (int8 §11, Cython §13-14, ONNX-contrib, Nexus pré-fix §20.1). Esta análise aplica
o mesmo critério na direção oposta: **recusar propostas atraentes quando conflitam com
evidência interna ou rompem o ciclo transplant→validação→inferência** — e aceitar a única
que se apoia em ativos que o projeto já possui.

*"Só entregar se ganho superar claramente a perda"* — critério acordado no §11, válido
para features tanto quanto para microbenchmarks.
