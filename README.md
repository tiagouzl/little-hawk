[![CI](https://github.com/tiagouzl/little-hawk/actions/workflows/ci.yml/badge.svg)](https://github.com/tiagouzl/little-hawk/actions/workflows/ci.yml)
# 🦅 Little Hawk

> Motor de inferência LLM streaming construído do zero em Python/NumPy.  
> Sem PyTorch. Sem CUDA. Sem frameworks. Só matemática.

```
attention and memory are the foundations of →
  the ability to remember things.
  memory is a part of the brain and memory...

30 camadas · 112ms/token · CPU · 8GB RAM · zero evicções
```

---

## O que é

Little Hawk é uma implementação manual e completa de inferência autoregressiva para modelos da família LLaMA/Qwen. O objetivo não foi criar mais um wrapper — foi entender e reconstruir cada peça da pilha de inferência sem abstrações escondendo a matemática.

O motor implementa:

- **StreamingKVCache O(1)** — cache de atenção de tamanho constante que nunca cresce, baseado na arquitetura do [StreamingLLM](https://arxiv.org/abs/2309.17453)
- **Attention Sinks** — slots imutáveis que ancoram a atenção no token inicial, permitindo geração infinita sem degradação
- **RoPE com Position Freeze** — posições codificadas congeladas na fase estacionária, preservando as distâncias relativas corretas
- **GQA → MHA expansion** — expansão de grouped-query attention para multi-head attention por repetição de grupos
- **SwiGLU MLP** — feed-forward com ativação `silu(gate) * up @ down`, exatamente como implementado no LLaMA 2+
- **BPE tokenizer real** — integração com o `tokenizer.json` nativo dos modelos via biblioteca Rust (`tokenizers`)

---

## Arquitetura do Cache

O problema central de inferência streaming é que o KV cache cresce linearmente com o número de tokens gerados. Em 10.000 tokens, a memória explode.

```
Cache tradicional:
  token 1   → KV[1]
  token 2   → KV[1], KV[2]
  token N   → KV[1], KV[2], ..., KV[N]   ← O(N) RAM

Little Hawk StreamingKVCache:
  ┌──────────────────────────────────────────────────┐
  │  Sink (4 slots)  │     Janela circular (508)      │
  │  tok[0..3] fixos │  tok[N-508..N] rotacionando    │
  └──────────────────────────────────────────────────┘
  Sempre 512 slots. Sempre. win_ptr avança módulo 508.
```

**Position Freeze:** quando o cache satura, as posições RoPE congelam. Q permanece em `pos=512`, sink em `0..3`, janela em `4..511`. O modelo sempre "enxerga" uma janela de tamanho fixo no mesmo lugar do espaço posicional — sem drift de atenção.

---

## Modelos suportados

| Modelo | Params | RAM (.npz) | Latência (CPU) | Idiomas |
|---|---|---|---|---|
| SmolLM-135M | 135M | ~330 MB | ~100ms/token | EN |
| Qwen2.5-0.5B | 500M | ~900 MB | ~400ms/token | EN, PT, ZH, multilíngue |

---

## Instalação

```bash
git clone https://github.com/tiagouzl/little-hawk
cd little-hawk

python3 -m venv venv
source venv/bin/activate

pip install numpy safetensors huggingface_hub tokenizers
```

---

## Uso rápido

### Modo demo (sem download, pesos aleatórios)

Valida o pipeline completo imediatamente:

```bash
python little_hawk_cli.py --prompt "hello world"
```

Validação rápida (sem inferência) — garante que os scripts compilam:

```bash
python -m py_compile little_hawk_cli.py little_hawk_transplant.py little_hawk_transplant_qwen.py
```

Lint opcional (requere `ruff`):

```bash
ruff check .
```

Formatação opcional (requere `ruff`):

```bash
ruff format .
```

### SmolLM-135M (inglês)

```bash
# Transplante — baixa ~540 MB, extrai 30 camadas (~3 min)
python little_hawk_transplant.py --layers 30

# Inferência
python little_hawk_cli.py \
  --weights little_hawk_weights.npz \
  --prompt "attention and memory are the foundations of"
```

### Qwen2.5-0.5B (multilíngue / português)

```bash
# Transplante — baixa ~1 GB, extrai 24 camadas
python little_hawk_transplant_qwen.py

# Inferência
python little_hawk_cli.py \
  --weights qwen_weights.npz \
  --prompt "atenção e memória são os fundamentos"
```

---

## Parâmetros CLI

```
--weights       Caminho para o .npz (omitir = modo demo)
--prompt        Texto de entrada
--max-tokens    Tokens a gerar (padrão: 80)
--temperature   Temperatura de amostragem (padrão: 0.7)
--top-k         Top-K sampling (padrão: 40)
--top-p         Nucleus sampling (padrão: 0.92)
--rep-penalty   Penalidade de repetição (padrão: 1.15; 1.0 desativa)
--no-panel      Sem painel de telemetria em tempo real
```

---

## API FastAPI (servidor opcional)

Suba o servidor:

```bash
make run-api
```

Chame o endpoint `/generate` (SSE):

```bash
curl -N -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"atenção e memória","max_tokens":32}'
```

Saída chega token a token (text/event-stream). Se `little_hawk_weights.npz` não existir, o servidor cai em modo demo.

---

## Telemetria em tempo real

O CLI exibe um painel atualizado a cada 8 tokens:

```
────────────────────────────────────────────
  LITTLE HAWK  30L · Attn+SwiGLU
────────────────────────────────────────────
  step          72
  win_ptr       76  / 508
  evicções       0
  latência   83.5 ms

  cache [sink|janela]
  [▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  73/512 slots  (5% sink)

  sink L0 (tok[0])
  [░░░░░░░░░░░░░░░░░░░░] 0.8%

  último token
  'memory'
────────────────────────────────────────────
```

`evicções = 0` significa que o cache O(1) está funcionando — nenhuma realocação de memória ocorreu.

---

## Dependências

| Biblioteca | Para quê |
|---|---|
| `numpy` | Toda a álgebra linear |
| `safetensors` | Leitura dos pesos do HuggingFace sem torch |
| `huggingface_hub` | Download dos modelos |
| `tokenizers` | BPE tokenizer Rust nativo |

Nenhum PyTorch. Nenhuma GPU.

---

## Hardware de referência

Todos os benchmarks foram medidos em:

```
Acer Aspire A515-54
CPU: Intel Core i5-10210U (4 cores, sem AVX-512)
RAM: 8 GB DDR4-2666
GPU: nenhuma (Intel UHD Graphics integrada, não usada)
OS:  Linux Mint 21 XFCE
```

---

## Estrutura do projeto

```
little-hawk/
├── little_hawk_cli.py              # Motor de inferência + CLI
├── little_hawk_transplant.py       # Extrator SmolLM-135M → .npz
├── little_hawk_transplant_qwen.py  # Extrator Qwen2.5-0.5B → .npz
├── requirements.txt
└── README.md
```

Os arquivos `.npz` e `_meta.json` gerados pelos transplants não são versionados (`.gitignore`). Cada usuário extrai localmente a partir dos modelos em cache do HuggingFace.

---

## Referências

- [StreamingLLM — Xiao et al., 2023](https://arxiv.org/abs/2309.17453) — base teórica do Attention Sink e StreamingKVCache
- [LLaMA 2 — Touvron et al., 2023](https://arxiv.org/abs/2307.09288) — arquitetura RMSNorm + SwiGLU + RoPE + GQA
- [RoPE — Su et al., 2021](https://arxiv.org/abs/2104.09864) — Rotary Position Embedding
- [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) — modelo doador principal
- [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) — modelo multilíngue

---

## Licença

MIT
