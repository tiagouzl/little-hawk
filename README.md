
<div align="center">
  <h1>🦅 Little Hawk</h1>
  <p><b>LLM streaming engine em Python/NumPy puro</b></p>
  <p>
    <a href="https://github.com/tiagouzl/little-hawk/actions/workflows/ci.yml"><img src="https://github.com/tiagouzl/little-hawk/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="#licenca"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
    <a href="#api-fastapi"><img src="https://img.shields.io/badge/api-fastapi-blue.svg" alt="FastAPI"></a>
    <a href="#estrutura-do-projeto"><img src="https://img.shields.io/badge/modular-estrutura-blue.svg" alt="Modular"></a>
  </p>
  <p>Sem PyTorch. Sem CUDA. Sem frameworks. Só matemática.</p>
</div>

> Motor de inferência LLM streaming construído do zero em Python/NumPy.  
> Sem PyTorch. Sem CUDA. Sem frameworks. Só matemática.

```
attention and memory are the foundations of →
  the ability to remember things.
  memory is a part of the brain and memory...

30 camadas · 112ms/token · CPU · 8GB RAM · zero evicções
```

---


## Visão geral

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

## Estrutura do projeto
## Como contribuir

Contribuições são bem-vindas! Para colaborar:

1. Fork este repositório
2. Crie um branch: `git checkout -b minha-feature`
3. Faça suas alterações e adicione testes
4. Rode `ruff check .` e `pytest`
5. Envie um PR explicando sua motivação

Sugestões, issues e discussões são incentivadas!

## Links úteis

- [Documentação oficial do FastAPI](https://fastapi.tiangolo.com/)
- [NumPy](https://numpy.org/)
- [HuggingFace Hub](https://huggingface.co/docs/hub/index)
- [Tokenizers](https://github.com/huggingface/tokenizers)


little-hawk/

Projeto modular e organizado para facilitar manutenção, testes e extensibilidade:

```
little-hawk/
├── api/                  # Servidor FastAPI (api/server.py)
├── cli/                  # Interface de linha de comando
├── engine/               # Motor de inferência
├── runtime/              # Tokenizer e núcleo de inferência
├── utils/                # Utilitários e configs
├── scripts/              # Scripts utilitários (ex: download de pesos)
├── examples/             # Exemplos de uso (ex: demo.py)
├── docs/                 # Documentação
├── data/                 # Dados/corpus/modelos (gitignored)
├── tests/                # Testes unitários
├── little_hawk_cli.py    # Wrapper CLI (compatibilidade)
├── setup.py              # Instalação local via pip
└── README.md
```
## Instalação via pip

```bash
pip install -e .
```

## API FastAPI

Suba o servidor:

```bash
uvicorn api.server:app --reload
```

Exemplo de requisição:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"atenção e memória","max_tokens":32}'
```

## Scripts utilitários

Baixe pesos de um modelo HuggingFace:

```bash
python scripts/download_weights.py <repo_id> <filename>
```

## Exemplos

Execute um exemplo de inferência:

```bash
python examples/demo.py
```


---

## Testes

Execute os testes unitários:

```bash
make test
# ou
python3 -m pytest tests/ -v
```

Os testes cobrem:
- Treino e encode/decode do tokenizer BPE
- Inicialização e step do motor
- Persistência de vocabulário

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
python little_hawk_cli.py infer --prompt "hello world"
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
python little_hawk_cli.py infer \
  --weights little_hawk_weights.npz \
  --prompt "attention and memory are the foundations of"
```

### Qwen2.5-0.5B (multilíngue / português)

```bash
# Transplante — baixa ~1 GB, extrai 24 camadas
python little_hawk_transplant_qwen.py

# Inferência
python little_hawk_cli.py infer \
  --weights qwen_weights.npz \
  --prompt "atenção e memória são os fundamentos"
```

---

## Comandos CLI

O CLI agora usa subcomandos para melhor organização:

```bash
# Ver ajuda geral
python little_hawk_cli.py --help

# Ver ajuda de um subcomando específico
python little_hawk_cli.py infer --help
```

### Subcomando `infer`

Executa inferência com o modelo:

```
--weights       Caminho para o .npz (omitir = modo demo)
--prompt        Texto de entrada (obrigatório)
--max-tokens    Tokens a gerar (padrão: 80)
--temperature   Temperatura de amostragem (padrão: 0.7)
--top-k         Top-K sampling (padrão: 40)
--top-p         Nucleus sampling (padrão: 0.92)
--rep-penalty   Penalidade de repetição (padrão: 1.15; 1.0 desativa)
--no-panel      Sem painel de telemetria em tempo real
```

### Subcomando `transplant`

Transplanta pesos de modelo HuggingFace:

```
--model         ID do modelo HF (ex: smollm-135m)
--layers        Número de camadas a extrair
--output        Nome do arquivo .npz de saída
```

### Subcomando `api`

Inicia servidor FastAPI:

```
--weights       Caminho para o .npz
--host          Host do servidor (padrão: 0.0.0.0)
--port          Porta do servidor (padrão: 8000)
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

## Docker

Build e run:

```bash
docker build -t little-hawk .
docker run --rm -p 8000:8000 little-hawk
```

Depois acesse o mesmo endpoint `/generate` via curl ou navegador.

> ⚠️ Pesos não são copiados para a imagem (ver `.dockerignore`). Para servir um modelo real, monte os arquivos de pesos e meta como volume ou aponte `LITTLE_HAWK_WEIGHTS`:
>
> ```bash
> docker run --rm -p 8000:8000 \
>   -v $PWD/little_hawk_weights.npz:/app/little_hawk_weights.npz \
>   -v $PWD/little_hawk_weights_meta.json:/app/little_hawk_weights_meta.json \
>   -e LITTLE_HAWK_WEIGHTS=/app/little_hawk_weights.npz \
>   little-hawk
> ```

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
little-hawk/

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

Distribuído sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
