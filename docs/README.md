# Documentação Little Hawk 🦅

**Little Hawk** é um motor de inferência autoregressiva para LLMs (LLaMA e Qwen) implementado em Python e NumPy puro, sem dependência de runtimes pesados durante a geração.

---

## 1. Arquitetura Modular

- **`engine/`**: Núcleo matemático puro de álgebra linear e Transformer:
  - `engine.py`: `MultiLayerEngine` para orquestração de camadas e alocação do `StreamingKVCache` $O(1)$.
  - `transformer.py`: `LlamaLayer` com RMSNorm, RoPE (`rotate_half`), projeções de atenção e MLP SwiGLU.
- **`runtime/`**: Mecânica de inferência, amostragem e processamento de texto:
  - `inference.py`: `LittleHawkInference`, sampler (`temperature`, `top_k`, `top_p`, `rep_penalty`, `min_p`) e telemetria.
  - `tokenizer.py`: `BPETokenizer` e `StreamDecoder` incremental byte-safe UTF-8.
- **`transplants/`**: Extratores de pesos a partir de arquivos `.safetensors` do HuggingFace (`smollm.py`, `qwen.py`).
- **`api/`**: Servidor FastAPI assíncrono com endpoints `/health` e `/generate` via Server-Sent Events (SSE).
- **`cli/`**: Interface de linha de comando com subcomandos `infer`, `transplant` e `api`.
- **`utils/`**: Constantes visuais ANSI, configurações padrão e validadores.

---

## 2. Conceito-Chave: Memória $O(1)$ (StreamingLLM)

O Little Hawk adota uma janela de atenção com memória fixa:
- **Attention Sinks ($S = 4$)**: Os primeiros 4 tokens são mantidos imutáveis para reter a massa de atenção residual.
- **Janela Deslizante ($W = 508$)**: Fila circular operada sem realocação (`np.append` nunca é chamado).
- **Position Freeze**: Congelamento de índices angulares para consultas após saturação do buffer ($>512$ tokens), prevenindo drift posicional.

---

## 3. Uso Rápido

### Instalação

```bash
pip install -e .
```

### Transplante de Pesos

```bash
# SmolLM-135M (padrão: 4 camadas, use 30 para o modelo completo)
python -m cli.main transplant --model smollm-135m --layers 4

# Qwen2.5-0.5B (padrão: 24 camadas)
python -m cli.main transplant --model qwen2.5-0.5b --layers 24
```

### Inferência via CLI

```bash
# Modo demo (pesos sintéticos)
python -m cli.main infer --prompt "Hello world"

# Com pesos reais
python -m cli.main infer --weights little_hawk_weights.npz --prompt "Once upon a time"
```

### Servidor de Streaming SSE

```bash
python -m cli.main api --weights little_hawk_weights.npz --port 8000
```
Para testar interativamente, abra `sse_demo.html` em seu navegador.

