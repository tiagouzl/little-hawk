# Transformer vs RWKV-7 — comparação controlada (CPU-only)

Pesos RWKV-7: `BlinkDL/rwkv7-g1` (`rwkv7-g1d-0.1b-20260129-ctx8192.pth`,
SHA256 `e10d7b19…`, Apache-2.0, 402 tensores bf16). Conversão torch→`.npz`
fp32 **offline** em `scripts/convert_rwkv.py`; runtime (`engine/rwkv_engine.py`,
`engine/rwkv_tokenizer.py`) é NumPy puro, sem torch/CUDA. Referência NumPy:
`RWKV-v7/rwkv_v7_numpy.py` + `run_rwkv7_qwen35.py` (oficiais).

Ficha RWKV (medida do checkpoint): 191.084.544 params, L12-D768-H12-N64,
vocab 65536, ctx treino 8192, tokenizer `rwkv_vocab_v20230424` (vendorizado,
byte-idêntico ao PIPELINE — `tests/test_rwkv.py`). Transformador:
SmolLM2-135M-Instruct, 176.097.600 params contados do `.npz` (embedding e
lm_head não amarrados), ctx treino 8192 (config HF).

## Como reproduzir

```bash
# 1. pesos (381 MB) + vocab + referência (data/ é local, não versionado)
venv/bin/python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('BlinkDL/rwkv7-g1','rwkv7-g1d-0.1b-20260129-ctx8192.pth',local_dir='data/rwkv')"
curl -sL -o data/rwkv/rwkv_vocab_v20230424.txt https://raw.githubusercontent.com/BlinkDL/RWKV-LM/main/RWKV-v7/rwkv_vocab_v20230424.txt
curl -sL -o data/rwkv/rwkv_v7_numpy.py https://raw.githubusercontent.com/BlinkDL/RWKV-LM/main/RWKV-v7/rwkv_v7_numpy.py
# 2. conversão offline (único passo que usa torch)
venv/bin/python scripts/convert_rwkv.py --src data/rwkv/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --out data/rwkv/rwkv7_g1d_01b_fp32.npz
# 3. testes (§5-6)  4. benchmark comum  5. needle  6. memória externa
venv/bin/python -m pytest tests/test_rwkv.py -q
venv/bin/python benchmarks/transformer_vs_rwkv/run_benchmark.py --contexts 128 512 1024 2048 --reps 3 --gen 32
venv/bin/python benchmarks/transformer_vs_rwkv/needle_3arm.py --contexts 600 1200 --arms fifo salience rwkv
venv/bin/python benchmarks/transformer_vs_rwkv/memory_probe.py
```

## Medido — bateria limpa (swap zerado via popup, governor performance, n=3, gen 32)

Memória: 2×4 GB DDR4-3200 @2667 **dual-channel** (`dmidecode`, ChannelA+B).
Throttle +35388 na bateria (sustentado ainda throttla; absolutos são piso).
Microcode: `intel-microcode` 3.20260210 já instalado e ativo (rev 0x100);
GDS segue `Vulnerable` mesmo assim — sem mitigação ativa, sem custo AVX2,
sem necessidade de reboot/re-baseline por essa causa.

| ctx | modelo | prefill_ms (média, n=3) | tok/s | RSS pós-load MB |
| --: | --- | ---: | ---: | ---: |
| 128 | transformer | 897 | 7.21 | 1013.3 |
| 512 | transformer | 3403 | 4.57 | 1089.7 |
| 1024 | transformer | 116219 | 4.41 | 1089.9 |
| 2048 | transformer | 625979¹ | 2.49 | 1094.1 |
| 128 | rwkv7 | 35973 | 3.33 | 1176.4 |
| 512 | rwkv7 | 142048 | 3.63 | 1176.9 |
| 1024 | rwkv7 | 243974 | 4.23 | 1148.6 |
| 2048 | rwkv7 | 547133¹ | 4.03 | 961.1 |

¹ 2048 do regime anterior (swap 90%, powersave) — estrutura mantida, absoluto
pendente de repetição limpa.

Leituras estruturais (robustas a throttling, que só desloca absolutos):
- **Prefill**: transformer escala bem até 512 (batched), depois cai no
  caminho chunked (>max_cap: resto sequencial — 2048 leva ~10 min).
  RWKV é linear desde o início (~260–500 ms/token).
- **Decode**: empate técnico em toda a faixa (2,4–4,3 tok/s ambos).
- **RSS não cresce com o contexto em nenhum dos dois**: transformer
  1093→1094 (streaming O(1) confirmado empiricamente), RWKV oscila
  961–1178 sem tendência 128→2048 (estado constante confirmado).

## Memória externa A/B/C/D (multi-turno PT, greedy)

| caso | transformer | rwkv7 |
| --- | --- | --- |
| A fato no histórico | miss | HIT |
| B histórico+eviction | miss | HIT |
| C `MEMORY:` injetado | miss | HIT |
| D keyword-retrieval | miss | HIT |

Confundidor documentado: transformer-instruct rodou sem template ChatML
(texto corrido `Usuário:/Assistente:`) e greedy degenera nesse formato
(mesma patologia do `chat_baseline` 2/40); RWKV-base lida bem com
continuação pura. O miss do transformer é falha de formato+decode, não
prova de incapacidade de retrieval.

## Needle HAWK-7319 @600 (instruct + pergunta explícita)

Transformer fifo e nexus-salience: 0/12 greedy **e** 0/12 sampled
(temp 0.7/min_p 0.05, seed 42). Com sampling o modelo entende a pergunta
("A: 1.5.202" — chuta números) mas não recupera: falha de capacidade do
checkpoint 135M, não só de decode. RWKV: **0/6** (d0.1 0/2, d0.5 0/2,
d0.9 0/2) — `Resposta: O identificador secreto é "11…/"12…`: entende a
pergunta e chuta número errado, mesma classe de falha do transformer.
Grade completa sem morte de worker desta vez (swap sob pressão, sem OOM).

## Limitações

- Equivalência NumPy-vs-`rwkv` oficial: top1 ✓, top5 4/5, mediana_rel 0.011
  (tolerâncias em `tests/test_rwkv.py`; resíduo é ordem-de-op fp amplificada
  pela recorrência, piores dims no rank 2000+ — irrelevantes p/ greedy).
- RSS isolado por modelo (Q2), 1024/2048 e bateria limpa 3 reps pendentes.
- Nenhum terceiro algoritmo de eviction implementado (§15 bloqueado até aqui).

## Respostas (Q1–Q10) — só o medido

- **Q1** estado constante? **Sim**: RWKV RSS 1177→1178→1108→961 MB em
  128→512→1024→2048 (sem tendência; variação é ruído de alocador).
- **Q2** diferença real de RSS: RWKV ~1175 MB vs transformer ~1015–1094 MB
  (RWKV +~160 MB: 191M params fp32 vs 176M).
- **Q3** prefill: transformer vence até 512 (~13 ms/tok batched); além do
  max_cap cai no chunked sequencial. RWKV linear sempre (~260–500 ms/tok).
- **Q4** decode: empate técnico em toda a faixa (2,4–4,3 tok/s ambos).
- **Q5** TTFT = prefill (tabela acima); RWKV paga prompt inteiro sequencial.
- **Q6** bytes/token: tráfego medido **pendente**; residente teórico KV
  70,8 MB vs estado RWKV 2,43 MB (29×). Banda memcpy 7,9 GB/s.
- **Q7** RWKV em PT-BR: sim — 4/4 na sonda multi-turno, respostas fluentes.
- **Q8** KV+eviction vs estado recorrente: needle inconclusivo (ambos 0 —
  HAWK-7319@600 está acima da capacidade dos dois checkpoints 0.1B);
  memória externa em formato livre favorece RWKV (4/4 vs 0/4).
- **Q9** `MEMORY:` injetado: HIT no RWKV; transformer inconclusivo (formato).
- **Q10** continuar recorrentes? Sim, pelo eixo **memória/fluência** —
  com o preço do prefill sequencial mapeado. §15: **não implementar**
  terceiro evictor — o needle não separou fifo de salience aqui (0/12
  ambos, piso de capacidade), logo não há justificativa medida.

