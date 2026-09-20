# Gates: Little Hawk revisão 2026-09-20

OWNS: runtime/inference.py, engine/rwkv_engine.py, api/server.py, cli/main.py, engine/__init__.py, engine/eviction.py, Makefile, Dockerfile, REVISAO_2026-09-20.md

Scope: aplicar correções P0-P2 e divergências baratas da REVISAO_2026-09-20 sem quebrar os 59 testes verdes.

- [x] G0: this ledger states outcomes that can fail
  CHECK: node /home/tiago/.agents/skills/unlazy/scripts/gate-lint.mjs GATES.md
  EXPECT: LINT OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/tiago/Downloads/Little Hawk; path=3fcc6661ca5f/22 entries; EXPECT=matched; output-sha256=48630b7361dd44ee870917b12c3d19b9d7bdea738aaca16bb04d4cab83b772d2; output-bytes=8

- [x] G1: generate com speculative_k retorna str como o caminho normal
  CHECK: venv/bin/python scripts/verify_g1_speculative.py
  EXPECT: G1 SPECULATIVE RETURN OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/tiago/Downloads/Little Hawk; path=3fcc6661ca5f/22 entries; EXPECT=matched; output-sha256=85df223890964bd704d4b71e55f11162124688fffe81981230bf3d61c5e758e8; output-bytes=25

- [x] G2: Rwkv7Engine sem atribuicao acidental e com eos distinto
  CHECK: venv/bin/python scripts/verify_g2_rwkv.py
  EXPECT: G2 RWKV IDS OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/tiago/Downloads/Little Hawk; path=3fcc6661ca5f/22 entries; EXPECT=matched; output-sha256=ba651c20786f4b92f6c274563ee447de69b2fc12e08a2ba4b2358083d0ca8604; output-bytes=15

- [x] G3: erro no worker da API vira evento de erro em vez de travar ate timeout
  CHECK: venv/bin/python scripts/verify_g3_api_error.py
  EXPECT: G3 API ERROR OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/tiago/Downloads/Little Hawk; path=3fcc6661ca5f/22 entries; EXPECT=matched; output-sha256=910b8dc70559d89c42e305c817eb5ae97d0d56b8bcad76dc1105adafe8a95f09; output-bytes=16

- [x] G4: CLI chat cai no demo sem pesos e no-panel respeita speculative sem mutar sys.argv
  CHECK: venv/bin/python scripts/verify_g4_cli.py
  EXPECT: G4 CLI OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/tiago/Downloads/Little Hawk; path=3fcc6661ca5f/22 entries; EXPECT=matched; output-sha256=7e421ac5732dcf30be23ef918ba480552d3f270d3c28fd91a9edfb46d5315c5e; output-bytes=10

- [x] G5: suite de testes continua verde
  CHECK: venv/bin/python -m pytest tests/ -q
  EXPECT: 59 passed
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/tiago/Downloads/Little Hawk; path=3fcc6661ca5f/22 entries; EXPECT=matched; output-sha256=f00099c68e5fb49e49f35b0f1b1ec1064a5a5372703522ee7c372fc1d28187f6; output-bytes=1128

- [x] G6: versao da API e docstrings batem com pyproject 0.9.0
  CHECK: venv/bin/python scripts/verify_g6_versions.py
  EXPECT: G6 VERSIONS OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/tiago/Downloads/Little Hawk; path=3fcc6661ca5f/22 entries; EXPECT=matched; output-sha256=ad9c4d22bb2fa9ff05a28cd16f107b72bdd094dd725e2c23fe4c46787ea2de27; output-bytes=15
