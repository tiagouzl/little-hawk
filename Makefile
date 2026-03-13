PY ?= python3

check:
	$(PY) -m py_compile little_hawk/*.py transplants/*.py

lint:
	@command -v ruff >/dev/null 2>&1 && ruff check . || (echo "ruff não encontrado; instale com 'pip install ruff' para lint" && exit 1)

fmt:
	@command -v ruff >/dev/null 2>&1 && ruff format . || (echo "ruff não encontrado; instale com 'pip install ruff' para formatar" && exit 1)

fmt-check:
	@command -v ruff >/dev/null 2>&1 && ruff format --check . || (echo "ruff não encontrado; instale com 'pip install ruff' para formatar" && exit 1)

run-smollm:
	$(PY) -m little_hawk.cli --weights little_hawk_weights.npz

run-qwen:
	$(PY) -m little_hawk.cli --weights qwen_weights.npz

.PHONY: check lint fmt fmt-check run-smollm run-qwen
