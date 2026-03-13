PY ?= python3

check:
	$(PY) -m py_compile little_hawk_cli.py little_hawk_transplant.py little_hawk_transplant_qwen.py

lint:
	@command -v ruff >/dev/null 2>&1 && ruff check . || (echo "ruff não encontrado; instale com 'pip install ruff' para lint" && exit 1)

fmt:
	@command -v ruff >/dev/null 2>&1 && ruff format . || (echo "ruff não encontrado; instale com 'pip install ruff' para formatar" && exit 1)

fmt-check:
	@command -v ruff >/dev/null 2>&1 && ruff format --check . || (echo "ruff não encontrado; instale com 'pip install ruff' para formatar" && exit 1)

.PHONY: check lint fmt fmt-check
