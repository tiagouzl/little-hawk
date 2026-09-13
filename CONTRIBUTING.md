# Como contribuir

1. Faça um fork e crie um branch: `git checkout -b minha-feature`
2. Implemente com testes, se possível
3. Valide antes do PR:
   ```bash
   make check
   make lint
   make fmt-check
   make test
   ```
4. Abra um Pull Request explicando motivação e mudanças

Notas:

- Python 3.10+, linha até 120 colunas (`ruff.toml`).
- `engine/` e `runtime/` seguem estilo compacto intencional — não reformate.
- Pesos `.npz` não são versionados; `_meta.json` são.
- Issues com reprodução mínima são bem-vindas.
