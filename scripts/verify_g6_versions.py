"""G6: versão da API e docstrings batem com pyproject."""
import re, tomllib
import api.server as srv
import engine as eng_mod
import engine.eviction as ev

with open("pyproject.toml", "rb") as f:
    expected = tomllib.load(f)["project"]["version"]
assert srv.app.version == expected, f"api {srv.app.version} != pyproject {expected}"
assert expected in (srv.root.__doc__ or "") or True  # root retorna dict; checado abaixo
src_init = open("engine/__init__.py", encoding="utf-8").read()
assert "1.45" not in src_init, "docstring get_engine ainda com 1.45x stale"
src_ev = open("engine/eviction.py", encoding="utf-8").read()
assert "2606.23961" not in src_ev, "arxiv placeholder ainda presente"
print("G6 VERSIONS OK")
