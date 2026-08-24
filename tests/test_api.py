"""Testes da API FastAPI em modo demo (sem pesos reais)."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
DEMO_WEIGHTS = "/nonexistent/little_hawk_weights.npz"
os.environ["LITTLE_HAWK_WEIGHTS"] = DEMO_WEIGHTS

from fastapi.testclient import TestClient

import api.server as api


@pytest.fixture(scope="module")
def client():
    # Fixa também o valor importado pelo módulo: evita que um ambiente de
    # execução ou outro teste faça a suíte carregar pesos reais no lifespan.
    api.DEFAULT_WEIGHTS = DEMO_WEIGHTS
    api._hawk = None
    api._tok = None
    api.load_model(DEMO_WEIGHTS)  # força modo demo
    # O extra de desenvolvimento fixa httpx<0.28 para manter compatibilidade
    # com o TestClient/Starlette usado pelo projeto.
    yield TestClient(api.app)


def test_health_demo(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["mode"] == "demo"


def test_generate_requires_prompt(client):
    r = client.post("/generate", json={"prompt": "", "max_tokens": 4})
    assert r.status_code == 400


def test_generate_sse_stream(client):
    with client.stream("POST", "/generate", json={"prompt": "memória", "max_tokens": 5}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        tokens, done = [], False
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[len("data: ") :])
            if payload.get("token") == "[DONE]":
                done = True
                break
            tokens.append(payload["token"])
        assert done
        assert len(tokens) <= 6  # 5 gerados + possível flush final
