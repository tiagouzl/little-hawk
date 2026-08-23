"""Testes da API FastAPI em modo demo (sem pesos reais)."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["LITTLE_HAWK_WEIGHTS"] = "/nonexistent/little_hawk_weights.npz"

from fastapi.testclient import TestClient

import api


@pytest.fixture(scope="module")
def client():
    api.load_model("/nonexistent/little_hawk_weights.npz")  # força modo demo
    return TestClient(api.app)


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
