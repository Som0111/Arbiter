from unittest.mock import MagicMock

import pytest
from conftest import TENANTS
from fastapi.testclient import TestClient

from arbiter.api import create_app
from arbiter.router import LLMResult


def test_keys_resolve_to_their_own_tenant(app, client, monkeypatch):
    monkeypatch.setattr("arbiter.api.call_llm", MagicMock(
        return_value=LLMResult("ok", "groq/openai/gpt-oss-20b", 1, 1, 0.001, False)))
    monkeypatch.setattr("arbiter.api.embed", MagicMock(return_value=[1.0, 0.0]))
    r = client.post("/generate", headers={"X-API-Key": "test-key-a"}, json={"prompt": "hi"})
    assert r.status_code == 200
    assert app.state.budget.get_remaining("tenant_a") < 0.10
    assert app.state.budget.get_remaining("tenant_test") == 999.0


def test_invalid_and_missing_key_401(client):
    body = {"prompt": "hi"}
    assert client.post("/generate", headers={"X-API-Key": "bad"}, json=body).status_code == 401
    assert client.post("/generate", json=body).status_code == 401


def test_unknown_tenant_env_var_raises_at_startup(monkeypatch):
    monkeypatch.setenv("ARBITER_KEY_GHOST_TENANT", "arb_ghost_xyz")
    with pytest.raises(RuntimeError, match="not defined in tenants.yaml"):
        create_app(TENANTS)


def test_no_keys_configured_rejects_everything(monkeypatch):
    monkeypatch.delenv("ARBITER_KEY_TENANT_TEST")
    monkeypatch.delenv("ARBITER_KEY_TENANT_A")
    c = TestClient(create_app(TENANTS))
    assert c.post("/generate", headers={"X-API-Key": "test-key"}, json={"prompt": "hi"}).status_code == 401
