import os

# Must be set before arbiter.api is imported (it builds the app at import time).
os.environ.setdefault("GROQ_API_KEY", "test-groq")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini")
os.environ["ARBITER_API_KEY"] = "test-key"

import pytest
from fastapi.testclient import TestClient

from arbiter.api import create_app

TENANTS = {
    "tenant_a": {"daily_budget_usd": 0.10, "tier_cap": "standard", "rate_limit_rpm": 20},
    "tenant_test": {"daily_budget_usd": 999.0, "tier_cap": "complex", "rate_limit_rpm": 999},
}
HEADERS = {"X-API-Key": "test-key"}


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "requests.jsonl"))
    return create_app(TENANTS)


@pytest.fixture
def client(app):
    return TestClient(app)
