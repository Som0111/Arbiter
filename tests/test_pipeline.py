from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from conftest import HEADERS, TENANTS

from arbiter.budget import BudgetTracker, RateLimiter
from arbiter.logger import RequestLog, read_logs, write_log
from arbiter.router import GatewayError, LLMResult


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    mock = MagicMock(return_value=LLMResult("hello", "groq/llama-3.1-8b-instant", 5, 7, 0.001, False))
    monkeypatch.setattr("arbiter.api.call_llm", mock)
    return mock


BODY = {"prompt": "hi", "tenant_id": "tenant_test"}
SCHEMA = {
    "response", "model_used", "tier", "tokens_in", "tokens_out", "cost_usd",
    "latency_ms", "cache_hit", "fallback_triggered", "request_id",
}


def test_valid_key_200_wrong_key_401(client):
    assert client.post("/generate", json=BODY, headers=HEADERS).status_code == 200
    assert client.post("/generate", json=BODY, headers={"X-API-Key": "nope"}).status_code == 401
    assert client.post("/generate", json=BODY).status_code == 401
    assert client.get("/stats").status_code == 401


def test_health_needs_no_key(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["tenant_count"] == len(TENANTS)


def test_generate_schema_and_side_effects(app, client, fake_llm):
    r = client.post("/generate", json=BODY, headers=HEADERS)
    body = r.json()
    assert set(body) == SCHEMA
    assert (body["response"], body["tier"], body["cost_usd"]) == ("hello", "simple", 0.001)
    fake_llm.assert_called_once_with("simple", "hi", 512)
    assert abs(app.state.budget.get_remaining("tenant_test") - (999.0 - 0.001)) < 1e-9
    assert [row["request_id"] for row in read_logs()] == [body["request_id"]]


def test_tier_cap_applied_to_llm_call(client, fake_llm):
    code = {"prompt": "Implement a function to debug this: def f(): pass", "tenant_id": "tenant_a"}
    client.post("/generate", json=code, headers=HEADERS)
    assert fake_llm.call_args.args[0] == "standard"  # complex capped for tenant_a


def test_all_models_exhausted_503_and_logged(client, fake_llm):
    fake_llm.side_effect = GatewayError("all models exhausted")
    r = client.post("/generate", json=BODY, headers=HEADERS)
    assert r.status_code == 503
    row = read_logs()[0]
    assert (row["success"], row["error"]) == (False, "all_models_exhausted")


def test_unknown_tenant_404_and_bad_input_422(client):
    r = client.post("/generate", json={"prompt": "hi", "tenant_id": "ghost"}, headers=HEADERS)
    assert r.status_code == 404
    r = client.post("/generate", json={"prompt": "", "tenant_id": "tenant_a"}, headers=HEADERS)
    assert r.status_code == 422


def test_rate_limit_blocks_after_rpm(client):
    body = {"prompt": "hi", "tenant_id": "tenant_a"}
    codes = [client.post("/generate", json=body, headers=HEADERS).status_code for _ in range(25)]
    assert codes[:20] == [200] * 20
    assert codes[20:] == [429] * 5
    r = client.post("/generate", json=body, headers=HEADERS)
    assert r.json() == {"error": "rate_limited"}


def test_over_budget_429(app, client):
    app.state.budget.record_actual("tenant_a", 0.10)
    r = client.post("/generate", json={"prompt": "hi", "tenant_id": "tenant_a"}, headers=HEADERS)
    assert r.status_code == 429
    assert r.json() == {"error": "budget_exceeded"}


def test_stats_aggregates_log(client):
    now = datetime.now(UTC).isoformat()
    for tier, cost, hit in [("simple", 0.001, False), ("complex", 0.003, True)]:
        write_log(RequestLog("r", "tenant_test", tier, "m", 1, 1, cost, 100.0, hit, False,
                             True, None, now))
    s = client.get("/stats", headers=HEADERS).json()
    assert s["total_requests"] == 2
    assert abs(s["total_cost_usd"] - 0.004) < 1e-9
    assert s["cache_hit_rate"] == 0.5
    assert s["tier_distribution"] == {"simple": 1, "standard": 0, "complex": 1}
    assert s["requests_last_24h"] == 2


def test_stats_empty(client):
    assert client.get("/stats", headers=HEADERS).json()["total_requests"] == 0


def test_budget_reserve_reconcile_and_daily_reset():
    day = ["2026-10-04"]
    b = BudgetTracker(TENANTS, today=lambda: day[0])
    assert b.check_and_reserve("tenant_a", 0.06)
    assert not b.check_and_reserve("tenant_a", 0.06)  # 0.06 + 0.06 > 0.10
    b.record_actual("tenant_a", 0.02, reserved=0.06)  # actual cheaper than estimate
    assert abs(b.get_remaining("tenant_a") - 0.08) < 1e-9
    day[0] = "2026-10-05"
    assert abs(b.get_remaining("tenant_a") - 0.10) < 1e-9


def test_rate_limiter_window_slides():
    t = [0.0]
    rl = RateLimiter(TENANTS, clock=lambda: t[0])
    assert all(rl.check("tenant_a") for _ in range(20))
    assert not rl.check("tenant_a")
    t[0] = 60.0
    assert rl.check("tenant_a")
