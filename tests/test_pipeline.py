from datetime import UTC, datetime

from conftest import HEADERS, TENANTS

from arbiter.budget import BudgetTracker, RateLimiter
from arbiter.logger import RequestLog, write_log

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


def test_generate_stub_schema(client):
    r = client.post("/generate", json=BODY, headers=HEADERS)
    assert set(r.json()) == SCHEMA
    assert r.json()["response"] == "stub"


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
