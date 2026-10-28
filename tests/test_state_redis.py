import json
import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import fakeredis
import pytest
import redis as redis_lib
from conftest import HEADERS, TENANTS
from fastapi.testclient import TestClient

from arbiter import state_redis
from arbiter.api import create_app
from arbiter.logger import read_logs
from arbiter.router import LLMResult
from arbiter.state_redis import (
    RedisBudgetTracker,
    RedisRateLimiter,
    RedisStateError,
    make_redis_client,
)

LIMITS = {"t": {"daily_budget_usd": 0.10, "tier_cap": "complex", "rate_limit_rpm": 3}}


@pytest.fixture
def r():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(state_redis, "RETRY_SLEEP_S", 0)


# --- rate limiter ---------------------------------------------------------------

def test_rate_limit_exact_boundary(r):
    rl = RedisRateLimiter(LIMITS, r)
    assert [rl.check("t") for _ in range(4)] == [True, True, True, False]


def test_rejected_request_does_not_consume_a_slot(r):
    rl = RedisRateLimiter(LIMITS, r)
    assert all(rl.check("t") for _ in range(3))
    assert not any(rl.check("t") for _ in range(5))
    assert r.zcard("arbiter:rl:t") == 3


def test_requests_sharing_a_timestamp_are_all_counted(r, monkeypatch):
    monkeypatch.setattr(state_redis.time, "time", lambda: 1000.0)
    rl = RedisRateLimiter(LIMITS, r)
    assert [rl.check("t") for _ in range(4)] == [True, True, True, False]
    assert r.zcard("arbiter:rl:t") == 3  # timestamp-only members would have collapsed to 1


def test_window_slides(r, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(state_redis.time, "time", lambda: now[0])
    rl = RedisRateLimiter(LIMITS, r)
    assert all(rl.check("t") for _ in range(3)) and not rl.check("t")
    now[0] = 1061.0
    assert rl.check("t")


def test_concurrent_requests_never_exceed_limit(r):
    rl = RedisRateLimiter({"t": {**LIMITS["t"], "rate_limit_rpm": 10}}, r)
    out = []
    threads = [threading.Thread(target=lambda: out.append(rl.check("t"))) for _ in range(40)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(out) == 10


def test_rate_limiter_redis_failure_raises_state_error(r):
    rl = RedisRateLimiter(LIMITS, r)
    rl._script = MagicMock(side_effect=redis_lib.ConnectionError("down"))
    with pytest.raises(RedisStateError):
        rl.check("t")


# --- budget ---------------------------------------------------------------------

def test_budget_gate_and_remaining(r):
    b = RedisBudgetTracker(LIMITS, r)
    assert b.check_and_reserve("t")
    b.record_actual("t", 0.04)
    assert b.get_remaining("t") == pytest.approx(0.06)
    b.record_actual("t", 0.06)
    assert not b.check_and_reserve("t")


def test_budget_is_a_soft_cap(r):
    b = RedisBudgetTracker(LIMITS, r)
    assert [b.check_and_reserve("t") for _ in range(3)] == [True] * 3  # nothing recorded yet
    for _ in range(3):
        b.record_actual("t", 0.05)  # three in-flight requests land: 0.15 > 0.10 cap
    assert b.get_remaining("t") < 0 and not b.check_and_reserve("t")


def test_budget_check_failure_raises_state_error(r):
    b = RedisBudgetTracker(LIMITS, r)
    b._check_script = MagicMock(side_effect=redis_lib.TimeoutError("slow"))
    with pytest.raises(RedisStateError):
        b.check_and_reserve("t")


def test_record_retries_then_succeeds_without_degrading(r):
    b = RedisBudgetTracker(LIMITS, r)
    real = b._record_script
    calls = []

    def flaky(**kw):
        calls.append(1)
        if len(calls) < 3:
            raise redis_lib.ConnectionError("blip")
        return real(**kw)

    b._record_script = flaky
    b.record_actual("t", 0.02, request_id="req_1")
    assert len(calls) == 3 and not b.degraded
    assert b.get_remaining("t") == pytest.approx(0.08)


def test_record_failure_degrades_logs_critical_and_writes_reconciliation(r, tmp_path, monkeypatch,
                                                                         caplog):
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "requests.jsonl"))
    b = RedisBudgetTracker(LIMITS, r)
    b._record_script = MagicMock(side_effect=redis_lib.ConnectionError("down"))
    with caplog.at_level(logging.CRITICAL), pytest.raises(RedisStateError):
        b.record_actual("t", 0.02, request_id="req_abc")
    assert b._record_script.call_count == state_redis.RECORD_ATTEMPTS
    assert b.degraded
    crit = [x for x in caplog.records if x.levelno == logging.CRITICAL]
    assert len(crit) == 1 and "req_abc" in crit[0].getMessage() and "t" in crit[0].getMessage()
    rows = [json.loads(line) for line in (tmp_path / "unrecorded_costs.jsonl").read_text().splitlines()]
    assert rows[0]["request_id"] == "req_abc" and rows[0]["tenant_id"] == "t"
    assert rows[0]["cost_usd"] == pytest.approx(0.02) and set(rows[0]) == {
        "request_id", "tenant_id", "cost_usd", "timestamp"}
    with pytest.raises(RedisStateError):  # degraded: every later check is rejected
        b.check_and_reserve("t")


# --- startup --------------------------------------------------------------------

def test_unreachable_redis_fails_fast_without_leaking_url():
    with pytest.raises(RuntimeError, match="unreachable at startup") as e:
        make_redis_client("redis://:s3cretpw@127.0.0.1:1/0")
    assert "s3cretpw" not in str(e.value)


def test_malformed_url_fails_fast():
    with pytest.raises(RuntimeError, match="unreachable at startup"):
        make_redis_client("not-a-url")


def test_create_app_aborts_when_redis_unreachable(monkeypatch):
    monkeypatch.setattr("arbiter.api.get_settings",
                        lambda: SimpleNamespace(redis_url="redis://127.0.0.1:1/0"))
    with pytest.raises(RuntimeError, match="unreachable at startup"):
        create_app(TENANTS)


# --- API integration --------------------------------------------------------------

@pytest.fixture
def redis_app(r, tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "requests.jsonl"))
    monkeypatch.setattr("arbiter.state_redis.make_redis_client", lambda url: r)
    monkeypatch.setattr("arbiter.api.get_settings", lambda: SimpleNamespace(redis_url="redis://x"))
    llm = MagicMock(return_value=LLMResult("ok", "groq/openai/gpt-oss-20b", 1, 1, 0.01, False))
    monkeypatch.setattr("arbiter.api.call_llm", llm)
    monkeypatch.setattr("arbiter.api.embed", MagicMock(return_value=[1.0, 0.0]))
    app = create_app(TENANTS)
    app.state.llm = llm
    return app


def test_app_uses_redis_state(redis_app):
    assert isinstance(redis_app.state.budget, RedisBudgetTracker)
    r_ = TestClient(redis_app).post("/generate", json={"prompt": "hi"}, headers=HEADERS)
    assert r_.status_code == 200
    assert redis_app.state.budget.get_remaining("tenant_test") == pytest.approx(999.0 - 0.01)


def test_runtime_redis_failure_returns_503_not_in_memory_fallback(redis_app):
    redis_app.state.limiter._script = MagicMock(side_effect=redis_lib.ConnectionError("down"))
    r_ = TestClient(redis_app).post("/generate", json={"prompt": "hi"}, headers=HEADERS)
    assert r_.status_code == 503 and r_.json()["error"] == "state_unavailable"
    redis_app.state.llm.assert_not_called()


def test_unrecorded_cost_returns_503_degrades_and_blocks_next_request(redis_app, tmp_path, caplog):
    redis_app.state.budget._record_script = MagicMock(side_effect=redis_lib.ConnectionError("down"))
    client = TestClient(redis_app)
    with caplog.at_level(logging.CRITICAL):
        first = client.post("/generate", json={"prompt": "hi"}, headers=HEADERS)
    assert first.status_code == 503 and first.json()["error"] == "state_unavailable"
    assert redis_app.state.llm.call_count == 1  # the provider call did happen
    assert redis_app.state.budget.degraded
    rid = read_logs()[0]["request_id"]
    assert read_logs()[0]["error"] == "cost_record_failed"
    recon = json.loads((tmp_path / "unrecorded_costs.jsonl").read_text().splitlines()[0])
    assert recon["request_id"] == rid and recon["cost_usd"] == pytest.approx(0.01)
    assert any(rid in x.getMessage() for x in caplog.records if x.levelno == logging.CRITICAL)

    second = client.post("/generate", json={"prompt": "another"}, headers=HEADERS)
    assert second.status_code == 503
    assert redis_app.state.llm.call_count == 1  # rejected before reaching the provider


def test_health_reports_redis_status(redis_app, r):
    client = TestClient(redis_app)
    assert client.get("/health").json()["redis"] == "ok"
    redis_app.state.budget._r = MagicMock(ping=MagicMock(side_effect=redis_lib.ConnectionError()))
    body = client.get("/health").json()
    assert body["redis"] == "unavailable" and body["status"] == "ok"


def test_health_has_no_redis_field_in_memory_mode(client):
    assert "redis" not in client.get("/health").json()
