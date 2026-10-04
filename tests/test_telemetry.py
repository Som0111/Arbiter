from dataclasses import fields
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from conftest import HEADERS

from arbiter.logger import (
    RequestLog,
    init_langfuse,
    read_logs,
    trace_request,
    write_log,
)
from arbiter.router import LLMResult

BODY = {"prompt": "What is 2+2?", "tenant_id": "tenant_test"}


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    llm = MagicMock(return_value=LLMResult("4", "groq/openai/gpt-oss-20b", 5, 1, 0.001, False))
    monkeypatch.setattr("arbiter.api.call_llm", llm)
    monkeypatch.setattr("arbiter.api.embed", MagicMock(return_value=[1.0, 0.0]))


def entry(**kw):
    base = {
        "request_id": "r", "tenant_id": "tenant_a", "tier": "simple", "model_used": "m",
        "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.01, "latency_ms": 100.0,
        "cache_hit": False, "fallback_triggered": False, "success": True, "error": None,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    return RequestLog(**(base | kw))


def test_request_writes_exactly_one_log_line(client):
    assert client.post("/generate", json=BODY, headers=HEADERS).status_code == 200
    assert len(read_logs()) == 1


def test_log_line_has_all_required_fields(client):
    client.post("/generate", json=BODY, headers=HEADERS)
    row = read_logs()[0]
    assert set(row) == {f.name for f in fields(RequestLog)}
    assert {"tier_capped", "cache_hit", "fallback_triggered", "classify_source"} <= set(row)
    assert (row["classify_source"], row["tier_capped"], row["success"]) == ("rules", False, True)


def test_capped_and_cache_hit_flags_logged(client):
    code = {"prompt": "Implement and debug this: def f(): pass", "tenant_id": "tenant_a"}
    client.post("/generate", json=code, headers=HEADERS)
    client.post("/generate", json=code, headers=HEADERS)  # same prompt -> cache hit
    first, second = read_logs()
    assert (first["tier"], first["tier_capped"], first["cache_hit"]) == ("standard", True, False)
    assert (second["cache_hit"], second["cost_usd"], second["saved_usd"]) == (True, 0.0, 0.001)


def test_stats_aggregates_known_log_lines(client):
    old = (datetime.now(UTC) - timedelta(hours=30)).isoformat()
    rows = [
        entry(tier="simple", cost_usd=0.01, latency_ms=100.0),
        entry(tier="simple", cost_usd=0.01, latency_ms=200.0, tenant_id="tenant_b"),
        entry(tier="complex", cost_usd=0.08, latency_ms=300.0, fallback_triggered=True),
        entry(tier="standard", cost_usd=0.0, latency_ms=400.0, cache_hit=True, saved_usd=0.02,
              timestamp=old),
    ]
    for r in rows:
        write_log(r)
    s = client.get("/stats", headers=HEADERS).json()
    assert s["total_requests"] == 4
    assert s["total_cost_usd"] == pytest.approx(0.10)
    assert s["cache_hit_rate"] == 0.25
    assert s["fallback_rate"] == 0.25
    assert s["estimated_saved_usd"] == pytest.approx(0.02)
    assert s["tier_distribution"] == {"simple": 2, "standard": 1, "complex": 1}
    assert s["cost_by_tier"] == pytest.approx({"simple": 0.02, "complex": 0.08, "standard": 0.0})
    assert s["cost_by_tenant"] == pytest.approx({"tenant_a": 0.09, "tenant_b": 0.01})
    assert s["latency_p50_ms"] == 250.0
    assert s["latency_p95_ms"] == pytest.approx(385.0)
    assert s["requests_last_24h"] == 3


def test_stats_skips_torn_lines(client, tmp_path):
    write_log(entry())
    with open(tmp_path / "requests.jsonl", "a") as f:
        f.write('{"request_id": "half-writ')
    assert client.get("/stats", headers=HEADERS).json()["total_requests"] == 1


def test_langfuse_skipped_when_keys_not_set():
    assert init_langfuse(SimpleNamespace(langfuse_public_key="")) is None


def test_langfuse_initialised_when_keys_set():
    cfg = SimpleNamespace(langfuse_public_key="pk", langfuse_secret_key="sk", langfuse_host="h")
    with patch("langfuse.Langfuse") as lf:
        assert init_langfuse(cfg) is lf.return_value
    lf.assert_called_once_with(public_key="pk", secret_key="sk", host="h")


def test_trace_has_classify_cache_and_llm_spans():
    lf = MagicMock()
    trace_request(lf, entry(classify_source="sklearn"), "prompt")
    names = [c.kwargs["name"] for c in lf.start_observation.return_value.start_observation.call_args_list]
    assert names == ["classify", "cache_lookup", "llm_call"]


def test_trace_skips_llm_span_on_cache_hit_and_none_client_is_noop():
    lf = MagicMock()
    trace_request(lf, entry(cache_hit=True), "prompt")
    names = [c.kwargs["name"] for c in lf.start_observation.return_value.start_observation.call_args_list]
    assert names == ["classify", "cache_lookup"]
    trace_request(None, entry(), "prompt")  # no raise


def test_tracing_failure_never_breaks_request(client):
    broken = MagicMock()
    broken.start_observation.side_effect = RuntimeError("langfuse down")
    with patch("arbiter.api.trace_request", wraps=trace_request) as t:
        t.side_effect = lambda lf, e, p: trace_request(broken, e, p)
        assert client.post("/generate", json=BODY, headers=HEADERS).status_code == 200
    assert len(read_logs()) == 1
