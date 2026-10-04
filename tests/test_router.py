from types import SimpleNamespace
from unittest.mock import patch

import litellm
import pytest

from arbiter.router import GatewayError, call_llm


def ok(tin=10, tout=20, text="answer"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=tin, completion_tokens=tout),
    )


def rate_limit():
    return litellm.RateLimitError("slow down", llm_provider="groq", model="m")


def timeout():
    return litellm.Timeout("timed out", model="m", llm_provider="groq")


@pytest.fixture(autouse=True)
def no_sleep_no_cost_db():
    with patch("arbiter.router.time.sleep") as sleep, \
            patch("litellm.completion_cost", return_value=0.0):
        yield sleep


def test_simple_calls_tier1_model():
    with patch("litellm.completion", return_value=ok()) as comp:
        r = call_llm("simple", "hi", 64)
    assert comp.call_args.kwargs["model"] == "groq/openai/gpt-oss-20b"
    assert comp.call_args.kwargs["max_tokens"] == 64
    assert (r.text, r.fallback_triggered) == ("answer", False)


def test_standard_calls_tier2_model():
    with patch("litellm.completion", return_value=ok()) as comp:
        call_llm("standard", "hi", 64)
    assert comp.call_args.kwargs["model"] == "gemini/gemini-3.1-flash-lite"


def test_complex_calls_tier3_first():
    with patch("litellm.completion", return_value=ok()) as comp:
        r = call_llm("complex", "hi", 64)
    assert comp.call_args.kwargs["model"] == "gemini/gemini-3.8-flash"
    assert r.model == "gemini/gemini-3.8-flash"
    assert r.fallback_triggered is False


def test_tier3_rate_limited_retries_3x_then_falls_back(no_sleep_no_cost_db):
    side = [rate_limit(), rate_limit(), rate_limit(), ok()]
    with patch("litellm.completion", side_effect=side) as comp:
        r = call_llm("complex", "hi", 64)
    models = [c.kwargs["model"] for c in comp.call_args_list]
    assert models == ["gemini/gemini-3.8-flash"] * 3 + ["groq/openai/gpt-oss-120b"]
    assert r.model == "groq/openai/gpt-oss-120b"
    assert r.fallback_triggered is True
    assert [c.args[0] for c in no_sleep_no_cost_db.call_args_list] == [1, 2]  # backoff


def test_all_models_timeout_raises_gateway_error():
    with patch("litellm.completion", side_effect=timeout()) as comp, pytest.raises(GatewayError):
        call_llm("complex", "hi", 64)
    assert comp.call_count == 6  # 3 attempts x 2 models


def test_non_retryable_error_skips_to_next_model():
    with patch("litellm.completion", side_effect=[RuntimeError("auth"), ok()]) as comp:
        r = call_llm("complex", "hi", 64)
    assert comp.call_count == 2
    assert r.fallback_triggered is True


def test_single_model_tier_exhausts_to_gateway_error():
    with patch("litellm.completion", side_effect=RuntimeError("down")), pytest.raises(GatewayError):
        call_llm("simple", "hi", 64)


@pytest.mark.parametrize("tier,model,tin,tout,expected", [
    ("simple", "groq/openai/gpt-oss-20b", 1_000_000, 1_000_000, 0.075 + 0.30),
    ("standard", "gemini/gemini-3.1-flash-lite", 1_000_000, 1_000_000, 0.25 + 1.50),
    ("complex", "gemini/gemini-3.8-flash", 1_000_000, 1_000_000, 0.75 + 3.75),
])
def test_manual_cost_when_litellm_returns_zero(tier, model, tin, tout, expected):
    with patch("litellm.completion", return_value=ok(tin, tout)):
        r = call_llm(tier, "hi", 64)
    assert r.model == model
    assert r.cost_usd == pytest.approx(expected)


def test_fallback_model_manual_cost():
    side = [RuntimeError("x"), ok(1_000_000, 1_000_000)]
    with patch("litellm.completion", side_effect=side):
        r = call_llm("complex", "hi", 64)
    assert r.cost_usd == pytest.approx(0.15 + 0.60)


def test_litellm_cost_preferred_when_available():
    with patch("litellm.completion", return_value=ok()), \
            patch("litellm.completion_cost", return_value=0.123):
        assert call_llm("simple", "hi", 64).cost_usd == 0.123
