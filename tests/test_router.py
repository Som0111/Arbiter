import os
from types import SimpleNamespace
from unittest.mock import patch

import litellm
import pytest

from arbiter import router
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
def no_sleep():
    with patch("arbiter.router.time.sleep") as sleep:
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
    assert comp.call_args.kwargs["model"] == "openai/qwen/qwen3.8-27b"
    assert r.model == "openai/qwen/qwen3.8-27b"
    assert r.fallback_triggered is False


def test_tier3_rate_limited_retries_3x_then_falls_back(no_sleep):
    side = [rate_limit(), rate_limit(), rate_limit(), ok()]
    with patch("litellm.completion", side_effect=side) as comp:
        r = call_llm("complex", "hi", 64)
    models = [c.kwargs["model"] for c in comp.call_args_list]
    assert models == ["openai/qwen/qwen3.8-27b"] * 3 + ["groq/openai/gpt-oss-120b"]
    assert r.model == "groq/openai/gpt-oss-120b"
    assert r.fallback_triggered is True
    assert [c.args[0] for c in no_sleep.call_args_list] == [1, 2]  # backoff


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
    ("complex", "openai/qwen/qwen3.8-27b", 1_000_000, 1_000_000, 0.80 + 4.0),
])
def test_cost_comes_from_config_pricing(tier, model, tin, tout, expected):
    with patch("litellm.completion", return_value=ok(tin, tout)):
        r = call_llm(tier, "hi", 64)
    assert r.model == model
    assert r.cost_usd == pytest.approx(expected)


def test_fallback_model_cost_uses_fallback_pricing():
    side = [RuntimeError("x"), ok(1_000_000, 1_000_000)]
    with patch("litellm.completion", side_effect=side):
        r = call_llm("complex", "hi", 64)
    assert r.cost_usd == pytest.approx(0.15 + 0.60)


def test_cost_uses_values_from_config_not_hardcoded(monkeypatch):
    custom = {a: {**p, "pricing": {"input_per_1m": 1.0, "output_per_1m": 2.0}}
              for a, p in router.model_list().items()}
    monkeypatch.setattr(router, "model_list", lambda: custom)
    with patch("litellm.completion", return_value=ok(1_000_000, 500_000)):
        assert call_llm("simple", "hi", 64).cost_usd == pytest.approx(1.0 + 1.0)


def test_every_configured_model_has_pricing():
    models = router.model_list()
    assert {"tier1", "tier2", "tier3", "tier3_fallback", "tier2_fallback"} <= set(models)
    for alias, params in models.items():
        assert params["pricing"]["input_per_1m"] > 0, alias
        assert params["pricing"]["output_per_1m"] > 0, alias


def _write_config(tmp_path, pricing_lines):
    path = tmp_path / "cfg.yaml"
    head = [
        "model_list:",
        "  - model_name: tier1",
        "    litellm_params:",
        "      model: groq/x",
        '      api_key: "os.environ/GROQ_API_KEY"',
    ]
    path.write_text(os.linesep.join(head + pricing_lines) + os.linesep)
    return path


@pytest.mark.parametrize("pricing_lines", [
    [],
    ["    pricing:", "      input_per_1m: 0.1"],
    ["    pricing:", "      input_per_1m: 0.1", "      output_per_1m: abc"],
    ["    pricing:", "      input_per_1m: -1", "      output_per_1m: 1"],
])
def test_missing_or_invalid_pricing_raises_at_config_load(tmp_path, pricing_lines):
    with pytest.raises(ValueError, match="tier1.*pricing"):
        router.load_model_list(_write_config(tmp_path, pricing_lines))


def test_valid_pricing_loads(tmp_path):
    lines = ["    pricing:", "      input_per_1m: 0.1", "      output_per_1m: 0.2"]
    models = router.load_model_list(_write_config(tmp_path, lines))
    assert models["tier1"]["pricing"] == {"input_per_1m": 0.1, "output_per_1m": 0.2}


def test_app_refuses_to_start_when_pricing_missing(monkeypatch):
    from arbiter.api import create_app

    def broken():
        raise ValueError("model 'tier9' is missing a valid pricing.input_per_1m")

    monkeypatch.setattr("arbiter.api.model_list", broken)
    with pytest.raises(ValueError, match="pricing"):
        create_app({})


def test_tier3_passes_groq_openai_compatible_api_base():
    with patch("litellm.completion", return_value=ok()) as comp:
        call_llm("complex", "hi", 64)
    assert comp.call_args.kwargs["api_base"] == "https://api.groq.com/openai/v1"


def test_standard_falls_back_to_groq_when_gemini_quota_exhausted():
    quota = litellm.RateLimitError("daily quota", llm_provider="gemini", model="m")
    with patch("litellm.completion", side_effect=[quota] * 3 + [ok()]) as comp:
        r = call_llm("standard", "hi", 64)
    assert comp.call_count == 4
    assert (r.model, r.fallback_triggered) == ("groq/openai/gpt-oss-120b", True)
