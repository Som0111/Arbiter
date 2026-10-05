"""LLM calls via LiteLLM with retry/backoff and cross-provider fallback."""
import os
import time
from dataclasses import dataclass
from functools import lru_cache

import litellm
import yaml

from arbiter.config import ROOT, get_settings

TIER_MODEL_MAP = {
    "simple": ["tier1"],
    "standard": ["tier2", "tier2_fallback"],
    "complex": ["tier3", "tier3_fallback"],
}

# USD per token; used when litellm's cost DB doesn't know the model (returns 0 / raises).
MANUAL_COST_PER_TOKEN = {
    "groq/openai/gpt-oss-20b": {"in": 0.075e-6, "out": 0.30e-6},
    "gemini/gemini-3.1-flash-lite": {"in": 0.25e-6, "out": 1.50e-6},
    "openai/qwen/qwen3.8-27b": {"in": 0.80e-6, "out": 4.0e-6},  # Groq-hosted; not in LiteLLM's DB
    "groq/openai/gpt-oss-120b": {"in": 0.15e-6, "out": 0.60e-6},
}

ATTEMPTS = 3
TIMEOUT_S = 15


class GatewayError(Exception):
    """Every model in the tier's chain was exhausted."""


@dataclass
class LLMResult:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    fallback_triggered: bool


@lru_cache
def _model_list() -> dict[str, dict]:
    """alias (tier1...) -> {"model": ..., "api_key": ...} from config/litellm_config.yaml."""
    with open(ROOT / "config" / "litellm_config.yaml") as f:
        entries = yaml.safe_load(f)["model_list"]
    return {e["model_name"]: e["litellm_params"] for e in entries}


def _resolve_key(ref: str) -> str:
    """'os.environ/GROQ_API_KEY' -> value from the environment, else from Settings (.env)."""
    name = ref.removeprefix("os.environ/")
    return os.environ.get(name) or getattr(get_settings(), name.lower(), "")


def _cost(response, model: str, tokens_in: int, tokens_out: int) -> float:
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:  # noqa: BLE001 - any cost-lookup failure falls back to the manual map
        cost = 0.0
    if cost:
        return float(cost)
    rates = MANUAL_COST_PER_TOKEN.get(model)
    return tokens_in * rates["in"] + tokens_out * rates["out"] if rates else 0.0


def call_llm(
    tier: str, prompt: str, max_tokens: int, aliases: list[str] | None = None
) -> LLMResult:
    """`aliases` overrides the tier's model chain (e.g. to pin a single model for a baseline)."""
    aliases = aliases or TIER_MODEL_MAP[tier]
    for alias in aliases:
        params = _model_list()[alias]
        model = params["model"]
        for attempt in range(ATTEMPTS):
            last = attempt == ATTEMPTS - 1
            try:
                response = litellm.completion(
                    model=model,
                    api_key=_resolve_key(params["api_key"]),
                    api_base=params.get("api_base"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    timeout=TIMEOUT_S,
                )
            except litellm.RateLimitError:
                if not last:
                    time.sleep(2**attempt)  # 1s, 2s
            except litellm.Timeout:
                if not last:
                    time.sleep(1)
            except Exception:  # noqa: BLE001
                break  # non-retryable (auth, bad request, provider down): next model
            else:
                tokens_in = response.usage.prompt_tokens
                tokens_out = response.usage.completion_tokens
                return LLMResult(
                    text=response.choices[0].message.content or "",
                    model=model,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost_usd=_cost(response, model, tokens_in, tokens_out),
                    fallback_triggered=alias != aliases[0],
                )
    raise GatewayError("all models exhausted")
