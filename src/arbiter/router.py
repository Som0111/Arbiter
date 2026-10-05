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


CONFIG_PATH = ROOT / "config" / "litellm_config.yaml"


def load_model_list(path=CONFIG_PATH) -> dict[str, dict]:
    """alias (tier1...) -> {"model", "api_key", "pricing": {...}, ...}; raises on missing pricing."""
    with open(path) as f:
        entries = yaml.safe_load(f)["model_list"]
    models = {}
    for e in entries:
        alias, params = e["model_name"], dict(e["litellm_params"])
        pricing = e.get("pricing") or {}
        for field in ("input_per_1m", "output_per_1m"):
            value = pricing.get(field)
            if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"{path}: model '{alias}' ({params.get('model')}) is missing a valid "
                    f"pricing.{field}; add pricing to every model_list entry"
                )
        params["pricing"] = {k: float(pricing[k]) for k in ("input_per_1m", "output_per_1m")}
        models[alias] = params
    return models


@lru_cache
def model_list() -> dict[str, dict]:
    return load_model_list()


def _resolve_key(ref: str) -> str:
    """'os.environ/GROQ_API_KEY' -> value from the environment, else from Settings (.env)."""
    name = ref.removeprefix("os.environ/")
    return os.environ.get(name) or getattr(get_settings(), name.lower(), "")


def token_cost(params: dict, tokens_in: int, tokens_out: int) -> float:
    """USD cost from the model's pricing block in litellm_config.yaml."""
    p = params["pricing"]
    return (tokens_in * p["input_per_1m"] + tokens_out * p["output_per_1m"]) / 1e6


def call_llm(
    tier: str, prompt: str, max_tokens: int, aliases: list[str] | None = None
) -> LLMResult:
    """`aliases` overrides the tier's model chain (e.g. to pin a single model for a baseline)."""
    aliases = aliases or TIER_MODEL_MAP[tier]
    for alias in aliases:
        params = model_list()[alias]
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
                    cost_usd=token_cost(params, tokens_in, tokens_out),
                    fallback_triggered=alias != aliases[0],
                )
    raise GatewayError("all models exhausted")
