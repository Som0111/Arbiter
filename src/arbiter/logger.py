import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class RequestLog:
    request_id: str
    tenant_id: str
    tier: str
    model_used: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    cache_hit: bool
    fallback_triggered: bool
    success: bool
    error: str | None
    timestamp: str  # ISO 8601
    saved_usd: float = 0.0  # cost avoided by a cache hit (cost_usd is 0 on hits)
    tier_capped: bool = False  # tier downgraded by the tenant's tier_cap
    classify_source: str = ""  # "rules" | "sklearn" | "default"


def init_langfuse(settings=None):
    """Langfuse client, or None when keys aren't configured. Tracing is optional."""
    from arbiter.config import get_settings

    settings = settings or get_settings()
    if not settings.langfuse_public_key:
        return None
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


def trace_request(lf, entry: RequestLog, prompt: str) -> None:
    """One trace per request: classify -> cache_lookup -> llm_call. Never raises."""
    if lf is None:
        return
    try:
        root = lf.start_observation(
            name="generate", input=prompt, metadata={"tenant_id": entry.tenant_id,
                                                    "request_id": entry.request_id})
        root.start_observation(
            name="classify", input=prompt, output=entry.tier,
            metadata={"source": entry.classify_source, "tier_capped": entry.tier_capped},
        ).end()
        root.start_observation(
            name="cache_lookup", output={"hit": entry.cache_hit}
        ).end()
        if not entry.cache_hit and entry.success:
            root.start_observation(
                name="llm_call", as_type="generation", model=entry.model_used,
                usage_details={"input": entry.tokens_in, "output": entry.tokens_out},
                cost_details={"total": entry.cost_usd},
                metadata={"latency_ms": entry.latency_ms,
                          "fallback_triggered": entry.fallback_triggered},
            ).end()
        root.update(output={"success": entry.success, "error": entry.error})
        root.end()
    except Exception:
        log.debug("langfuse trace failed", exc_info=True)


def log_path() -> str:
    return os.environ.get("LOG_PATH", "logs/requests.jsonl")


def write_log(entry: RequestLog, path: str | None = None) -> None:
    path = path or log_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(asdict(entry)) + "\n")


def read_logs(path: str | None = None) -> list[dict]:
    path = path or log_path()
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn/partial line must not break /stats
    return rows


def compute_stats(rows: list[dict]) -> dict:
    n = len(rows)
    lat = [r["latency_ms"] for r in rows]
    cutoff = datetime.now(UTC) - timedelta(hours=24)
    by_tenant: dict[str, float] = {}
    by_tier: dict[str, float] = {}
    tiers = {"simple": 0, "standard": 0, "complex": 0}
    for r in rows:
        by_tenant[r["tenant_id"]] = by_tenant.get(r["tenant_id"], 0.0) + r["cost_usd"]
        by_tier[r["tier"]] = by_tier.get(r["tier"], 0.0) + r["cost_usd"]
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    return {
        "total_requests": n,
        "total_cost_usd": sum(r["cost_usd"] for r in rows),
        "estimated_saved_usd": sum(r.get("saved_usd", 0.0) for r in rows),
        "cache_hit_rate": sum(r["cache_hit"] for r in rows) / n if n else 0.0,
        "fallback_rate": sum(r["fallback_triggered"] for r in rows) / n if n else 0.0,
        "tier_distribution": tiers,
        "cost_by_tenant": by_tenant,
        "cost_by_tier": by_tier,
        "latency_p50_ms": float(np.percentile(lat, 50)) if n else 0.0,
        "latency_p95_ms": float(np.percentile(lat, 95)) if n else 0.0,
        "requests_last_24h": sum(
            datetime.fromisoformat(r["timestamp"]) >= cutoff for r in rows
        ),
    }
