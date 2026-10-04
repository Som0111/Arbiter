import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

import numpy as np


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
