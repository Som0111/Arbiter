"""Write 20 random RequestLog entries for dashboard testing.

Usage: python scripts/generate_sample_logs.py [--force]
Refuses to touch an existing non-empty log unless --force is given (which overwrites it).
"""
import os
import random
import sys
import uuid
from datetime import UTC, datetime, timedelta

from arbiter.logger import RequestLog, log_path, write_log
from arbiter.router import MANUAL_COST_PER_TOKEN, TIER_MODEL_MAP, _model_list

TENANTS = ["tenant_a", "tenant_b", "tenant_test"]
LATENCY_MS = {"simple": (200, 900), "standard": (500, 2500), "complex": (1200, 7000)}


def make_entry(rng: random.Random, now: datetime) -> RequestLog:
    tier = rng.choices(["simple", "standard", "complex"], weights=[8, 8, 4])[0]
    model = _model_list()[TIER_MODEL_MAP[tier][0]]["model"]
    fallback = tier == "complex" and rng.random() < 0.15
    if fallback:
        model = _model_list()[TIER_MODEL_MAP[tier][1]]["model"]
    tokens_in, tokens_out = rng.randint(10, 600), rng.randint(5, 700)
    rates = MANUAL_COST_PER_TOKEN[model]
    cost = tokens_in * rates["in"] + tokens_out * rates["out"]
    hit = rng.random() < 0.15
    ts = now - timedelta(minutes=rng.randint(0, 24 * 60))
    return RequestLog(
        request_id=f"req_{uuid.uuid4().hex[:12]}",
        tenant_id=rng.choice(TENANTS),
        tier=tier,
        model_used=model,
        tokens_in=0 if hit else tokens_in,
        tokens_out=0 if hit else tokens_out,
        cost_usd=0.0 if hit else cost,
        latency_ms=float(rng.randint(5, 40) if hit else rng.randint(*LATENCY_MS[tier])),
        cache_hit=hit,
        fallback_triggered=fallback and not hit,
        success=True,
        error=None,
        timestamp=ts.isoformat(),
        saved_usd=cost if hit else 0.0,
        tier_capped=False,
        classify_source=rng.choice(["rules", "rules", "sklearn"]),
    )


if __name__ == "__main__":
    path = log_path()
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    if exists and "--force" not in sys.argv:
        sys.exit(f"{path} already has data; pass --force to overwrite it.")
    if exists:
        os.remove(path)
    rng, now = random.Random(7), datetime.now(UTC)
    for _ in range(20):
        write_log(make_entry(rng, now), path)
    print(f"Wrote 20 sample entries to {path}")
