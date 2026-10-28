#!/usr/bin/env python
"""
Async load test for Arbiter. Measures p50/p95 latency and success rate.

Usage:
  python scripts/load_test.py --url http://localhost:8000 --key arb_...
  python scripts/load_test.py --url https://arbiter-dzrv.onrender.com --concurrency 3 --requests 12

Prompts repeat every 6 requests, so with --requests > 6 later ones can be semantic-cache
hits (reported separately) and pull p50 down; compare runs with the same --requests.
"""
import argparse
import asyncio
import statistics
import time

import httpx

PROMPTS = [
    "What is the capital of France?",
    "Translate 'hello world' to Spanish.",
    "Explain the difference between TCP and UDP.",
    "Summarize the key differences between Python lists and tuples.",
    "Write a Python function that merges two sorted lists into one sorted list, with unit tests.",
    "Explain step by step how a transformer attention mechanism works, including the math.",
]


async def single_request(
    client: httpx.AsyncClient, url: str, key: str, prompt: str
) -> dict:
    t0 = time.perf_counter()
    try:
        r = await client.post(
            f"{url}/generate",
            headers={"X-API-Key": key},
            json={"prompt": prompt, "max_tokens": 100},
            timeout=90.0,
        )
        latency = (time.perf_counter() - t0) * 1000
        ok = r.status_code == 200
        body = r.json() if ok else {}
        return {"ok": ok, "status": r.status_code, "latency_ms": latency,
                "tier": body.get("tier", "?"), "cache_hit": body.get("cache_hit", False)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": 0, "latency_ms": (time.perf_counter() - t0) * 1000,
                "tier": "err", "error": type(exc).__name__}


async def run(url: str, key: str, n: int, concurrency: int) -> None:
    sem = asyncio.Semaphore(concurrency)
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n)]

    async def bounded(prompt):
        async with sem:
            return await single_request(client, url, key, prompt)

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[bounded(p) for p in prompts])

    ok = [r for r in results if r["ok"]]
    lats = sorted(r["latency_ms"] for r in ok)
    print(f"\nRequests: {n} | Concurrency: {concurrency} | OK: {len(ok)}/{n}")
    if lats:
        p50 = statistics.median(lats)
        p95 = lats[int(len(lats) * 0.95)] if len(lats) >= 20 else lats[-1]  # <20 samples: max
        print(f"Latency  p50={p50:.0f}ms  p95={p95:.0f}ms  max={lats[-1]:.0f}ms")
        cache_hits = sum(r["cache_hit"] for r in ok)
        print(f"Cache hits: {cache_hits}/{len(ok)}")
        tier_counts: dict[str, int] = {}
        for r in ok:
            tier_counts[r["tier"]] = tier_counts.get(r["tier"], 0) + 1
        print(f"Tier distribution: {tier_counts}")
    errs = [r for r in results if not r["ok"]]
    if errs:
        codes = {r.get("error", r["status"]) for r in errs}
        print(f"Errors ({len(errs)}): {codes}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--key", required=True, help="X-API-Key value (tenant resolved server-side)")
    p.add_argument("--requests", type=int, default=12)
    p.add_argument("--concurrency", type=int, default=3)
    args = p.parse_args()
    asyncio.run(run(args.url, args.key, args.requests, args.concurrency))
