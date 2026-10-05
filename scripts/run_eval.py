"""Eval: all-premium baseline vs the routed gateway, scored by an LLM judge.

    python scripts/run_eval.py [--resume] [--limit N]

--resume reuses finished tasks from reports/*.json (errored tasks are retried).
--limit N runs a random N-task sample, for a cheap smoke test.
Writes reports/{baseline_results,routed_results,quality_scores}.json and EVAL_REPORT.md.
"""
import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import litellm
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
BENCHMARK = ROOT / "data" / "benchmark" / "benchmark.json"
TENANT = "tenant_test"
MAX_TOKENS = 2048  # reasoning models spend output tokens thinking; 512 truncates answers
JUDGE_MODEL = "groq/openai/gpt-oss-120b"  # not the baseline model (Gemini free tier is ~15 RPD)
JUDGE_CLIP_CHARS = 5000  # Groq free tier caps ~8k tokens/min, so keep judge prompts small
GEMINI_PAUSE_S, GROQ_PAUSE_S, JUDGE_PAUSE_S, RETRY_SLEEP_S = 2.0, 3.0, 20.0, 60
TIERS = ("simple", "standard", "complex")

JUDGE_PROMPT = """You are a strict, impartial grader. Below is a task and two candidate responses.
Score EACH response from 1 (poor) to 5 (excellent) on correctness, relevance and completeness.

TASK:
{prompt}

RESPONSE A:
{a}

RESPONSE B:
{b}

Reply with ONLY a JSON object, no markdown fences:
{{"A": {{"correctness": n, "relevance": n, "completeness": n}},
  "B": {{"correctness": n, "relevance": n, "completeness": n}},
  "rationale": "<one sentence on the main difference>"}}"""


# ---------- pure helpers (unit-tested) ----------

def parse_judge(text: str) -> dict:
    """Extract {"A": score, "B": score, "rationale": str}; score = mean of the 3 dimensions."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in judge reply: {text[:120]!r}")
    data = json.loads(m.group(0))
    out = {"rationale": str(data.get("rationale", ""))}
    for side in ("A", "B"):
        dims = [float(data[side][k]) for k in ("correctness", "relevance", "completeness")]
        out[side] = float(np.clip(np.mean(dims), 1, 5))
    return out


def _pct(vals, q):
    return float(np.percentile(vals, q)) if len(vals) else 0.0


def _mean(vals):
    return float(np.mean(vals)) if len(vals) else 0.0


def compute_metrics(tasks, baseline, routed, scores) -> dict:
    tmap = {t["id"]: t for t in tasks}
    b = {r["id"]: r for r in baseline if "error" not in r}
    r_ = {r["id"]: r for r in routed if "error" not in r}
    q = {s["id"]: s for s in scores if "error" not in s}
    both = [t["id"] for t in tasks if t["id"] in b and t["id"] in r_]
    base_cost = sum(b[i]["cost_usd"] for i in both)
    routed_cost = sum(r_[i]["cost_usd"] for i in both)

    def savings(bc, rc):
        return (bc - rc) / bc * 100 if bc else 0.0

    scored = [i for i in both if i in q]
    per_tier = {}
    for tier in TIERS:
        ids = [i for i in both if tmap[i]["tier"] == tier]
        sc = [i for i in ids if i in q]
        bc, rc = sum(b[i]["cost_usd"] for i in ids), sum(r_[i]["cost_usd"] for i in ids)
        per_tier[tier] = {
            "tasks": len(ids),
            "routing_accuracy": _mean([r_[i]["tier_assigned"] == tier for i in ids]) * 100,
            "baseline_cost": bc, "routed_cost": rc, "savings_pct": savings(bc, rc),
            "baseline_score": _mean([q[i]["baseline_score"] for i in sc]),
            "routed_score": _mean([q[i]["routed_score"] for i in sc]),
        }
    worse = sorted(
        (i for i in scored if q[i]["routed_score"] < q[i]["baseline_score"]),
        key=lambda i: q[i]["routed_score"] - q[i]["baseline_score"],
    )
    return {
        "n_tasks": len(tasks),
        "n_compared": len(both),
        "routing_accuracy": _mean([r_[i]["tier_assigned"] == tmap[i]["tier"] for i in both]) * 100,
        "baseline_cost": base_cost,
        "routed_cost": routed_cost,
        "cost_savings_pct": savings(base_cost, routed_cost),
        "quality_retention": _mean([q[i]["routed_score"] >= q[i]["baseline_score"] - 0.5
                                    for i in scored]) * 100,
        "routed_meets_quality_bar": _mean([q[i]["routed_score"] >= tmap[i]["quality_bar"]
                                           for i in scored]) * 100,
        "n_scored": len(scored),
        "baseline_p50_ms": _pct([b[i]["latency_ms"] for i in both], 50),
        "baseline_p95_ms": _pct([b[i]["latency_ms"] for i in both], 95),
        "routed_p50_ms": _pct([r_[i]["latency_ms"] for i in both], 50),
        "routed_p95_ms": _pct([r_[i]["latency_ms"] for i in both], 95),
        "fallback_rate": _mean([r_[i]["fallback_triggered"] for i in both]) * 100,
        "cache_hit_rate": _mean([r_[i]["cache_hit"] for i in both]) * 100,
        "per_tier": per_tier,
        "failures": [
            {"id": i, "expected_tier": tmap[i]["tier"], "assigned_tier": r_[i]["tier_assigned"],
             "model": r_[i]["model_used"], "baseline_score": q[i]["baseline_score"],
             "routed_score": q[i]["routed_score"], "rationale": q[i]["rationale"],
             "a_is_baseline": q[i]["a_is_baseline"]}
            for i in worse[:5]
        ],
        "worse_total": len(worse),
        "skipped": [
            {"id": r["id"], "step": step, "error": r["error"]}
            for step, rows in (("baseline", baseline), ("routed", routed), ("judge", scores))
            for r in rows if "error" in r
        ],
    }


def render_report(m: dict, baseline_model: str) -> str:
    L = [
        "# Arbiter - Eval Report", "",
        (f"Compared {m['n_compared']} of {m['n_tasks']} benchmark tasks "
         f"({m['n_scored']} judged). Baseline = every task on `{baseline_model}`. "
         "Costs are estimated at list prices."), "",
        "## Summary", "", "| Metric | Value |", "|---|---|",
        f"| Routing accuracy | {m['routing_accuracy']:.1f}% |",
        f"| Cost savings vs all-premium | {m['cost_savings_pct']:.1f}% |",
        f"| Quality retention (routed >= baseline - 0.5) | {m['quality_retention']:.1f}% |",
        f"| Routed answers meeting task quality_bar | {m['routed_meets_quality_bar']:.1f}% |",
        f"| Baseline p50 / p95 latency | {m['baseline_p50_ms']:.0f}ms / {m['baseline_p95_ms']:.0f}ms |",
        f"| Routed p50 / p95 latency | {m['routed_p50_ms']:.0f}ms / {m['routed_p95_ms']:.0f}ms |",
        f"| Fallback rate | {m['fallback_rate']:.1f}% |",
        f"| Cache hit rate (eval run) | {m['cache_hit_rate']:.1f}% |",
        f"| Total cost: baseline / routed | ${m['baseline_cost']:.5f} / ${m['routed_cost']:.5f} |",
        "", "## Per-tier breakdown (by expected tier)", "",
        ("| Tier | Tasks | Routing acc. | Baseline cost | Routed cost | Savings | "
         "Baseline score | Routed score |"),
        "|---|---|---|---|---|---|---|---|",
    ]
    for tier, t in m["per_tier"].items():
        L.append(f"| {tier} | {t['tasks']} | {t['routing_accuracy']:.0f}% | "
                 f"${t['baseline_cost']:.5f} | ${t['routed_cost']:.5f} | {t['savings_pct']:.1f}% | "
                 f"{t['baseline_score']:.2f} | {t['routed_score']:.2f} |")
    L += ["", "## Failure analysis", "",
          (f"{m['worse_total']} task(s) scored lower when routed; the {len(m['failures'])} "
           "largest gaps:"), ""]
    for f in m["failures"]:
        who = "A=baseline, B=routed" if f["a_is_baseline"] else "A=routed, B=baseline"
        misrouted = "" if f["expected_tier"] == f["assigned_tier"] else " (**misrouted**)"
        L.append(f"- **{f['id']}** expected `{f['expected_tier']}`, routed to "
                 f"`{f['assigned_tier']}`{misrouted} on `{f['model']}`: routed "
                 f"{f['routed_score']:.2f} vs baseline {f['baseline_score']:.2f}. "
                 f"Judge ({who}): {f['rationale']}")
    if not m["failures"]:
        L.append("- None: routed was never scored below baseline.")
    if m["skipped"]:
        L += ["", "## Skipped tasks", ""]
        L += [f"- {s['id']} ({s['step']}): {s['error']}" for s in m["skipped"]]
    return "\n".join(L) + "\n"


# ---------- live steps ----------

def with_retry(call):
    """Run call(); on any failure sleep 60s and retry once; then give back {'error': ...}."""
    for attempt in (1, 2):
        try:
            return call()
        except Exception as e:  # noqa: BLE001 - provider errors vary; the eval must keep going
            if attempt == 2:
                return {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"    failed ({type(e).__name__}); sleeping {RETRY_SLEEP_S}s then retrying once")
            time.sleep(RETRY_SLEEP_S)


def run_step(name, path, tasks, fn, resume, pause):
    done = {}
    if resume and path.exists():
        done = {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))}
    for n, t in enumerate(tasks, 1):
        if t["id"] in done and "error" not in done[t["id"]]:
            continue
        res = with_retry(lambda t=t: fn(t))
        done[t["id"]] = {"id": t["id"], **res}
        path.write_text(json.dumps([done[x["id"]] for x in tasks if x["id"] in done], indent=2),
                        encoding="utf-8")
        status = "ERROR " + res["error"] if "error" in res else "ok"
        print(f"[{name} {n}/{len(tasks)}] {t['id']} {status}")
        time.sleep(pause)
    return [done[t["id"]] for t in tasks]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    REPORTS.mkdir(exist_ok=True)
    os.environ["LOG_PATH"] = str(REPORTS / "eval_requests.jsonl")  # keep eval out of logs/
    if not args.resume and os.path.exists(os.environ["LOG_PATH"]):
        os.remove(os.environ["LOG_PATH"])

    from fastapi.testclient import TestClient

    from arbiter.api import create_app
    from arbiter.config import get_settings
    from arbiter.router import TIER_MODEL_MAP, _model_list, _resolve_key, call_llm

    tasks = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    if args.limit:
        tasks = sorted(random.Random(0).sample(tasks, args.limit), key=lambda t: t["id"])
    baseline_model = _model_list()[TIER_MODEL_MAP["complex"][0]]["model"]
    pause = GEMINI_PAUSE_S if baseline_model.startswith("gemini/") else GROQ_PAUSE_S

    def baseline(task):
        t0 = time.perf_counter()
        r = call_llm("complex", task["prompt"], MAX_TOKENS, aliases=[TIER_MODEL_MAP["complex"][0]])
        return {"response": r.text, "model": r.model, "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out, "cost_usd": r.cost_usd,
                "latency_ms": (time.perf_counter() - t0) * 1000}

    client = TestClient(create_app())
    headers = {"X-API-Key": get_settings().arbiter_api_key}

    def routed(task):
        resp = client.post("/generate", headers=headers, json={
            "prompt": task["prompt"], "tenant_id": TENANT, "max_tokens": MAX_TOKENS})
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:150]}")
        d = resp.json()
        return {"response": d["response"], "model_used": d["model_used"],
                "tier_assigned": d["tier"], "tokens_in": d["tokens_in"],
                "tokens_out": d["tokens_out"], "cost_usd": d["cost_usd"],
                "latency_ms": d["latency_ms"], "cache_hit": d["cache_hit"],
                "fallback_triggered": d["fallback_triggered"]}

    print(f"== STEP 1: baseline on {baseline_model} ==")
    base = run_step("baseline", REPORTS / "baseline_results.json", tasks, baseline,
                    args.resume, pause)
    print("== STEP 2: routed via /generate ==")
    rout = run_step("routed", REPORTS / "routed_results.json", tasks, routed, args.resume,
                    GROQ_PAUSE_S)

    print(f"== STEP 3: judging with {JUDGE_MODEL} ==")
    rng = random.Random(0)
    base_by, rout_by = {r["id"]: r for r in base}, {r["id"]: r for r in rout}
    flips = {t["id"]: rng.random() < 0.5 for t in tasks}  # randomise A/B to cancel position bias

    def judge(task):
        b, r = base_by[task["id"]], rout_by[task["id"]]
        if "error" in b or "error" in r:
            raise RuntimeError("missing baseline or routed response")
        a_is_base = flips[task["id"]]
        a, bb = (b["response"], r["response"]) if a_is_base else (r["response"], b["response"])
        resp = litellm.completion(
            model=JUDGE_MODEL, api_key=_resolve_key("os.environ/GROQ_API_KEY"),
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                prompt=task["prompt"], a=a[:JUDGE_CLIP_CHARS], b=bb[:JUDGE_CLIP_CHARS])}],
            temperature=0, max_tokens=1500, reasoning_effort="low", timeout=60)
        s = parse_judge(resp.choices[0].message.content)
        base_s, rout_s = (s["A"], s["B"]) if a_is_base else (s["B"], s["A"])
        return {"baseline_score": base_s, "routed_score": rout_s,
                "rationale": s["rationale"], "a_is_baseline": a_is_base}

    scores = run_step("judge", REPORTS / "quality_scores.json", tasks, judge, args.resume,
                      JUDGE_PAUSE_S)

    m = compute_metrics(tasks, base, rout, scores)
    (REPORTS / "EVAL_REPORT.md").write_text(render_report(m, baseline_model), encoding="utf-8")
    print(f"\nWrote {REPORTS / 'EVAL_REPORT.md'}")
    print(f"routing accuracy {m['routing_accuracy']:.1f}% | cost savings "
          f"{m['cost_savings_pct']:.1f}% | quality retention {m['quality_retention']:.1f}%")


if __name__ == "__main__":
    main()
