import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "run_eval", Path(__file__).resolve().parent.parent / "scripts" / "run_eval.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

TASKS = [
    {"id": "s1", "tier": "simple", "quality_bar": 3},
    {"id": "m1", "tier": "standard", "quality_bar": 3},
    {"id": "c1", "tier": "complex", "quality_bar": 4},
    {"id": "c2", "tier": "complex", "quality_bar": 4},
]


def base(i, cost, lat):
    return {"id": i, "cost_usd": cost, "latency_ms": lat}


def rout(i, cost, lat, tier, fb=False, hit=False):
    return {"id": i, "cost_usd": cost, "latency_ms": lat, "tier_assigned": tier,
            "model_used": "m", "fallback_triggered": fb, "cache_hit": hit}


def score(i, b, r):
    return {"id": i, "baseline_score": b, "routed_score": r, "rationale": "why",
            "a_is_baseline": True}


def test_parse_judge_averages_dimensions_and_tolerates_fences():
    text = ('```json\n{"A": {"correctness": 5, "relevance": 4, "completeness": 3},'
            ' "B": {"correctness": 1, "relevance": 1, "completeness": 2}, "rationale": "A wins"}\n```')
    out = ev.parse_judge(text)
    assert out["A"] == pytest.approx(4.0)
    assert out["B"] == pytest.approx(4 / 3)
    assert out["rationale"] == "A wins"


def test_parse_judge_rejects_garbage():
    with pytest.raises(ValueError):
        ev.parse_judge("I cannot grade this")


def test_compute_metrics_known_values():
    baseline = [base("s1", 1.0, 100), base("m1", 1.0, 200), base("c1", 1.0, 300),
                {"id": "c2", "error": "boom"}]
    routed = [rout("s1", 0.2, 50, "simple"), rout("m1", 0.5, 100, "complex", fb=True),
              rout("c1", 1.0, 300, "complex", hit=True), rout("c2", 1.0, 1, "complex")]
    scores = [score("s1", 4.0, 4.0), score("m1", 4.0, 3.0), score("c1", 4.0, 4.5)]
    m = ev.compute_metrics(TASKS, baseline, routed, scores)
    assert m["n_compared"] == 3  # c2 skipped: baseline errored
    assert m["routing_accuracy"] == pytest.approx(2 / 3 * 100)  # m1 misrouted
    assert m["baseline_cost"] == 3.0 and m["routed_cost"] == pytest.approx(1.7)
    assert m["cost_savings_pct"] == pytest.approx((3.0 - 1.7) / 3.0 * 100)
    assert m["quality_retention"] == pytest.approx(2 / 3 * 100)  # m1 dropped by 1.0 > 0.5
    assert m["routed_meets_quality_bar"] == 100.0  # 4.0>=3, 3.0>=3, 4.5>=4
    assert m["fallback_rate"] == pytest.approx(100 / 3)
    assert m["cache_hit_rate"] == pytest.approx(100 / 3)
    assert [f["id"] for f in m["failures"]] == ["m1"]
    assert m["skipped"] == [{"id": "c2", "step": "baseline", "error": "boom"}]
    assert m["per_tier"]["simple"]["savings_pct"] == pytest.approx(80.0)


def test_zero_baseline_cost_does_not_divide_by_zero():
    m = ev.compute_metrics(TASKS[:1], [base("s1", 0.0, 1)], [rout("s1", 0.0, 1, "simple")], [])
    assert m["cost_savings_pct"] == 0.0 and m["quality_retention"] == 0.0


def test_report_renders_all_sections():
    baseline = [base("s1", 1.0, 100)]
    routed = [rout("s1", 0.5, 50, "simple")]
    m = ev.compute_metrics(TASKS[:1], baseline, routed, [score("s1", 4.0, 3.0)])
    md = ev.render_report(m, "gemini/x")
    for heading in ("## Summary", "## Per-tier breakdown", "## Failure analysis"):
        assert heading in md
    assert "50.0%" in md and "s1" in md
