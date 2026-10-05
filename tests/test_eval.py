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


def _scenario():
    baseline = [base("s1", 1.0, 1), base("m1", 1.0, 1), base("c1", 2.0, 1), base("c2", 2.0, 1)]
    routed = [rout("s1", 0.1, 1, "simple"), rout("m1", 0.5, 1, "complex"),
              rout("c1", 1.0, 1, "complex", fb=True), rout("c2", 0.5, 1, "standard")]
    scores = [score(t["id"], 4.0, 4.0) for t in TASKS]
    return ev.compute_metrics(TASKS, baseline, routed, scores)


def test_confusion_matrix_values_and_rows_sum_to_tier_counts():
    m = _scenario()
    assert m["confusion"] == [[1, 0, 0], [0, 0, 1], [0, 1, 1]]
    expected_rows = [sum(t["tier"] == tier for t in TASKS) for tier in ev.TIERS]
    assert [sum(row) for row in m["confusion"]] == expected_rows
    assert sum(map(sum, m["confusion"])) == m["n_compared"] == 4


def test_routing_precision_recall_f1():
    prf = _scenario()["routing_prf"]
    assert (prf["simple"]["precision"], prf["simple"]["recall"], prf["simple"]["f1"]) == (1, 1, 1)
    assert (prf["standard"]["precision"], prf["standard"]["recall"]) == (0, 0)
    assert prf["complex"]["precision"] == pytest.approx(0.5)
    assert prf["complex"]["recall"] == pytest.approx(0.5)
    assert prf["complex"]["f1"] == pytest.approx(0.5)
    assert _scenario()["routing_macro"]["recall"] == pytest.approx((1 + 0 + 0.5) / 3)


def test_report_contains_confusion_matrix_prf_and_fallback_sections():
    m = _scenario()
    md = ev.render_report(m, "model-x", {"Judge": "j", "Benchmark": "40 tasks"})
    assert "## Routing confusion matrix" in md
    assert "Predicted" in md and "Actual S [ 1]" in md and "Actual C [ 0]" in md
    assert "Actual M [ 0]" in md  # standard is M, not a second S
    assert "| **macro avg** |" in md
    assert "## Fallback usage" in md and "## Evaluation metadata" in md
    assert "| Judge | j |" in md


def test_fallback_reporting_and_savings_excluding_fallback():
    fb = _scenario()["fallback"]
    assert fb["count"] == 1 and fb["by_expected_tier"]["complex"] == 1
    assert fb["tasks"] == [{"id": "c1", "expected_tier": "complex", "model": "m"}]
    assert fb["n_without_fallback"] == 3
    # without c1: baseline 1+1+2=4, routed 0.1+0.5+0.5=1.1
    assert fb["savings_without_fallback_pct"] == pytest.approx((4 - 1.1) / 4 * 100)


def test_judge_prompt_has_anchors_and_parse_keeps_dimension_scores():
    for needle in ("correctness", "Do NOT reward length", "cut off", "REFERENCE HINTS"):
        assert needle in ev.JUDGE_PROMPT
    text = ('{"A": {"correctness": 5, "relevance": 5, "completeness": 2},'
            ' "B": {"correctness": 3, "relevance": 4, "completeness": 4}, "rationale": "r"}')
    out = ev.parse_judge(text)
    assert out["A_dims"] == {"correctness": 5.0, "relevance": 5.0, "completeness": 2.0}
    assert out["B"] == pytest.approx(11 / 3)
    clipped = ev.parse_judge(text.replace('"correctness": 5', '"correctness": 9'))
    assert clipped["A_dims"]["correctness"] == 5.0  # out-of-range scores are clamped
