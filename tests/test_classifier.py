import json
import shutil
from pathlib import Path

import pytest

from arbiter import classifier
from arbiter.classifier import (
    MODEL_PATH,
    TRAIN_PATH,
    apply_tier_cap,
    classify,
    classify_with_source,
    train_model,
)
from arbiter.config import load_tenants

CODE_PROMPT = (
    "Implement a Python class `LRUCache` with get and put methods running in O(1). "
    "Here is my partial attempt, please fix and complete it:\n\n```python\n"
    "class LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n"
    "        self.data = {}\n\n    def get(self, key):\n        return self.data.get(key, -1)\n```\n"
    "Use collections.OrderedDict, add type hints, and include unit tests covering eviction order, "
    "updating an existing key, and capacity of one."
)
QA_PROMPT = (
    "The city council approved a new bus route connecting the airport to the central station, "
    "with stops at the stadium and the university campus. Service runs every fifteen minutes "
    "between six in the morning and midnight, and a single ticket costs three dollars. The "
    "council expects the route to carry about four thousand passengers a day by next spring, "
    "and says the project is funded by a regional transport grant rather than local taxes. "
    "Based on this, how often do buses run and who is paying for the route?"
)


@pytest.fixture
def trained(tmp_path, monkeypatch):
    path = tmp_path / "clf.pkl"
    acc = train_model(model_path=path)
    monkeypatch.setattr(classifier, "MODEL_PATH", path)
    classifier._load_model.cache_clear()
    return acc


def test_short_factual_is_simple():
    assert classify("What is the capital of Japan?") == "simple"


def test_long_code_is_complex():
    assert classify(CODE_PROMPT) == "complex"
    assert classify_with_source(CODE_PROMPT)[1] == "rules"


def test_very_long_prompt_is_complex():
    assert classify("lorem ipsum dolor sit amet " * 400) == "complex"


def test_medium_qa_is_standard_via_sklearn(trained):
    assert classify_with_source(QA_PROMPT) == ("standard", "sklearn")


def test_ambiguous_without_model_defaults_to_standard(tmp_path, monkeypatch):
    monkeypatch.setattr(classifier, "MODEL_PATH", tmp_path / "missing.pkl")
    assert classify_with_source(QA_PROMPT) == ("standard", "default")


def test_tenant_a_cap_downgrades_code_question():
    cap = load_tenants()["tenant_a"]["tier_cap"]
    tier = classify(CODE_PROMPT)
    assert apply_tier_cap(tier, cap) == ("standard", True)


def test_tier_cap_noop_cases():
    assert apply_tier_cap("simple", "standard") == ("simple", False)
    assert apply_tier_cap("complex", "complex") == ("complex", False)


def test_cv_accuracy_at_least_70(trained):
    assert trained >= 0.70


# ---- Fix 1: training data is separate from the benchmark ----

def _norm(text):
    return " ".join(text.lower().split())


def test_benchmark_prompts_absent_from_training_data():
    bench = {_norm(t["prompt"]) for t in json.loads(
        (TRAIN_PATH.parent / "benchmark" / "benchmark.json").read_text(encoding="utf-8"))}
    train = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    assert len(train) == 60
    assert not [r["prompt"][:40] for r in train if _norm(r["prompt"]) in bench]


def test_classifier_trains_without_benchmark_data(tmp_path, monkeypatch):
    source = Path(classifier.__file__).read_text(encoding="utf-8")
    assert "benchmark" not in str(TRAIN_PATH) and "benchmark.json" not in source
    # train from a copy of the training file alone, with the benchmark made unreachable
    only_train = tmp_path / "train.json"
    shutil.copy(TRAIN_PATH, only_train)
    real_open = open

    def guarded_open(path, *a, **k):
        assert "benchmark" not in str(path), f"benchmark read during training: {path}"
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", guarded_open)
    assert train_model(only_train, tmp_path / "m.pkl") >= 0.5
    assert MODEL_PATH.name == "classifier.pkl"


# ---- Fix 1: adversarial boundary cases ----

def test_summarize_plus_bug_fix_is_complex():
    assert classify("Summarize this Python program and explain the bug.") == "complex"
    assert classify("Summarize this script and fix the bug in the loop.") == "complex"


def test_what_is_inside_long_multistep_reasoning_is_not_simple():
    prompt = ("A warehouse ships 240 boxes on Monday and 15 percent more on Tuesday, then loses "
              "10 boxes to damage each day. What is the total number of undamaged boxes shipped "
              "over the two days? Work through it step by step and verify each intermediate "
              "result before giving the final answer, since the totals feed into a larger "
              "inventory calculation for the whole week.")
    assert classify(prompt) != "simple"


def test_short_prompt_with_deep_reasoning_is_complex():
    assert classify("Prove that the square root of 3 is irrational.") == "complex"
    assert classify("Derive the sum of the first n odd numbers.") == "complex"


def test_long_prompt_with_only_simple_factual_content_is_simple():
    prompt = ("Our book club meets every second Thursday at the community centre, and this month "
              "we are reading a collection of short stories set in coastal towns. Several members "
              "brought snacks last time, and we are trying to decide on a theme for the next "
              "evening, maybe something maritime. Before that, a quick trivia question for the "
              "group chat: what is the chemical symbol for sodium?")
    assert classify(prompt) == "simple"


def test_define_is_simple_and_production_implementation_is_complex():
    assert classify("Define recursion.") == "simple"
    assert classify("Write a complete production implementation of a rate limiter.") == "complex"
    assert classify("Explain recursion with an implementation and complexity analysis.") == "complex"
