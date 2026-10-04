import pytest

from arbiter import classifier
from arbiter.classifier import apply_tier_cap, classify, classify_with_source, train_model
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
