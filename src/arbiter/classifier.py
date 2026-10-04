"""Tier classifier: confident rules first, sklearn for ambiguous prompts."""
import json
import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import tiktoken

from arbiter.config import ROOT

MODEL_PATH = ROOT / "data" / "classifier.pkl"
TRAIN_PATH = ROOT / "data" / "benchmark" / "classifier_train.json"
TIER_ORDER = ("simple", "standard", "complex")

CODE_KEYWORDS = ("def ", "class ", "import ", " for ", "while ", "return ", "async def",
                 "```python")
COMPLEX_KEYWORDS = ("def ", " class ", "```", "bug", "debug", "implement", "write a function")
SIMPLE_KEYWORDS = ("summarize", "translate", "what is", "define", "yes or no")
REASONING_KEYWORDS = ("explain", "compare", "why", "analyze", "difference between")
SIMPLE_MARKERS = ("summarize", "translate", "what is", "define")


@lru_cache
def _encoding():
    return tiktoken.get_encoding("cl100k_base")  # loaded lazily: first use may download the BPE


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def _has(text: str, keywords: tuple[str, ...]) -> bool:
    return any(k in text for k in keywords)


def rule_tier(prompt: str) -> str | None:
    """First of the 4 confident rules that matches, else None (ambiguous)."""
    text = prompt.lower()
    tokens = count_tokens(prompt)
    if tokens < 50 and not _has(text, CODE_KEYWORDS):
        return "simple"
    if tokens > 800:
        return "complex"
    if _has(text, COMPLEX_KEYWORDS):
        return "complex"
    if _has(text, SIMPLE_KEYWORDS):
        return "simple"
    return None  # token-range / fallback rules are not confident -> sklearn


def features(prompt: str) -> list[float]:
    text = prompt.lower()
    words = re.findall(r"\S+", prompt)
    return [
        count_tokens(prompt),
        int(_has(text, CODE_KEYWORDS)),
        int(_has(text, REASONING_KEYWORDS)),
        int(_has(text, SIMPLE_MARKERS)),
        max(1, len(re.findall(r"[.!?]+", prompt))),
        float(np.mean([len(w) for w in words])) if words else 0.0,
    ]


def train_model(train_path: Path = TRAIN_PATH, model_path: Path = MODEL_PATH) -> float:
    """Fit on the labelled benchmark, save to model_path, return 5-fold CV accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = json.loads(Path(train_path).read_text(encoding="utf-8"))
    X = np.array([features(r["prompt"]) for r in rows])
    y = np.array([r["tier"] for r in rows])
    # token_count dwarfs the other features, so scale inside the pipeline (also inside CV folds)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    accuracy = float(cross_val_score(model, X, y, cv=cv).mean())
    model.fit(X, y)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return accuracy


@lru_cache
def _load_model(path: str):
    return joblib.load(path)


def classify_with_source(prompt: str) -> tuple[str, str]:
    """Returns (tier, source) where source is "rules", "sklearn" or "default"."""
    if tier := rule_tier(prompt):
        return tier, "rules"
    if not Path(MODEL_PATH).exists():
        return "standard", "default"  # never fail a request because the model file is missing
    return str(_load_model(str(MODEL_PATH)).predict([features(prompt)])[0]), "sklearn"


def classify(prompt: str) -> str:
    return classify_with_source(prompt)[0]


def apply_tier_cap(tier: str, tier_cap: str) -> tuple[str, bool]:
    """Downgrade tier to the tenant's cap. Returns (tier, tier_capped)."""
    if TIER_ORDER.index(tier) > TIER_ORDER.index(tier_cap):
        return tier_cap, True
    return tier, False
