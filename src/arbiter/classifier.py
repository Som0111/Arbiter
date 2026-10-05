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
# Deliberately NOT under data/benchmark/: training data must never overlap the eval set.
TRAIN_PATH = ROOT / "data" / "classifier_train.json"
TIER_ORDER = ("simple", "standard", "complex")

CODE_KEYWORDS = ("def ", "class ", "import ", " for ", "while ", "return ", "async def",
                 "```python")
COMPLEX_KEYWORDS = ("def ", " class ", "```", "bug", "debug", "implement", "write a function",
                    "step by step", "step-by-step", "prove ", "derive", "derivation")
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
    """First of the 4 confident rules that matches, else None (ambiguous).

    Complexity keywords are checked before the short-prompt rule, otherwise a short
    "summarize this program and fix the bug" would be called simple.
    """
    text = prompt.lower()
    tokens = count_tokens(prompt)
    if tokens > 800:
        return "complex"
    if _has(text, COMPLEX_KEYWORDS):
        return "complex"
    if tokens < 50 and not _has(text, CODE_KEYWORDS):
        return "simple"
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


def _xy(train_path: Path):
    rows = json.loads(Path(train_path).read_text(encoding="utf-8"))
    return (np.array([features(r["prompt"]) for r in rows]),
            np.array([r["tier"] for r in rows]))


def _pipeline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # token_count dwarfs the other features, so scale inside the pipeline (also inside CV folds)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))


def _cv():
    from sklearn.model_selection import StratifiedKFold

    return StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def train_model(train_path: Path = TRAIN_PATH, model_path: Path = MODEL_PATH) -> float:
    """Fit on the labelled training set, save to model_path, return 5-fold CV accuracy."""
    from sklearn.model_selection import cross_val_score

    X, y = _xy(train_path)
    model = _pipeline()
    accuracy = float(cross_val_score(model, X, y, cv=_cv()).mean())
    model.fit(X, y)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return accuracy


def cv_report(train_path: Path = TRAIN_PATH) -> dict:
    """5-fold cross-validated accuracy plus per-tier and macro-averaged precision/recall."""
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.model_selection import cross_val_predict

    X, y = _xy(train_path)
    pred = cross_val_predict(_pipeline(), X, y, cv=_cv())
    p, r, _, support = precision_recall_fscore_support(y, pred, labels=list(TIER_ORDER),
                                                       zero_division=0)
    return {
        "accuracy": float((pred == y).mean()),
        "per_tier": {t: {"precision": float(p[i]), "recall": float(r[i]), "support": int(support[i])}
                     for i, t in enumerate(TIER_ORDER)},
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
    }


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
