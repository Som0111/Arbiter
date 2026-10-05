"""Validate data/benchmark/benchmark.json and assert it doesn't leak into the classifier training set."""
import json
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
BENCHMARK = DATA / "benchmark" / "benchmark.json"
TRAIN = DATA / "classifier_train.json"
FIELDS = {"id", "tier", "prompt", "expected_keywords", "quality_bar", "max_cost_usd",
          "max_latency_ms"}
TIERS = ("simple", "standard", "complex")


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def main() -> int:
    tasks = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    errors = []
    for t in tasks:
        tid = t.get("id", "?")
        if missing := FIELDS - t.keys():
            errors.append(f"{tid}: missing {sorted(missing)}")
            continue
        if t["tier"] not in TIERS:
            errors.append(f"{tid}: bad tier {t['tier']!r}")
        if not (isinstance(t["quality_bar"], int) and 1 <= t["quality_bar"] <= 5):
            errors.append(f"{tid}: quality_bar must be int 1-5")
        if not t["prompt"].strip():
            errors.append(f"{tid}: empty prompt")
    errors += [f"duplicate id {i}" for i, n in Counter(t.get("id") for t in tasks).items() if n > 1]

    # Leakage: the classifier must never be trained on prompts it is evaluated on.
    train = json.loads(TRAIN.read_text(encoding="utf-8"))
    bench_prompts = {normalize(t["prompt"]): t.get("id", "?") for t in tasks if "prompt" in t}
    errors += [f"LEAK: benchmark task {bench_prompts[normalize(r['prompt'])]} also in {TRAIN.name}"
               for r in train if normalize(r["prompt"]) in bench_prompts]

    if errors:
        print("Validation FAILED:\n  " + "\n  ".join(errors))
        return 1
    counts = Counter(t["tier"] for t in tasks)
    print(f"Validation passed. simple={counts['simple']} standard={counts['standard']} "
          f"complex={counts['complex']} total={len(tasks)}")
    print(f"No leakage: 0 of {len(train)} training prompts appear in the benchmark.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
