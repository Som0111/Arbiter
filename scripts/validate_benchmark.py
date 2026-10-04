"""Validate data/benchmark/benchmark.json and regenerate classifier_train.json."""
import json
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "benchmark"
FIELDS = {"id", "tier", "prompt", "expected_keywords", "quality_bar", "max_cost_usd",
          "max_latency_ms"}
TIERS = ("simple", "standard", "complex")


def main() -> int:
    tasks = json.loads((DATA / "benchmark.json").read_text(encoding="utf-8"))
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
    if errors:
        print("Validation FAILED:\n  " + "\n  ".join(errors))
        return 1
    counts = Counter(t["tier"] for t in tasks)
    train = [{"prompt": t["prompt"], "tier": t["tier"]} for t in tasks]
    (DATA / "classifier_train.json").write_text(
        json.dumps(train, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Validation passed. simple={counts['simple']} standard={counts['standard']} "
          f"complex={counts['complex']} total={len(tasks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
