# Arbiter - Eval Report

Compared 40 of 40 benchmark tasks (40 judged). Baseline = every task on `openai/qwen/qwen3.8-27b`. Costs are estimated at list prices from `config/litellm_config.yaml`.

## Evaluation metadata

| Item | Value |
|---|---|
| Run date (UTC) | 2026-10-05 13:54 |
| Benchmark | 40 tasks, `benchmark.json` sha256 `4b256cd9bdae` |
| Classifier training set | 60 examples, `classifier_train.json` sha256 `4aae2c304508`, separate from the benchmark |
| Train/benchmark overlap | 0 prompts (normalised exact match) |
| Classifier 5-fold CV | accuracy 70.0%, macro precision 0.70, macro recall 0.70 |
| Tier 1 (simple) | `groq/openai/gpt-oss-20b` ($0.075/$0.3 per 1M) |
| Tier 2 (standard) | `gemini/gemini-3.1-flash-lite` ($0.25/$1.5 per 1M) then `groq/openai/gpt-oss-120b` ($0.15/$0.6 per 1M) |
| Tier 3 (complex) | `openai/qwen/qwen3.8-27b` ($0.8/$4 per 1M) then `groq/openai/gpt-oss-120b` ($0.15/$0.6 per 1M) |
| Baseline | every task pinned to `openai/qwen/qwen3.8-27b`, no fallback |
| Judge | `groq/openai/gpt-oss-120b`, rubric v2, A/B order randomised per task, responses clipped to 5000 chars |
| Generation | max_tokens 2048, tenant `tenant_test`, empty semantic cache |

## Summary

| Metric | Value |
|---|---|
| Routing accuracy | 80.0% |
| Cost savings vs all-premium | 37.8% |
| Quality retention (routed >= baseline - 0.5) | 97.5% |
| Routed answers meeting task quality_bar | 100.0% |
| Baseline p50 / p95 latency | 349ms / 12806ms |
| Routed p50 / p95 latency | 3144ms / 22079ms |
| Fallback rate | 10.0% |
| Cache hit rate (eval run) | 0.0% |
| Total cost: baseline / routed | $0.04667 / $0.02901 |

## Per-tier breakdown (by expected tier)

| Tier | Tasks | Routing acc. | Baseline cost | Routed cost | Savings | Baseline score | Routed score |
|---|---|---|---|---|---|---|---|
| simple | 15 | 100% | $0.00305 | $0.00053 | 82.6% | 4.87 | 4.91 |
| standard | 15 | 67% | $0.00719 | $0.00412 | 42.7% | 4.96 | 4.98 |
| complex | 10 | 70% | $0.03643 | $0.02436 | 33.1% | 5.00 | 4.93 |

## Routing confusion matrix

Rows = expected tier, columns = tier the classifier assigned (S = simple, M = standard, C = complex).

```
              Predicted
           S      M      C
Actual S [15]  [ 0]  [ 0]   (n=15)
Actual M [ 2]  [10]  [ 3]   (n=15)
Actual C [ 0]  [ 3]  [ 7]   (n=10)
```

| Tier | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| simple | 0.88 | 1.00 | 0.94 | 15 |
| standard | 0.77 | 0.67 | 0.71 | 15 |
| complex | 0.70 | 0.70 | 0.70 | 10 |
| **macro avg** | 0.78 | 0.79 | 0.78 | 40 |

## Fallback usage

4 of 40 routed requests (10.0%) were answered by a fallback model instead of the tier's primary model. By expected tier: simple 0, standard 1, complex 3.

Models that answered the routed requests: `gemini/gemini-3.1-flash-lite` x11, `groq/openai/gpt-oss-120b` x4, `groq/openai/gpt-oss-20b` x17, `openai/qwen/qwen3.8-27b` x8.

Fallback tasks: m013 (`groq/openai/gpt-oss-120b`), c003 (`groq/openai/gpt-oss-120b`), c006 (`groq/openai/gpt-oss-120b`), c010 (`groq/openai/gpt-oss-120b`).

Cost savings excluding fallback tasks (36 tasks): **23.7%** vs 37.8% overall. A gap between the two means the headline number is influenced by fallback models, which can be cheaper than the baseline model.

## Failure analysis

2 task(s) scored lower when routed; the 2 largest gaps:

- **c006** expected `complex`, routed to `complex` on `groq/openai/gpt-oss-120b`: routed 4.33 vs baseline 5.00. Judge (A=baseline, B=routed): Both answers are accurate and on‑topic, but B is truncated and thus misses part of the required content.
- **m003** expected `standard`, routed to `standard` on `gemini/gemini-3.1-flash-lite`: routed 4.67 vs baseline 5.00. Judge (A=routed, B=baseline): B gives a concise, fully correct docstring exactly as requested, while A adds extra optional formats and commentary beyond the task.
