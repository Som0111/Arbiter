# Arbiter - Eval Report

Compared 40 of 40 benchmark tasks (40 judged). Baseline = every task on `openai/qwen/qwen3.8-27b`. Costs are estimated at list prices.

## Summary

| Metric | Value |
|---|---|
| Routing accuracy | 82.5% |
| Cost savings vs all-premium | 40.2% |
| Quality retention (routed >= baseline - 0.5) | 90.0% |
| Routed answers meeting task quality_bar | 100.0% |
| Baseline p50 / p95 latency | 349ms / 12806ms |
| Routed p50 / p95 latency | 1901ms / 8424ms |
| Fallback rate | 12.5% |
| Cache hit rate (eval run) | 0.0% |
| Total cost: baseline / routed | $0.04667 / $0.02792 |

## Per-tier breakdown (by expected tier)

| Tier | Tasks | Routing acc. | Baseline cost | Routed cost | Savings | Baseline score | Routed score |
|---|---|---|---|---|---|---|---|
| simple | 15 | 100% | $0.00305 | $0.00050 | 83.7% | 4.64 | 4.78 |
| standard | 15 | 67% | $0.00719 | $0.00418 | 41.9% | 4.82 | 4.84 |
| complex | 10 | 80% | $0.03643 | $0.02325 | 36.2% | 4.80 | 4.83 |

## Failure analysis

8 task(s) scored lower when routed; the 5 largest gaps:

- **s003** expected `simple`, routed to `simple` on `groq/openai/gpt-oss-20b`: routed 4.00 vs baseline 5.00. Judge (A=baseline, B=routed): Response A provides a full, informative answer while B gives only the minimal correct answer.
- **m003** expected `standard`, routed to `standard` on `gemini/gemini-3.1-flash-lite`: routed 4.00 vs baseline 5.00. Judge (A=routed, B=baseline): Response B provides a concise, accurate docstring directly matching the task, while A adds extra optional styles and notes that are less focused.
- **c006** expected `complex`, routed to `complex` on `groq/openai/gpt-oss-120b`: routed 4.00 vs baseline 5.00. Judge (A=baseline, B=routed): Response A fully meets the word count and covers all required points, while B is cut off and thus less complete.
- **s010** expected `simple`, routed to `simple` on `groq/openai/gpt-oss-20b`: routed 4.33 vs baseline 5.00. Judge (A=routed, B=baseline): Both give the correct temperature, but B provides a full step‑by‑step calculation while A only states the result.
- **s011** expected `simple`, routed to `simple` on `groq/openai/gpt-oss-20b`: routed 4.67 vs baseline 5.00. Judge (A=routed, B=baseline): Both answers are correct and relevant, but B provides additional contextual information, making it more complete.
