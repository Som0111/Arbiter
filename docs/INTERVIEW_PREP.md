# Arbiter — Interview Prep

Numbers below come from the 40-task eval in `reports/EVAL_REPORT.md` (classifier trained on a separate 60-example set, so routing accuracy is held-out).

## How does your router decide which model to use?

In order: (1) count tokens with tiktoken; (2) four confident rules, stopping at the first match: over 800 tokens is complex, code/debug/multi-step keywords (`def `, `bug`, `implement`, `step by step`, `prove`, a code fence) is complex (checked before the short-prompt rule, so a short "summarize this and fix the bug" isn't called simple), under 50 tokens with no code keywords is simple, simple-task keywords (`translate`, `what is`) is simple; (3) anything ambiguous goes to a logistic regression, trained on 60 examples that don't overlap the benchmark, over six features (token count, code/reasoning/simple-marker flags, sentence count, average word length); (4) the tenant's `tier_cap` is applied last as a hard ceiling, so a capped tenant asking a code question is downgraded and the log records `tier_capped=true`. The tier then maps to a model chain in `config/litellm_config.yaml`.

## What happens when the premium model is rate-limited?

The router catches `RateLimitError`, backs off 1s then 2s, and tries up to three times. Timeouts get a 1s wait. Any other error moves straight to the next model in the chain. If every model fails, the API returns `503 all_models_exhausted` and logs the failure. This is not hypothetical: during the eval Groq and Gemini free-tier limits triggered fallbacks, and 4 of 40 routed requests (10%) were answered by `gpt-oss-120b` instead of the primary model.

## Why not just use an LLM to classify tasks?

It would add a model call, roughly a second and some cost, to every request in order to save money on that request. Rules plus sklearn run in under 5 ms for free. The trade-off is accuracy: the classifier is 70.0% in cross-validation and routing accuracy on the held-out benchmark is 80.0%. I also caught and fixed a leak: an earlier 82.5% came from training the classifier on the benchmark prompts, so I moved training data to a separate file and added a check that fails if any benchmark prompt appears in it. I would revisit an LLM classifier only if misrouting costs more than the classification call.

## How did you measure cost savings?

The same 40 tasks ran twice: once with every task forced to the tier-3 model (the all-premium baseline), once through the gateway as `tenant_test`. Savings are `(baseline cost - routed cost) / baseline cost` = 37.8% ($0.0467 vs $0.0290), using list prices from one pricing block in `litellm_config.yaml` that is validated at startup. To check that cheap routing didn't just trade away quality, an LLM judge scored both answers 1-5 against an anchored rubric (blind, A/B order randomised, no credit for length); 97.5% of tasks scored within 0.5 of the baseline and 100% met their `quality_bar`. Caveats I would raise myself: only 40 tasks, one run, a judge from the same model family as the routed models that still scores everything about 4.9, and four requests answered by a fallback model that is cheaper than the baseline, so excluding them the saving is 23.7%.

## What breaks at 10x traffic?

(1) The rate limiter, budget tracker and cache are in-process, so they cannot be shared across instances; they need Redis. (2) The JSONL log is a single-file write path and `/stats` re-reads the whole file; it needs a queue plus a real database. (3) The free-tier quotas (Gemini ~15 requests/day per model, Groq ~8k tokens/minute) would be exhausted at once, so real traffic needs paid accounts. (4) The cache is capped at 500 entries with global LRU eviction, but it is still per-process. (5) The budget gate only checks that a tenant has money left and books real cost afterwards, so concurrent requests can overshoot a budget slightly; at scale I would reserve an estimated cost up front.

## Why is the cache keyed by tenant_id and tier?

Tenant isolation prevents one customer's cached answer from being served to another. Tier isolation keeps the log honest: a hit is attributed to the same tier the request was classified into, so the routing decision and the cost story never disagree. The cost of this is a lower hit rate (0% in the eval, since all 40 prompts were distinct). The cache is bounded at 500 entries: a hit refreshes recency, and when full the globally least recently used entry is evicted (counted in `/stats`). Hits also use a 0.92 cosine threshold on 768-dimension embeddings; in a quick check a paraphrase scored 0.985 and an unrelated question 0.714.

## What's the trade-off between rules and an ML classifier?

Rules are transparent, instant and easy to debug, but brittle: a standard task that contains the word "summarize" or a code snippet gets routed by keyword, which is why 5 of 15 standard tasks were misrouted (the confusion matrix shows 2 went to simple and 3 to complex). The sklearn model generalises beyond keywords but is trained on only 40 examples, so it is weak and hard to audit. Using rules for the clear cases and the model only for ambiguous ones keeps most requests on the fast path. The next improvements would be more labelled data from real traffic and routing on measured outcomes (for example, escalate when a cheap tier's answer is judged poor).
