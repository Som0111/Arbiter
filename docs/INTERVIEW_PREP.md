# Arbiter — Interview Prep

Numbers below come from the 40-task eval in `reports/EVAL_REPORT.md`.

## How does your router decide which model to use?

In order: (1) count tokens with tiktoken; (2) four confident rules, stopping at the first match: under 50 tokens with no code keywords is simple, over 800 tokens is complex, code/debug keywords (`def `, `bug`, `implement`, a code fence) is complex, simple-task keywords (`translate`, `what is`) is simple; (3) anything ambiguous goes to a logistic regression over six features (token count, code/reasoning/simple-marker flags, sentence count, average word length); (4) the tenant's `tier_cap` is applied last as a hard ceiling, so a capped tenant asking a code question is downgraded and the log records `tier_capped=true`. The tier then maps to a model chain in `config/litellm_config.yaml`.

## What happens when the premium model is rate-limited?

The router catches `RateLimitError`, backs off 1s then 2s, and tries up to three times. Timeouts get a 1s wait. Any other error moves straight to the next model in the chain. If every model fails, the API returns `503 all_models_exhausted` and logs the failure. This is not hypothetical: during the eval, Gemini's free tier returned daily-quota 429s and the complex tasks c006-c010 fell back from Qwen to `gpt-oss-120b` (12.5% fallback rate overall).

## Why not just use an LLM to classify tasks?

It would add a model call, roughly a second and some cost, to every request in order to save money on that request. Rules plus sklearn run in under 5 ms for free. The trade-off is accuracy: the classifier is 72.5% in cross-validation, and routing accuracy on the benchmark is 82.5%. I would revisit an LLM classifier only if misrouting costs more than the classification call.

## How did you measure cost savings?

The same 40 tasks ran twice: once with every task forced to the tier-3 model (the all-premium baseline), once through the gateway as `tenant_test`. Savings are `(baseline cost - routed cost) / baseline cost` = 40.2% ($0.0467 vs $0.0279), using list prices from LiteLLM's table with a manual fallback map. To check that cheap routing didn't just trade away quality, an LLM judge scored both answers 1-5 (blind, with A/B order randomised per task); 90% of tasks scored within 0.5 of the baseline and 100% met their `quality_bar`. Caveats I would raise myself: only 40 tasks, one run, a judge from the same model family as the routed models, and five complex tasks where the baseline itself was served by the fallback.

## What breaks at 10x traffic?

(1) The rate limiter, budget tracker and cache are in-process, so they cannot be shared across instances; they need Redis. (2) The JSONL log is a single-file write path and `/stats` re-reads the whole file; it needs a queue plus a real database. (3) The free-tier quotas (Gemini ~15 requests/day per model, Groq ~8k tokens/minute) would be exhausted at once, so real traffic needs paid accounts. (4) The cache has no size cap. (5) The budget gate only checks that a tenant has money left and books real cost afterwards, so concurrent requests can overshoot a budget slightly; at scale I would reserve an estimated cost up front.

## Why is the cache keyed by tenant_id and tier?

Tenant isolation prevents one customer's cached answer from being served to another. Tier isolation keeps the log honest: a hit is attributed to the same tier the request was classified into, so the routing decision and the cost story never disagree. The cost of this is a lower hit rate (0% in the eval, since all 40 prompts were distinct). Hits also use a 0.92 cosine threshold on 768-dimension embeddings; in a quick check a paraphrase scored 0.985 and an unrelated question 0.714.

## What's the trade-off between rules and an ML classifier?

Rules are transparent, instant and easy to debug, but brittle: a standard task that contains the word "summarize" or a code snippet gets routed by keyword, which is why 5 of 15 standard tasks were misrouted. The sklearn model generalises beyond keywords but is trained on only 40 examples, so it is weak and hard to audit. Using rules for the clear cases and the model only for ambiguous ones keeps most requests on the fast path. The next improvements would be more labelled data from real traffic and routing on measured outcomes (for example, escalate when a cheap tier's answer is judged poor).
