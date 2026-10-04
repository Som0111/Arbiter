# Arbiter — Human Guide
**Plain-language explanation of the entire project**

---

## What is Arbiter?

Arbiter is an LLM gateway. Think of it as a smart switchboard that sits between your application and multiple AI models.

Every time someone sends a prompt, Arbiter asks: *"How hard is this task?"* Then it picks the cheapest model that can handle it well — not always the most expensive one.

**Why does this matter?** If you route everything to GPT-4 or Llama-70B, you pay top dollar for every single request, even when a much cheaper 8B model would give an equally good answer for a simple question.

---

## The Three Models (Tiers)

| Tier | Model | Used for | Cost |
|---|---|---|---|
| 1 — Simple | Groq Llama 3.1 8B | One-liners, translation, factual Q&A | Cheapest |
| 2 — Standard | Gemini 1.5 Flash | Summaries, short code, Q&A with context | Medium |
| 3 — Complex | Groq Llama 3.1 70B | Long code, debugging, deep reasoning | Expensive |

If the Tier 3 model is unavailable (rate-limited), it falls back to Gemini 1.5 Pro.

---

## Architecture — How a Request Flows

```
User sends: POST /generate
  {"prompt": "Write a Python function to sort a list", "tenant_id": "tenant_b"}

        │
        ▼
┌─────────────────┐
│   Auth Check    │  Is the X-API-Key header valid?
│                 │  No → 401 Unauthorized
└────────┬────────┘
         │
        ▼
┌─────────────────┐
│  Rate Limiter   │  Has this tenant sent too many requests in the last minute?
│                 │  Yes → 429 Too Many Requests
└────────┬────────┘
         │
        ▼
┌─────────────────┐
│  Budget Check   │  Has this tenant exceeded their daily spend limit?
│                 │  Yes → 429 Budget Exceeded
└────────┬────────┘
         │
        ▼
┌─────────────────┐
│  Classifier     │  How complex is this prompt?
│  (Rules first,  │  → Counts tokens
│   sklearn next) │  → Looks for code keywords ("def", "debug", "implement")
│                 │  → Assigns tier: simple / standard / complex
│                 │  Also applies tenant's tier cap (tenant_a can't use Tier 3)
└────────┬────────┘
         │
        ▼
┌─────────────────┐
│  Semantic Cache │  Has a very similar question been asked recently?
│                 │  → Converts prompt to a vector (Gemini embedding)
│                 │  → Checks if any cached vector is ≥ 92% similar
│                 │  → Cache hit: return saved answer instantly, cost = $0
│                 │  → Cache miss: continue
└────────┬────────┘
         │
        ▼
┌─────────────────┐
│  LLM Router     │  Calls the right model via LiteLLM
│  (LiteLLM)      │  → If model times out: retry with backoff (1s, 2s, 4s)
│                 │  → If all retries fail: try fallback model
│                 │  → If all models fail: return 503 error
└────────┬────────┘
         │
        ▼
┌─────────────────┐
│  Logger         │  Records everything to logs/requests.jsonl
│  + Langfuse     │  → model used, tokens, cost, latency, cache hit, fallback
└────────┬────────┘
         │
        ▼
Response returned to user:
  {
    "response": "def sort_list(lst): return sorted(lst)...",
    "model_used": "groq/llama-3.1-70b-versatile",
    "tier": "complex",
    "tokens_in": 18,
    "tokens_out": 45,
    "cost_usd": 0.000038,
    "latency_ms": 812,
    "cache_hit": false,
    "fallback_triggered": false
  }
```

---

## Each Component Explained

### 1. Auth (auth.py)
Every request must include a header: `X-API-Key: your-secret-key`. This is set in your `.env` file as `ARBITER_API_KEY`. Without it, the gateway rejects the request. This prevents random people on the internet from using your LLM quota.

### 2. Rate Limiter (budget.py)
Each tenant has a requests-per-minute limit (e.g., tenant_a = 20 RPM). The limiter uses a sliding window: it counts how many requests arrived in the last 60 seconds. If over the limit, it blocks the new request.

### 3. Budget Tracker (budget.py)
Each tenant has a daily dollar budget. The tracker accumulates actual LLM costs (from the response) and blocks requests once the budget is exhausted. The budget resets at midnight (UTC). Cost data comes from LiteLLM's built-in cost tracking.

### 4. Task Classifier (classifier.py)
This is the brain of the routing decision. It works in two steps:

**Step 1 — Rules (fast, free, runs first):**
- Count tokens in the prompt
- Look for keywords: `def`, `class`, `debug`, `implement` → complex
- Look for keywords: `summarize`, `what is`, `translate` → simple
- Short prompts (<50 tokens, no code) → simple
- Long prompts (>800 tokens) → complex
- Rules cover ~70% of cases

**Step 2 — ML Classifier (runs when rules are ambiguous):**
- A simple logistic regression model trained on the 40 benchmark tasks
- Features: token count, has code keywords, has reasoning keywords, sentence count
- This is not a neural network — it's a ~5KB scikit-learn model saved as a `.pkl` file
- Runs in <1ms locally

**Why not use an LLM to classify?** Because calling GPT-4 to decide whether to call GPT-4 is both slow and expensive. The rule+sklearn approach adds <5ms latency and costs nothing.

### 5. Semantic Cache (cache.py)
When someone asks a question, the cache:
1. Converts the prompt to a vector (768 numbers) using Gemini's embedding API
2. Compares it to every cached vector using cosine similarity (a measure of how similar two vectors are — 1.0 = identical, 0.0 = completely unrelated)
3. If similarity ≥ 0.92 → the questions are essentially the same → return the cached answer

The cache is keyed by `(tenant_id, tier)`. This is critical: a manager's answer must never be served to a regular employee (different tier), and tenant A's data must never leak to tenant B. Each bucket is completely isolated.

Cache entries expire after 1 hour (TTL).

### 6. LiteLLM Router (router.py)
LiteLLM is a Python library that gives every LLM provider the same API. Instead of writing separate code for Groq and Gemini, you write one call and LiteLLM handles the translation.

Retry logic:
- On `RateLimitError` → wait 1s, 2s, 4s (exponential backoff) and retry up to 3 times
- On `Timeout` → wait 1s, retry once
- If primary model exhausts retries → move to fallback model
- If all models fail → return HTTP 503

LiteLLM also tracks token usage and computes cost per request automatically.

### 7. Logger (logger.py)
Every request — successful or not — gets one line appended to `logs/requests.jsonl`. This is a text file where each line is a JSON object. Example line:
```json
{"request_id": "abc123", "tenant_id": "tenant_b", "tier": "complex", "model_used": "groq/llama-3.1-70b-versatile", "tokens_in": 18, "tokens_out": 45, "cost_usd": 0.000038, "latency_ms": 812, "cache_hit": false, "fallback_triggered": false, "success": true, "timestamp": "2026-10-04T11:00:00Z"}
```

Langfuse (optional) provides a web UI to explore individual traces — you can see exactly which classifier ruled triggered, how long the cache lookup took, and what the LLM returned.

### 8. Cost Dashboard (ui/dashboard.py)
A Streamlit web app that reads `requests.jsonl` and shows:
- Total spend per tenant
- Which tier gets the most traffic
- Latency by tier
- Cache savings

The most important chart: **"What would this have cost if everything went to Tier 3?"** vs actual cost. That number is the headline of the résumé bullet.

### 9. Eval Suite (scripts/run_eval.py)
Runs 40 pre-written tasks twice:
1. **Baseline**: everything forced to the Tier 3 model (no routing)
2. **Routed**: goes through Arbiter normally

Then compares:
- Cost: did routing save money?
- Quality: did cheaper models produce good-enough answers?
- Routing accuracy: did the classifier assign the right tier?

A Gemini Flash LLM judge scores each answer 1–5.

---

## Tenant System

Tenants represent different users or teams. Example config in `config/tenants.yaml`:

| Tenant | Daily budget | Tier cap | Rate limit |
|---|---|---|---|
| tenant_a | $0.10 | standard (max Tier 2) | 20 RPM |
| tenant_b | $1.00 | complex (all tiers) | 60 RPM |
| tenant_test | unlimited | complex | unlimited |

The tier cap is a hard ceiling: even if the classifier says "complex," tenant_a's requests never go above Tier 2.

---

## What "Free" Means Here

Everything runs within free-tier limits:

| Service | Free limit | How we stay within it |
|---|---|---|
| Groq | 14,400 req/day per model | Rate limiter + retry backoff |
| Gemini Flash | 1,500 req/day | Rate limiter per tenant |
| Gemini Pro | 50 req/day | Only used as Tier 3 fallback |
| Gemini embeddings | 1,500 req/day | Cache avoids redundant embeddings |
| Render | 750 hrs/month | One service, Docker |
| Langfuse | Unlimited (free tier) | Tracing only, no compute |
| GitHub Actions | 2,000 min/month | Tests run only on push |

**Important:** The cost we track and report is what it *would* cost at real production pricing. The free tier means we don't pay during development, but the numbers in EVAL_REPORT.md reflect actual model pricing.

---

## How to Run Locally

```bash
# 1. Clone and install
git clone https://github.com/Som0111/arbiter.git
cd arbiter
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Create .env from template
cp .env.template .env
# Fill in GROQ_API_KEY, GEMINI_API_KEY, ARBITER_API_KEY

# 3. Start the API
make run
# API is now at http://localhost:8000

# 4. Start the dashboard (separate terminal)
make dashboard
# Dashboard is at http://localhost:8501

# 5. Test it
curl -X POST http://localhost:8000/generate \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 2+2?", "tenant_id": "tenant_test"}'

# 6. Run tests
make test

# 7. Run eval
python scripts/run_eval.py
```

---

## API Reference

### GET /health
No auth required. Returns system status.
```json
{
  "status": "ok",
  "version": "0.1.0",
  "uptime_seconds": 3600,
  "tenant_count": 3
}
```

### POST /generate
Auth required (`X-API-Key` header).
```json
// Request
{
  "prompt": "Write a function to reverse a string",
  "tenant_id": "tenant_b",
  "task_type": null,       // optional hint
  "max_tokens": 512,
  "priority": "standard"
}

// Response
{
  "response": "def reverse_string(s): return s[::-1]",
  "model_used": "groq/llama-3.1-70b-versatile",
  "tier": "complex",
  "tokens_in": 11,
  "tokens_out": 12,
  "cost_usd": 0.000015,
  "latency_ms": 634,
  "cache_hit": false,
  "fallback_triggered": false,
  "request_id": "req_20261004_abc123"
}
```

### GET /stats
Auth required. Returns aggregated usage from all logged requests.

---

## Key Design Decisions

### Why rules before ML classifier?
Rules run in <1ms with zero cost. They handle ~70% of clearly simple or clearly complex prompts. The ML classifier only fires when the prompt is ambiguous. This keeps routing cost near zero on the common path — calling an LLM to decide which LLM to call would be wasteful.

### Why is the cache keyed by (tenant_id, tier)?
Without tenant isolation, a cached answer from one tenant could leak to another. Without tier isolation, a high-quality Tier 3 answer could be served as a response to a Tier 2 request from a different user — the cache hit would mask the fact that the routing decision changed. Each bucket is independent.

### Why LiteLLM instead of calling Groq/Gemini SDKs directly?
LiteLLM gives every provider the same interface. Adding a new provider (e.g., Mistral, Cohere) means adding a line to the config file, not rewriting API code. It also handles retries, fallbacks, and cost tracking in one place.

### Why not a neural classifier for tier detection?
The benchmark has 40 examples. A neural classifier trained on 40 examples would overfit badly. Logistic regression on handcrafted features generalises better at this scale. More importantly, the classifier runs locally — no API call, no latency, no cost.

---

## Honest Limitations

1. **Render free tier sleeps** after 15 minutes of inactivity. The first request after a sleep takes ~20–30 seconds to respond.
2. **Logs reset on Render restart** because free tier has ephemeral storage. Fix in production: mount a volume or point logs to an external store.
3. **Cache is in-memory** — it resets on server restart. In production, Redis (e.g. Upstash free tier) would persist it.
4. **Classifier trained on 40 examples** — unusual prompt styles may be misclassified.
5. **Gemini Pro 50 RPD limit** — the fallback model is rarely available at scale. In a real system, tier 3 would need a paid account.

---

## Common Interview Questions

**Q: How does the router decide which model to use?**
A: In order: (1) count tokens, (2) check for code/complexity keywords, (3) if ambiguous, run a logistic regression classifier trained on 40 labeled examples. Tenant tier caps are applied last as a hard override.

**Q: What happens when the premium model is rate-limited?**
A: The router catches `RateLimitError`, backs off exponentially (1s, 2s, 4s), retries up to 3 times. If all retries fail, it falls through to the next model in the fallback chain. If all models are exhausted, the API returns 503.

**Q: Why not just use an LLM to classify tasks?**
A: That would mean calling a model to decide which model to call — adding latency and cost to every request. Rules + sklearn add <5ms and cost $0.

**Q: How did you measure cost savings?**
A: Ran 40 benchmark tasks twice: once forcing all requests to Tier 3 (baseline), once through the full routing system. Computed `(baseline_cost - routed_cost) / baseline_cost`. The routed system also scored equivalent quality (LLM-judge ≥ baseline score −0.5) to confirm savings didn't come at a quality penalty.

**Q: What breaks at 10x traffic?**
A: (1) In-memory rate limiter and cache don't survive horizontal scaling — need Redis. (2) JSONL log becomes a write bottleneck — need async logging or a proper DB. (3) Gemini Pro's 50 RPD limit would hit immediately. (4) Render free tier can't handle concurrent load.

**Q: Why is the semantic cache keyed by both tenant_id and tier?**
A: Tenant isolation prevents data leakage between customers. Tier isolation ensures we don't serve a Tier-3 quality answer to a Tier-1 slot without recording the correct model + cost — the routing decision and the cache hit must tell the same story.
