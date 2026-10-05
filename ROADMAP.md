# Arbiter — ROADMAP
**LLM Gateway with Intelligent Model Routing**

> Claude Code owns all phases. Every `🧑 HUMAN` block is a step only you can do — read it, complete it, then tell Claude Code to continue.

> **Update (Phases 6-8):** the model names below were retired or quota-limited by the time of the build. Current tiers: `groq/openai/gpt-oss-20b` / `gemini/gemini-3.1-flash-lite` / `openai/qwen/qwen3.8-27b` (Groq), fallback `groq/openai/gpt-oss-120b`, embeddings `gemini-embedding-001`. The classifier now trains on `data/classifier_train.json` (60 examples, separate from the benchmark, so the eval is held-out), and model pricing lives in `config/litellm_config.yaml`. See README.md and that file for the source of truth.

---

## Project Overview

Arbiter is a production-style LLM gateway that routes every request to the cheapest model that can handle it, proving: cost-aware routing, multi-provider fallback, per-tenant budgets, and request-level telemetry.

**Three tiers:**
- **Tier 1 — Simple**: `groq/llama-3.1-8b-instant` (~$0.05/M tokens)
- **Tier 2 — Standard**: `gemini/gemini-1.5-flash` (~$0.075/M tokens)
- **Tier 3 — Complex**: `groq/llama-3.1-70b-versatile` (~$0.59/M tokens)

**The number that goes on the résumé:** "routing cut estimated inference spend by ~55–60% while holding task quality above threshold."

---

## Stack (all free)

| Layer | Tool | Free limit |
|---|---|---|
| API | FastAPI | unlimited |
| Provider abstraction | LiteLLM | unlimited |
| Tier 1 model | Groq `llama-3.1-8b-instant` | 30 RPM, 14,400 RPD |
| Tier 2 model | Gemini `gemini-1.5-flash` | 15 RPM, 1,500 RPD |
| Tier 3 model | Groq `llama-3.1-70b-versatile` | 30 RPM, 14,400 RPD |
| Tier 3 fallback | Gemini `gemini-1.5-pro` | 2 RPM, 50 RPD |
| Embeddings (cache) | Gemini `text-embedding-004` | 100 RPM, 1,500 RPD |
| Task classifier | scikit-learn (local) | unlimited |
| Tracing | Langfuse (free tier) | unlimited |
| Dashboard | Streamlit | unlimited |
| Deployment | Render free tier | 750 hrs/month |
| CI | GitHub Actions | 2,000 min/month |

---

## Directory Structure

```
arbiter/
├── src/
│   └── arbiter/
│       ├── __init__.py
│       ├── api.py           # FastAPI app
│       ├── auth.py          # API key middleware
│       ├── classifier.py    # Tier classifier (rules + sklearn)
│       ├── router.py        # LiteLLM calls, retry, fallback
│       ├── cache.py         # Semantic cache (Gemini embeddings)
│       ├── budget.py        # Per-tenant rate limit + budget
│       ├── logger.py        # JSONL logger + Langfuse
│       └── config.py        # Load settings
├── tests/
│   ├── conftest.py
│   ├── test_pipeline.py
│   ├── test_classifier.py
│   ├── test_router.py
│   ├── test_cache.py
│   └── test_telemetry.py
├── data/
│   └── benchmark/
│       ├── benchmark.json
│       └── classifier_train.json
├── scripts/
│   ├── validate_benchmark.py
│   ├── train_classifier.py
│   └── run_eval.py
├── ui/
│   └── dashboard.py
├── reports/
│   └── .gitkeep
├── docs/
│   ├── HUMAN_GUIDE.md
│   └── INTERVIEW_PREP.md
├── config/
│   └── tenants.yaml
├── logs/                    # gitignored
├── .github/
│   └── workflows/ci.yml
├── Dockerfile
├── render.yaml
├── pyproject.toml
├── .env.template
├── .gitignore
├── Makefile
├── ROADMAP.md
└── README.md
```

---

## Auto-Correction Protocol

> Claude Code follows this after **every** checkpoint.

```
CHECKPOINT FAILED:
  1. Read the full error output — identify the root cause category:
     - ImportError / ModuleNotFoundError  → fix pyproject.toml, run: pip install -e ".[dev]"
     - FileNotFoundError                 → create the missing file/directory first
     - AssertionError in test            → read the test, fix the implementation
     - RateLimitError (LLM provider)     → add retry/backoff or mock the call in tests
     - KeyError / AttributeError         → trace back to the data structure, fix schema
     - Pydantic ValidationError          → fix model field types
  2. Apply the fix.
  3. Re-run the checkpoint command.
  4. Repeat up to 3 times.
  5. If still failing after 3 attempts:
     STOP. Report to human:
       - Phase + subphase name
       - Full error message (last attempt)
       - What was tried in each of the 3 attempts
```

This protocol applies to all phases. Never skip a checkpoint.

---

## Phase 0 — Foundation

**Goal:** Working repo, CI passing, all accounts set up.

### 0.1 Repo skeleton
- Create all directories from the structure above.
- Create empty `__init__.py` in `src/arbiter/`.
- Create `logs/.gitkeep` and `reports/.gitkeep`.

### 0.2 pyproject.toml
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "arbiter"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.111",
    "uvicorn[standard]>=0.30",
    "litellm>=1.40",
    "scikit-learn>=1.5",
    "tiktoken>=0.7",
    "numpy>=1.26",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "httpx>=0.27",
    "streamlit>=1.36",
    "plotly>=5.22",
    "pandas>=2.2",
    "python-dotenv>=1.0",
    "pyyaml>=6.0",
    "langfuse>=2.30",
    "google-generativeai>=0.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2",
    "pytest-asyncio>=0.23",
    "httpx>=0.27",
    "ruff>=0.4",
    "pytest-cov>=5.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

### 0.3 .env.template
```
# Required
GROQ_API_KEY=
GEMINI_API_KEY=
ARBITER_API_KEY=any-long-random-string-for-gateway-auth

# Optional (tracing)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 0.4 .gitignore
Include: `.env`, `logs/`, `__pycache__/`, `.pytest_cache/`, `dist/`, `*.egg-info/`, `model_cache/`, `.venv/`

### 0.5 Makefile
```makefile
install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src/ tests/

run:
	uvicorn arbiter.api:app --app-dir src --reload --port 8000

dashboard:
	streamlit run ui/dashboard.py --server.port 8501
```

### 0.6 GitHub Actions CI
File: `.github/workflows/ci.yml`
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -e ".[dev]"
      - run: pytest -q
      - run: ruff check src/ tests/
```

### 0.7 config/tenants.yaml
```yaml
tenants:
  tenant_a:
    daily_budget_usd: 0.10
    tier_cap: standard        # never goes to Tier 3
    rate_limit_rpm: 20

  tenant_b:
    daily_budget_usd: 1.00
    tier_cap: complex         # full access
    rate_limit_rpm: 60

  tenant_test:
    daily_budget_usd: 999.0
    tier_cap: complex
    rate_limit_rpm: 999
```

### Checkpoint 0

```bash
pip install -e ".[dev]"
pytest -q           # should show: no tests ran, no errors
ruff check src/     # should show: no issues
```

**Expected:** All pass with 0 errors.

---

### 🧑 HUMAN — Phase 0 Setup

Before Phase 1, complete all of the following:

1. **GitHub repo**: Create a new public repo named `arbiter`. Push the skeleton.
2. **Groq API key**: Sign up at [console.groq.com](https://console.groq.com) → API Keys → Create Key.
3. **Gemini API key**: Go to [aistudio.google.com](https://aistudio.google.com) → Get API key. (Same one used in SebiSage is fine.)
4. **Langfuse** (optional but recommended): Sign up at [cloud.langfuse.com](https://cloud.langfuse.com) → New Project → copy Public Key + Secret Key.
5. Fill in `.env` with all keys.
6. Confirm CI passes on GitHub (green checkmark on first push).

Tell Claude Code: **"Phase 0 done, proceed to Phase 1."**

---

## Phase 1 — Benchmark Dataset

**Goal:** 40 labeled tasks that become the source of truth for classifier training and eval. Write these before writing any model code.

### 1.1 Benchmark schema
File: `data/benchmark/benchmark.json`
```json
[
  {
    "id": "s001",
    "tier": "simple",
    "prompt": "Summarise in one sentence: The quick brown fox jumps over the lazy dog.",
    "expected_keywords": ["fox", "dog"],
    "quality_bar": 3,
    "max_cost_usd": 0.0002,
    "max_latency_ms": 3000
  }
]
```

Field definitions:
- `tier`: `simple` | `standard` | `complex`
- `quality_bar`: minimum acceptable LLM-judge score (1–5)
- `max_cost_usd`: per-request cost ceiling for routing to be valid
- `max_latency_ms`: p95 latency ceiling

### 1.2 Claude Code writes 40 starter tasks
Distribution: **15 simple + 15 standard + 10 complex**

**Simple tasks (15):** One-line summarisation, single-word translation, "what is X" factual lookup, format conversion (date, unit), yes/no classification on short text.

**Standard tasks (15):** 3–5 paragraph summarisation, multi-sentence Q&A over provided context, function docstring generation (<20 lines), sentiment analysis with explanation, bullet-point extraction from a paragraph.

**Complex tasks (10):** Full function implementation (sorting, parsing, API call), multi-step reasoning (word problem with steps), compare-and-contrast two technical concepts (>300-word answer expected), debug a buggy code snippet with explanation, write a regex with test cases.

### 1.3 Validation script
File: `scripts/validate_benchmark.py`
- Load `benchmark.json`.
- Assert: all required fields present, tier values valid, quality_bar in 1–5, IDs unique.
- Print count per tier.
- Exit 0 on success, exit 1 on any failure.

### 1.4 Classifier training labels
File: `data/benchmark/classifier_train.json`
Auto-generated from `benchmark.json` — extract `prompt` + `tier` pairs for sklearn.

### Checkpoint 1

```bash
python scripts/validate_benchmark.py
```

**Expected:** "Validation passed. simple=15 standard=15 complex=10 total=40"

---

### 🧑 HUMAN — Phase 1 Review

1. Open `data/benchmark/benchmark.json`.
2. Read through all 40 tasks.
3. Adjust any `quality_bar` values that seem off.
4. Replace any tasks that feel unrealistic or too easy.
5. Confirm the split (15/15/10) looks right.

Tell Claude Code: **"Benchmark looks good, proceed to Phase 2."**

---

## Phase 2 — Core Request Pipeline

**Goal:** Working FastAPI app with auth, rate limiting, budget checks, and stub response.

### 2.1 config.py
```python
from pydantic_settings import BaseSettings
import yaml

class Settings(BaseSettings):
    groq_api_key: str
    gemini_api_key: str
    arbiter_api_key: str
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    class Config:
        env_file = ".env"

def load_tenants() -> dict:
    with open("config/tenants.yaml") as f:
        return yaml.safe_load(f)["tenants"]
```

### 2.2 auth.py
- FastAPI dependency: reads `X-API-Key` header.
- Compares against `settings.arbiter_api_key`.
- Returns 401 if missing or wrong.

### 2.3 budget.py
```
BudgetTracker:
  - In-memory dict: {tenant_id: {date: cost_usd_today}}
  - check_and_reserve(tenant_id, estimated_cost) → bool (False = over budget)
  - record_actual(tenant_id, actual_cost) — called after LLM response
  - get_remaining(tenant_id) → float
  - Reads daily limit from tenants.yaml
  - Date key resets tracker automatically (no cron needed)
```

Rate limiter:
```
RateLimiter:
  - In-memory dict: {tenant_id: deque of timestamps}
  - check(tenant_id) → bool (False = over limit)
  - Sliding window: count requests in last 60 seconds, compare to rpm limit
```

### 2.4 api.py — request/response models
```python
class GenerateRequest(BaseModel):
    prompt: str
    tenant_id: str
    task_type: str | None = None   # optional hint: "summarize" | "code" | "qa"
    max_tokens: int = 512
    priority: str = "standard"     # "low" | "standard" | "high"

class GenerateResponse(BaseModel):
    response: str
    model_used: str
    tier: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    cache_hit: bool
    fallback_triggered: bool
    request_id: str
```

### 2.5 api.py — endpoints
- `GET /health` — no auth — returns version, uptime, tenant count
- `POST /generate` — auth required — stub: return `{"response": "stub", ...}`
- `GET /stats` — auth required — reads from `logs/requests.jsonl`, returns aggregates

### 2.6 logger.py
```python
@dataclass
class RequestLog:
    request_id: str
    tenant_id: str
    tier: str
    model_used: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    cache_hit: bool
    fallback_triggered: bool
    success: bool
    error: str | None
    timestamp: str   # ISO 8601

def write_log(entry: RequestLog, path: str = "logs/requests.jsonl"):
    os.makedirs("logs", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(asdict(entry)) + "\n")
```

### Checkpoint 2

```bash
pytest tests/test_pipeline.py -v
```

Tests must cover:
- Valid API key → 200; wrong key → 401
- Rate limit: 20 requests in 1 second for tenant_a → last requests blocked
- Budget: over-budget tenant → 429 with `{"error": "budget_exceeded"}`
- `/health` returns 200 without API key
- `/generate` stub returns correct schema

**Auto-correction note:** If `pydantic_settings` import fails → check pyproject.toml deps, re-run `pip install -e ".[dev]"`.

---

## Phase 3 — Task Classifier

**Goal:** Classify any prompt into `simple` / `standard` / `complex` using rules first, sklearn second.

### 3.1 Token counter (classifier.py)
```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))
```

### 3.2 Rule engine
Rules run in this order (stop at first match):

| Condition | Tier |
|---|---|
| token_count < 50 AND no code keywords | simple |
| token_count > 800 | complex |
| has any of: `def `, ` class `, `````, `bug`, `debug`, `implement`, `write a function` | complex |
| has any of: `summarize`, `translate`, `what is`, `define`, `yes or no` | simple |
| token_count 50–200 | standard |
| token_count 200–800 | standard |
| fallback | standard |

Code keywords: `def `, `class `, `import `, ` for `, `while `, `return `, `async def`, ````python`

### 3.3 Train sklearn classifier
File: `scripts/train_classifier.py`

Features per prompt:
- `token_count`
- `has_code` (bool → int)
- `has_reasoning` (bool → int): keywords = `explain`, `compare`, `why`, `analyze`, `difference between`
- `has_simple_markers` (bool → int): keywords = `summarize`, `translate`, `what is`, `define`
- `sentence_count`
- `avg_word_length`

Classifier: `LogisticRegression(max_iter=1000, C=1.0)`
- Train on `classifier_train.json` (40 examples)
- 5-fold cross-validation, print accuracy
- Save model to `data/classifier.pkl`

### 3.4 Classifier pipeline in classifier.py
```
classify(prompt: str) -> str:
  1. Run rule engine → if confident match → return tier
  2. Load classifier.pkl → predict → return tier
```

"Confident" = rule matched in top 4 rules (strong signal). Ambiguous inputs (token count 50–200, no clear keywords) fall through to sklearn.

### 3.5 Tier cap enforcement
After classification, apply tenant's `tier_cap`:
- If classified as `complex` but tenant cap = `standard` → downgrade to `standard`
- Log a `tier_capped=true` flag

### Checkpoint 3

```bash
python scripts/train_classifier.py    # prints CV accuracy
pytest tests/test_classifier.py -v
```

Tests must cover:
- Short factual prompt → `simple`
- Long code prompt → `complex`
- Medium Q&A prompt → `standard`
- tenant_a (cap=standard) asking code question → `standard` (capped)
- CV accuracy ≥ 0.70

**Auto-correction note:** If accuracy < 0.70 → add more training examples programmatically (augment the 40 with 20 paraphrased versions); retrain.

---

## Phase 4 — LiteLLM Router

**Goal:** Full LLM call pipeline with retry, backoff, and cross-provider fallback.

### 4.1 LiteLLM config
File: `config/litellm_config.yaml`
```yaml
model_list:
  - model_name: tier1
    litellm_params:
      model: groq/llama-3.1-8b-instant
      api_key: "os.environ/GROQ_API_KEY"

  - model_name: tier2
    litellm_params:
      model: gemini/gemini-1.5-flash
      api_key: "os.environ/GEMINI_API_KEY"

  - model_name: tier3
    litellm_params:
      model: groq/llama-3.1-70b-versatile
      api_key: "os.environ/GROQ_API_KEY"

  - model_name: tier3_fallback
    litellm_params:
      model: gemini/gemini-1.5-pro
      api_key: "os.environ/GEMINI_API_KEY"
```

### 4.2 Tier → model mapping
```python
TIER_MODEL_MAP = {
    "simple":   ["tier1"],
    "standard": ["tier2"],
    "complex":  ["tier3", "tier3_fallback"],
}
```

### 4.3 router.py — call with retry + fallback
```
call_llm(tier: str, prompt: str, max_tokens: int) -> LLMResult:
  models = TIER_MODEL_MAP[tier]
  for model in models:
    for attempt in range(3):
      try:
        response = litellm.completion(model=model, messages=[...], timeout=15)
        return LLMResult(
          text=response.choices[0].message.content,
          model=model,
          tokens_in=response.usage.prompt_tokens,
          tokens_out=response.usage.completion_tokens,
          cost_usd=litellm.completion_cost(response),
          fallback_triggered=(model != TIER_MODEL_MAP[tier][0])
        )
      except RateLimitError:
        sleep(2 ** attempt)   # 1s, 2s, 4s
      except Timeout:
        sleep(1)
      except Exception as e:
        break   # move to next model
  raise GatewayError("all models exhausted")
```

### 4.4 Cost estimation
LiteLLM tracks cost via `litellm.completion_cost(response)`. If it returns 0 (model not in cost DB), use a manual fallback map:
```python
MANUAL_COST_PER_TOKEN = {
    "groq/llama-3.1-8b-instant":     {"in": 0.05e-6, "out": 0.08e-6},
    "gemini/gemini-1.5-flash":        {"in": 0.075e-6, "out": 0.30e-6},
    "groq/llama-3.1-70b-versatile":  {"in": 0.59e-6, "out": 0.79e-6},
    "gemini/gemini-1.5-pro":          {"in": 3.50e-6, "out": 10.50e-6},
}
```

### 4.5 Wire router into /generate
Replace stub in `api.py`:
1. Rate limit check → 429 if blocked
2. Budget check → 429 if blocked
3. Classify prompt → tier
4. Apply tier cap
5. Check cache (Phase 5 — use stub `cache.get()` for now, always returns None)
6. Call LLM via router
7. Record cost in budget tracker
8. Write to JSONL logger
9. Return `GenerateResponse`

### Checkpoint 4

```bash
pytest tests/test_router.py -v
```

Tests must cover (all with **mocked** LiteLLM calls — no real API calls in unit tests):
- Simple prompt → calls tier1 model
- Complex prompt → calls tier3 first
- tier3 RateLimitError → retries 3x → falls to tier3_fallback
- All models timeout → raises GatewayError → API returns 503
- Cost calculated correctly for each model
- fallback_triggered=True when fallback model used

**Auto-correction note:** Mock LiteLLM using `unittest.mock.patch("litellm.completion")`. If import path differs, read litellm source to find correct patch target.

---

## Phase 5 — Semantic Cache

**Goal:** Cache similar queries so identical-ish requests skip the LLM entirely.

### 5.1 Embedding via Gemini API
```python
import google.generativeai as genai

def embed(text: str) -> list[float]:
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_query"
    )
    return result["embedding"]
```

### 5.2 SemanticCache class
```python
@dataclass
class CacheEntry:
    embedding: list[float]
    response: str
    model: str
    cost_usd: float
    created_at: float   # time.time()

class SemanticCache:
    def __init__(self, ttl_seconds=3600, similarity_threshold=0.92):
        self._store: dict[str, list[CacheEntry]] = {}  # keyed by (tenant_id, tier)
        self.ttl = ttl_seconds
        self.threshold = similarity_threshold

    def get(self, tenant_id, tier, query_embedding) -> CacheEntry | None:
        # 1. Purge expired entries
        # 2. Look up (tenant_id, tier) bucket
        # 3. Compute cosine similarity with each entry
        # 4. Return best match if >= threshold, else None

    def put(self, tenant_id, tier, query_embedding, entry: CacheEntry):
        # Append to (tenant_id, tier) bucket
```

### 5.3 Cosine similarity (no external library)
```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

### 5.4 Wire cache into /generate
After tier classification, before LLM call:
1. Get query embedding (`embed(prompt)`)
2. `hit = cache.get(tenant_id, tier, embedding)`
3. If hit → return cached response, set `cache_hit=True`, log with `cost_usd=0`
4. If miss → call LLM → `cache.put(...)` with result

Note: cache is keyed by `(tenant_id, tier)` — a manager's cached answer is never served to a different tier or tenant. This is the key correctness property.

### 5.5 Cache stats in /stats
Add to stats endpoint:
- `cache_hit_rate`: hits / total requests
- `cache_size`: number of entries
- `estimated_saved_usd`: sum of cost_usd of all cache hits

### Checkpoint 5

```bash
pytest tests/test_cache.py -v
```

Tests must cover (mock `embed()` to return fixed vectors):
- Identical query → cache hit
- Different tenant, same query → cache miss (separate bucket)
- Different tier, same query → cache miss
- Entry older than TTL → cache miss (purged)
- Cosine similarity < threshold → cache miss

**Auto-correction note:** If `google.generativeai` import fails → `pip install google-generativeai`. If embed returns wrong shape → check model name is `models/text-embedding-004` not `text-embedding-004`.

---

## Phase 6 — Telemetry

**Goal:** Every request produces a structured log line. Stats endpoint aggregates them. Langfuse optional trace.

### 6.1 logger.py — full log entry
Ensure `RequestLog` dataclass includes all fields from Phase 2 plus:
- `tier_capped: bool` (was tier downgraded due to tenant cap?)
- `cache_hit: bool`
- `fallback_triggered: bool`
- `classify_source: str` — `"rules"` or `"sklearn"`

### 6.2 Langfuse integration
```python
def init_langfuse():
    if settings.langfuse_public_key:
        from langfuse import Langfuse
        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host
        )
    return None   # tracing is optional, never blocks requests
```

Trace structure per request:
- Span: `classify` (input=prompt, output=tier, source=classify_source)
- Span: `cache_lookup` (hit=bool)
- Span: `llm_call` (model=model_used, tokens_in, tokens_out, cost_usd, latency_ms)

### 6.3 /stats endpoint
Read all lines from `logs/requests.jsonl`, compute:
```json
{
  "total_requests": 0,
  "total_cost_usd": 0.0,
  "cache_hit_rate": 0.0,
  "estimated_saved_usd": 0.0,
  "fallback_rate": 0.0,
  "tier_distribution": {"simple": 0, "standard": 0, "complex": 0},
  "cost_by_tenant": {},
  "cost_by_tier": {},
  "latency_p50_ms": 0.0,
  "latency_p95_ms": 0.0,
  "requests_last_24h": 0
}
```

Use `numpy.percentile` for p50/p95.

### Checkpoint 6

```bash
pytest tests/test_telemetry.py -v
```

Tests must cover:
- Full request flow writes exactly one log line
- Log line has all required fields
- /stats returns correct aggregates for a known set of log lines
- Langfuse init skipped gracefully when keys not set

---

## Phase 7 — Cost Dashboard

**Goal:** Streamlit app that visualises the JSONL log. Shows cost, routing distribution, latency, fallback rate.

### 7.1 ui/dashboard.py

Sections (use `st.columns` for layout):

**Top row — KPI cards (4 cards):**
- Total requests
- Total estimated cost ($)
- Cache hit rate (%)
- Avg cost per request ($)

**Chart 1 — Cost per tier (bar chart)**
- X: tier name, Y: cumulative cost_usd

**Chart 2 — Tier distribution (pie chart)**
- simple vs standard vs complex request count

**Chart 3 — Requests over time (line chart)**
- X: hour of day, Y: request count, grouped by tier

**Chart 4 — Latency distribution (bar chart)**
- p50 and p95 latency for each tier

**Chart 5 — Cost savings table**
- Row per tier: actual cost | all-premium cost | savings %
- "All-premium cost" = recalculate assuming every request used Tier 3 model

**Bottom — Raw log table**
- Last 50 requests, sortable

### 7.2 Load and refresh
- `@st.cache_data(ttl=30)` to reload JSONL every 30 seconds
- Handle empty log file gracefully (show "No data yet")

### Checkpoint 7

```bash
# Generate 20 fake log entries first
python scripts/generate_sample_logs.py
streamlit run ui/dashboard.py --server.headless true &
sleep 5
curl http://localhost:8501/_stcore/health   # should return {"status": "ok"}
pkill -f streamlit
```

`scripts/generate_sample_logs.py`: write 20 random `RequestLog` entries to `logs/requests.jsonl` for dashboard testing.

**Auto-correction note:** If streamlit health check fails → check if port 8501 is occupied; try port 8502.

---

## Phase 8 — Eval Suite

**Goal:** Run 40 benchmark tasks against (a) all-premium baseline and (b) routed system. Compute the résumé numbers.

### 8.1 scripts/run_eval.py

```
STEP 1 — All-premium baseline
  For each task in benchmark.json:
    Call LLM directly using Tier 3 model (groq/llama-3.1-70b-versatile)
    Record: response, tokens_in, tokens_out, cost_usd, latency_ms
    Save to reports/baseline_results.json

STEP 2 — Routed system
  For each task in benchmark.json:
    POST to /generate (use tenant_test, no budget cap)
    Record: response, model_used, tier_assigned, cost_usd, latency_ms, cache_hit
    Save to reports/routed_results.json

STEP 3 — Quality scoring
  For each task:
    Send prompt + baseline_response + routed_response to Gemini Flash
    Ask: "Score each response 1-5 for correctness, relevance, and completeness."
    Save scores to reports/quality_scores.json

STEP 4 — Compute metrics
  routing_accuracy = % tasks where tier_assigned == expected_tier
  cost_savings_pct = (baseline_total_cost - routed_total_cost) / baseline_total_cost × 100
  quality_retention = % tasks where routed_score >= baseline_score - 0.5
  latency_p50/p95 for both systems
  fallback_rate = % tasks where fallback_triggered

STEP 5 — Write EVAL_REPORT.md
```

### 8.2 Rate limit guard
Add `time.sleep(2)` between each Gemini call in baseline (2 RPM limit for Pro). For Groq calls, sleep 0.5s (rate limit headroom).

### 8.3 reports/EVAL_REPORT.md template
```markdown
# Arbiter — Eval Report

## Summary
| Metric | Value |
|---|---|
| Routing accuracy | X% |
| Cost savings vs all-premium | X% |
| Quality retention (routed ≥ baseline −0.5) | X% |
| Baseline p50 / p95 latency | Xms / Xms |
| Routed p50 / p95 latency | Xms / Xms |
| Fallback rate | X% |
| Cache hit rate (eval run) | X% |

## Per-tier breakdown
[table]

## Failure analysis
[5 tasks where routed score < baseline score, and why]
```

### Checkpoint 8

```bash
python scripts/run_eval.py
# Expect: reports/EVAL_REPORT.md created, no unhandled exceptions
```

**Rate limit handling:** if `RateLimitError` during eval → catch it, sleep 60s, retry once. If still fails → skip that task and note in report.

---

### 🧑 HUMAN — Phase 8 Review

1. Open `reports/EVAL_REPORT.md`.
2. Check cost savings % — should be 40–65%. If <20%, the routing rules may be too conservative (most tasks going to complex).
3. Check quality retention — should be ≥ 85%. If lower, raise `quality_bar` thresholds in benchmark.
4. Fill in the résumé bullet with the real numbers.

Tell Claude Code: **"Eval looks good, proceed to Phase 9."**

---

## Phase 9 — Render Deployment

**Goal:** Live public URL on Render free tier.

### 9.1 Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
COPY config/ config/
COPY ui/ ui/
COPY data/ data/

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "arbiter.api:app", "--app-dir", "src", \
     "--host", "0.0.0.0", "--port", "8000"]
```

Note: Dashboard is a separate Streamlit process — on Render free tier, deploy only the FastAPI API. Dashboard runs locally.

### 9.2 render.yaml
```yaml
services:
  - type: web
    name: arbiter
    runtime: docker
    plan: free
    healthCheckPath: /health
    envVars:
      - key: GROQ_API_KEY
        sync: false
      - key: GEMINI_API_KEY
        sync: false
      - key: ARBITER_API_KEY
        sync: false
      - key: LANGFUSE_PUBLIC_KEY
        sync: false
      - key: LANGFUSE_SECRET_KEY
        sync: false
```

### 9.3 Startup log persistence
Render free tier has ephemeral storage — `logs/requests.jsonl` resets on restart. Note this in README under "Honest Limitations." Fix: in Phase 9, logs write to `os.environ.get("LOG_PATH", "logs/requests.jsonl")` so a future upgrade can point to a mounted volume.

### 9.4 Health check
`GET /health` must respond within 10 seconds or Render marks the deploy as failed.

### Checkpoint 9

```bash
docker build -t arbiter .
docker run --env-file .env -p 8000:8000 arbiter &
sleep 5
curl http://localhost:8000/health   # should return 200
docker stop $(docker ps -q --filter "ancestor=arbiter")
```

---

### 🧑 HUMAN — Phase 9 Deploy

1. Go to [render.com](https://render.com) → New → Web Service.
2. Connect GitHub → select `arbiter` repo.
3. Choose "Use render.yaml".
4. Add all env vars in the Render dashboard (GROQ_API_KEY, GEMINI_API_KEY, ARBITER_API_KEY, LANGFUSE keys).
5. Click Deploy.
6. Copy the live URL (e.g. `https://arbiter.onrender.com`).
7. Test: `curl https://arbiter.onrender.com/health`.

Tell Claude Code: **"Deployed at [URL], proceed to Phase 10."**

---

## Phase 10 — Polish + Documentation

**Goal:** README with numbers, interview prep doc, failure modes, clean CI.

### 10.1 README.md sections
1. One-line description + live URL badge
2. Architecture diagram (Mermaid)
3. Key results table (copy from EVAL_REPORT.md)
4. How to run locally
5. API reference (GET /health, POST /generate, GET /stats)
6. Design decisions (3 entries: why rules before sklearn, why per-tenant cache keys, why deterministic cost map)
7. Honest limitations (5 items)
8. Résumé bullet (filled in)

### 10.2 docs/INTERVIEW_PREP.md
Questions and answers for:
- "How does your router decide which model to use?"
- "What happens when the premium model is rate-limited?"
- "Why not just use an LLM to classify tasks?"
- "How did you measure cost savings?"
- "What breaks at 10x traffic?"
- "Why is the cache keyed by tenant_id and tier?"
- "What's the trade-off between rules and ML classifier?"

### 10.3 Failure modes (document in README)
Five real failure modes, e.g.:
1. Render free tier sleeps after 15 min → first request takes ~30s to wake
2. Gemini Pro 50 RPD limit hit during heavy eval → fallback to Flash
3. Cache grows unbounded in production (fix: add max_size + eviction)
4. JSONL log lost on Render restart (fix: external storage)
5. Classifier trained on 40 examples — unusual prompts may misclassify

### 10.4 Final CI check
Add to `.github/workflows/ci.yml`:
- Lint (ruff)
- Tests (pytest -q)
- Docker build (check it builds without error — no run, just build)

### Final Checkpoint

```bash
pytest -q                          # all tests pass
ruff check src/ tests/             # no lint errors
docker build -t arbiter . --no-cache   # builds cleanly
curl https://[render-url]/health   # live URL responds
```

**Expected:** All green.

---

## Phase Summary

| Phase | What Claude Code builds | Human step? |
|---|---|---|
| 0 | Repo skeleton, CI, config | 🧑 Create accounts + GitHub repo |
| 1 | 40 benchmark tasks | 🧑 Review + approve benchmark |
| 2 | FastAPI, auth, rate limit, budget | — |
| 3 | Classifier (rules + sklearn) | — |
| 4 | LiteLLM router, retry, fallback | — |
| 5 | Semantic cache (Gemini embeddings) | — |
| 6 | JSONL logger, /stats, Langfuse | — |
| 7 | Streamlit cost dashboard | — |
| 8 | Eval suite, EVAL_REPORT.md | 🧑 Review eval numbers |
| 9 | Docker, render.yaml, deployment | 🧑 Deploy on Render, share URL |
| 10 | README, interview prep, polish | — |

---

## Résumé Bullet Template

Fill in after Phase 8:

> Developed a multi-provider LLM gateway routing requests across 3 model tiers (Groq 8B / Gemini Flash / Groq 70B) by task complexity, latency SLO, and per-tenant budget; implemented rule-based + sklearn classifier for tier assignment, semantic caching with Gemini embeddings, provider fallback with exponential backoff, and request-level token/cost telemetry; eval-driven routing cut estimated inference spend by **[X]%** while holding quality retention above **[Y]%** on 40 benchmark tasks.
