import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Literal

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from arbiter import __version__
from arbiter.auth import require_api_key
from arbiter.budget import BudgetTracker, RateLimiter
from arbiter.cache import CacheEntry, SemanticCache, embed
from arbiter.classifier import apply_tier_cap, classify_with_source
from arbiter.config import load_tenants
from arbiter.logger import (
    RequestLog,
    compute_stats,
    init_langfuse,
    read_logs,
    trace_request,
    write_log,
)
from arbiter.router import GatewayError, call_llm, model_list


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    tenant_id: str
    task_type: str | None = None  # optional hint: "summarize" | "code" | "qa"
    max_tokens: int = Field(default=512, ge=1, le=8192)
    priority: Literal["low", "standard", "high"] = "standard"


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


def create_app(tenants: dict | None = None) -> FastAPI:
    tenants = tenants if tenants is not None else load_tenants()
    model_list()  # fail at startup, not at request time, if any model lacks pricing
    langfuse = init_langfuse()

    @asynccontextmanager
    async def lifespan(_app):
        yield
        if langfuse:
            langfuse.shutdown()  # flush pending traces

    app = FastAPI(title="Arbiter", version=__version__, lifespan=lifespan)
    app.state.tenants = tenants
    app.state.budget = BudgetTracker(tenants)
    app.state.limiter = RateLimiter(tenants)
    app.state.cache = SemanticCache()
    started = time.monotonic()

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "version": __version__,
            "uptime_seconds": int(time.monotonic() - started),
            "tenant_count": len(tenants),
        }

    @app.post("/generate", response_model=GenerateResponse, dependencies=[Depends(require_api_key)])
    async def generate(req: GenerateRequest):
        t0 = time.perf_counter()
        if req.tenant_id not in tenants:
            return JSONResponse({"error": "unknown_tenant"}, status_code=404)
        if not app.state.limiter.check(req.tenant_id):
            return JSONResponse({"error": "rate_limited"}, status_code=429)
        # Gate only: blocks an exhausted tenant. Real cost is booked after the call, so
        # concurrent in-flight requests can overshoot the budget slightly.
        if not app.state.budget.check_and_reserve(req.tenant_id, 0.0):
            return JSONResponse({"error": "budget_exceeded"}, status_code=429)

        request_id = f"req_{uuid.uuid4().hex[:12]}"
        tier, source = classify_with_source(req.prompt)
        tier, capped = apply_tier_cap(tier, tenants[req.tenant_id]["tier_cap"])

        def log(model="", tin=0, tout=0, cost=0.0, hit=False, fb=False, error=None, saved=0.0):
            entry = RequestLog(
                request_id, req.tenant_id, tier, model, tin, tout, cost,
                (time.perf_counter() - t0) * 1000, hit, fb, error is None, error,
                datetime.now(UTC).isoformat(), saved, capped, source,
            )
            write_log(entry)
            trace_request(langfuse, entry, req.prompt)

        try:
            embedding = await asyncio.to_thread(embed, req.prompt)
        except Exception:  # noqa: BLE001 - the cache is an optimisation; never fail the request
            embedding = None
        if embedding is not None and (hit := app.state.cache.get(req.tenant_id, tier, embedding)):
            log(hit.model, hit=True, saved=hit.cost_usd)
            return GenerateResponse(
                response=hit.response,
                model_used=hit.model,
                tier=tier,
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                cache_hit=True,
                fallback_triggered=False,
                request_id=request_id,
            )

        try:
            # call_llm blocks (sync LiteLLM + backoff sleeps), so keep it off the event loop
            res = await asyncio.to_thread(call_llm, tier, req.prompt, req.max_tokens)
        except GatewayError:
            log(error="all_models_exhausted")
            return JSONResponse({"error": "all_models_exhausted"}, status_code=503)

        app.state.budget.record_actual(req.tenant_id, res.cost_usd)
        if embedding is not None:
            app.state.cache.put(req.tenant_id, tier, embedding,
                                CacheEntry(embedding, res.text, res.model, res.cost_usd, time.time()))
        log(res.model, res.tokens_in, res.tokens_out, res.cost_usd, fb=res.fallback_triggered)
        return GenerateResponse(
            response=res.text,
            model_used=res.model,
            tier=tier,
            tokens_in=res.tokens_in,
            tokens_out=res.tokens_out,
            cost_usd=res.cost_usd,
            latency_ms=(time.perf_counter() - t0) * 1000,
            cache_hit=False,
            fallback_triggered=res.fallback_triggered,
            request_id=request_id,
        )

    @app.get("/stats", dependencies=[Depends(require_api_key)])
    async def stats():
        cache = app.state.cache
        return {
            **compute_stats(read_logs()),
            "cache_size": cache.size(),
            "cache_max_size": cache.max_size,
            "cache_evictions_total": cache.evictions_total,
        }

    return app


app = create_app()
