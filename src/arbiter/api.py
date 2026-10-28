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
from arbiter.auth import make_require_api_key
from arbiter.cache import CacheEntry, SemanticCache, embed
from arbiter.classifier import apply_tier_cap, classify_with_source
from arbiter.config import get_settings, load_tenants
from arbiter.logger import (
    RequestLog,
    compute_stats,
    init_langfuse,
    read_logs,
    trace_request,
    write_log,
)
from arbiter.router import GatewayError, call_llm, model_list
from arbiter.state_redis import RedisStateError, make_budget_tracker, make_rate_limiter


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    tenant_id: str | None = None  # deprecated and ignored; the tenant comes from the API key
    task_type: str | None = None  # optional hint: "summarize" | "code" | "qa"
    max_tokens: int = Field(default=512, ge=1, le=8192)
    priority: Literal["low", "standard", "high"] = "standard"


STATE_UNAVAILABLE = {
    "error": "state_unavailable",
    "message": "State backend is unavailable; request rejected to protect budget integrity",
}


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
    require_api_key = make_require_api_key(tenants)  # fails fast on keys for unknown tenants
    model_list()  # fail at startup, not at request time, if any model lacks pricing
    langfuse = init_langfuse()

    @asynccontextmanager
    async def lifespan(_app):
        yield
        if langfuse:
            langfuse.shutdown()  # flush pending traces

    app = FastAPI(title="Arbiter", version=__version__, lifespan=lifespan)
    app.state.tenants = tenants
    redis_url = get_settings().redis_url  # set -> Redis state; unreachable at startup raises
    app.state.budget = make_budget_tracker(tenants, redis_url)
    app.state.limiter = make_rate_limiter(tenants, redis_url)
    app.state.cache = SemanticCache()
    started = time.monotonic()

    @app.get("/health")
    async def health():
        result = {
            "status": "ok",
            "version": __version__,
            "uptime_seconds": int(time.monotonic() - started),
            "tenant_count": len(tenants),
        }
        if (r := getattr(app.state.budget, "_r", None)) is not None:  # Redis-backed state
            try:
                await asyncio.to_thread(r.ping)
                result["redis"] = "ok"
            except Exception:  # noqa: BLE001 - /health must answer even when Redis is down
                result["redis"] = "unavailable"
        return result

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest, tenant_id: str = Depends(require_api_key)):
        t0 = time.perf_counter()
        if getattr(app.state.budget, "degraded", False):  # a cost went unrecorded; see state_redis
            return JSONResponse(STATE_UNAVAILABLE, status_code=503)
        try:
            # state calls may hit Redis over the network, so keep them off the event loop
            if not await asyncio.to_thread(app.state.limiter.check, tenant_id):
                return JSONResponse(
                    {"error": "rate_limited", "message": "Request rate limit exceeded"}, status_code=429)
            # Gate only: blocks an exhausted tenant. Real cost is booked after the call, so
            # concurrent in-flight requests can overshoot the budget slightly.
            if not await asyncio.to_thread(app.state.budget.check_and_reserve, tenant_id, 0.0):
                return JSONResponse(
                    {"error": "budget_exceeded", "message": "Daily budget exhausted"}, status_code=429)
        except RedisStateError:
            return JSONResponse(STATE_UNAVAILABLE, status_code=503)

        request_id = f"req_{uuid.uuid4().hex[:12]}"
        tier, source = classify_with_source(req.prompt)
        tier, capped = apply_tier_cap(tier, tenants[tenant_id]["tier_cap"])

        def log(model="", tin=0, tout=0, cost=0.0, hit=False, fb=False, error=None, saved=0.0):
            entry = RequestLog(
                request_id, tenant_id, tier, model, tin, tout, cost,
                (time.perf_counter() - t0) * 1000, hit, fb, error is None, error,
                datetime.now(UTC).isoformat(), saved, capped, source,
            )
            write_log(entry)
            trace_request(langfuse, entry, req.prompt)

        try:
            embedding = await asyncio.to_thread(embed, req.prompt)
        except Exception:  # noqa: BLE001 - the cache is an optimisation; never fail the request
            embedding = None
        if embedding is not None and (hit := app.state.cache.get(tenant_id, tier, embedding)):
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
            return JSONResponse(
                {"error": "all_models_exhausted", "message": "All models failed or are unavailable"},
                status_code=503)

        try:
            await asyncio.to_thread(app.state.budget.record_actual, tenant_id, res.cost_usd,
                                    0.0, request_id)
        except RedisStateError:
            # The provider call already happened and may have cost money, but the spend is
            # NOT recorded (tracker is now degraded and has written a reconciliation record).
            log(res.model, res.tokens_in, res.tokens_out, res.cost_usd,
                fb=res.fallback_triggered, error="cost_record_failed")
            return JSONResponse(STATE_UNAVAILABLE, status_code=503)
        if embedding is not None:
            app.state.cache.put(tenant_id, tier, embedding,
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
