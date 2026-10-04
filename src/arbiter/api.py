import time
import uuid
from typing import Literal

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from arbiter import __version__
from arbiter.auth import require_api_key
from arbiter.budget import BudgetTracker, RateLimiter
from arbiter.config import load_tenants
from arbiter.logger import compute_stats, read_logs


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
    app = FastAPI(title="Arbiter", version=__version__)
    app.state.tenants = tenants
    app.state.budget = BudgetTracker(tenants)
    app.state.limiter = RateLimiter(tenants)
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
        if not app.state.budget.check_and_reserve(req.tenant_id, 0.0):
            return JSONResponse({"error": "budget_exceeded"}, status_code=429)
        # Stub until the router lands (Phase 4).
        return GenerateResponse(
            response="stub",
            model_used="stub",
            tier="simple",
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            cache_hit=False,
            fallback_triggered=False,
            request_id=f"req_{uuid.uuid4().hex[:12]}",
        )

    @app.get("/stats", dependencies=[Depends(require_api_key)])
    async def stats():
        return compute_stats(read_logs())

    return app


app = create_app()
