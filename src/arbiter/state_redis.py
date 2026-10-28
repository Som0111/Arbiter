"""Redis-backed rate limiter and budget tracker with atomic operations.

Failure policy:
- Startup: if REDIS_URL is set but Redis is unreachable, raise RuntimeError (fail fast).
- Runtime: if a Redis call fails, raise RedisStateError - the caller returns HTTP 503.
  We do NOT fall back to in-memory, because that would silently lose all Redis state
  (spent budgets, rate limit windows) accumulated since startup.
"""
import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime

import redis as redis_lib
from redis import Redis

log = logging.getLogger(__name__)

RECORD_ATTEMPTS = 3
RETRY_SLEEP_S = 0.1


class RedisStateError(RuntimeError):
    """Redis became unavailable after startup. The request must be rejected."""


def _utc_today() -> str:
    return datetime.now(UTC).date().isoformat()


# Atomically check whether a tenant is under budget: 1 = allowed, 0 = over budget.
# KEYS[1]=spend_key, ARGV[1]=daily_limit, ARGV[2]=estimated_cost
_CHECK_BUDGET_LUA = """
local spent = tonumber(redis.call('GET', KEYS[1]) or '0')
if spent >= tonumber(ARGV[1]) or spent + tonumber(ARGV[2]) > tonumber(ARGV[1]) then return 0 end
return 1
"""

# Atomically increment spend and refresh the key's TTL.
# KEYS[1]=spend_key, ARGV[1]=delta, ARGV[2]=ttl_seconds
_RECORD_SPEND_LUA = """
local new = redis.call('INCRBYFLOAT', KEYS[1], ARGV[1])
redis.call('EXPIRE', KEYS[1], ARGV[2])
return new
"""

# Sliding window: drop expired members, count, and insert only if allowed.
# KEYS[1]=key, ARGV = now, window_start, limit, unique member
_CHECK_AND_ADD_RATE_LIMIT_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window_start = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)
local count = redis.call('ZCARD', key)
if count >= limit then
  redis.call('EXPIRE', key, 120)
  return 0
end
redis.call('ZADD', key, now, member)
redis.call('EXPIRE', key, 120)
return 1
"""


class RedisRateLimiter:
    """Atomic sliding 60-second window per tenant using a Redis sorted set.

    Rejected requests do not consume a slot. Each request uses a unique member, so
    requests sharing a timestamp are all counted. Raises RedisStateError on Redis failure.
    """

    def __init__(self, tenants: dict, r: Redis):
        self._rpm = {t: int(c["rate_limit_rpm"]) for t, c in tenants.items()}
        self._script = r.register_script(_CHECK_AND_ADD_RATE_LIMIT_LUA)

    def check(self, tenant_id: str) -> bool:
        now = time.time()
        try:
            return bool(self._script(
                keys=[f"arbiter:rl:{tenant_id}"],
                args=[str(now), str(now - 60.0), str(self._rpm[tenant_id]), uuid.uuid4().hex],
            ))
        except redis_lib.RedisError as e:
            raise RedisStateError(f"rate limiter Redis call failed: {e}") from e


class RedisBudgetTracker:
    """Soft daily spend cap per tenant. Check and record are separate Redis operations.

    This is a **soft cap**: the check passes while spend is under the limit, and the real
    cost is recorded only after the LLM call returns. Concurrent in-flight requests can
    therefore overshoot the daily limit slightly - intentional, and the same as the
    in-memory BudgetTracker. A hard cap would need the exact cost before the call.

    If recording a cost still fails after RECORD_ATTEMPTS tries, the provider call has
    already happened and may have incurred cost. We log CRITICAL, append a redacted record
    to unrecorded_costs.jsonl next to the request log, set `degraded`, and raise
    RedisStateError. While degraded every check raises, so /generate returns 503 until the
    process is restarted; replay unrecorded_costs.jsonl into Redis before or after that.
    Retries use INCRBYFLOAT, which is not idempotent: a write that succeeded but lost its
    reply can be counted twice (overcounting spend, never undercounting).

    Raises RedisStateError on any Redis failure.
    """

    def __init__(self, tenants: dict, r: Redis):
        self._limits = {t: float(c["daily_budget_usd"]) for t, c in tenants.items()}
        self._r = r
        self._check_script = r.register_script(_CHECK_BUDGET_LUA)
        self._record_script = r.register_script(_RECORD_SPEND_LUA)
        self.degraded = False

    def _spend_key(self, tenant_id: str) -> str:
        return f"arbiter:budget:{tenant_id}:{_utc_today()}"

    def get_remaining(self, tenant_id: str) -> float:
        try:
            spent = float(self._r.get(self._spend_key(tenant_id)) or 0)
        except redis_lib.RedisError as e:
            raise RedisStateError(f"budget get_remaining Redis call failed: {e}") from e
        return self._limits[tenant_id] - spent

    def check_and_reserve(self, tenant_id: str, estimated_cost: float = 0.0) -> bool:
        """True if today's spend (plus estimated_cost) is under the cap. Reserves nothing."""
        if self.degraded:
            raise RedisStateError("budget state degraded: a cost could not be recorded")
        try:
            return bool(self._check_script(
                keys=[self._spend_key(tenant_id)],
                args=[str(self._limits[tenant_id]), str(estimated_cost)],
            ))
        except redis_lib.RedisError as e:
            raise RedisStateError(f"budget check Redis call failed: {e}") from e

    def record_actual(self, tenant_id: str, actual_cost: float, reserved: float = 0.0,
                      request_id: str = "") -> None:
        delta = actual_cost - reserved
        if delta == 0:
            return
        last: Exception | None = None
        for attempt in range(RECORD_ATTEMPTS):
            try:
                self._record_script(keys=[self._spend_key(tenant_id)],
                                    args=[str(delta), "172800"])  # expire after 2 days
                return
            except redis_lib.RedisError as e:
                last = e
                if attempt < RECORD_ATTEMPTS - 1:
                    time.sleep(RETRY_SLEEP_S * (attempt + 1))
        self.degraded = True
        log.critical("UNRECORDED COST request_id=%s tenant_id=%s cost_usd=%s: Redis write failed "
                     "after %d attempts; budget state is degraded",
                     request_id, tenant_id, delta, RECORD_ATTEMPTS)
        _write_unrecorded(request_id, tenant_id, delta)
        raise RedisStateError("budget record Redis call failed; cost not recorded") from last


def _write_unrecorded(request_id: str, tenant_id: str, cost_usd: float) -> None:
    """Best-effort reconciliation record (no keys, no prompts)."""
    try:
        from arbiter.logger import log_path

        path = os.path.join(os.path.dirname(log_path()) or ".", "unrecorded_costs.jsonl")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps({"request_id": request_id, "tenant_id": tenant_id,
                                "cost_usd": cost_usd,
                                "timestamp": datetime.now(UTC).isoformat()}) + "\n")
    except Exception:
        log.exception("could not write unrecorded cost record")


def make_redis_client(redis_url: str) -> Redis:
    """Connect to Redis. Raises RuntimeError if unreachable (called at startup)."""
    try:
        r = redis_lib.from_url(redis_url, socket_connect_timeout=3, socket_timeout=5,
                               decode_responses=True)
        r.ping()
        return r
    except (redis_lib.RedisError, ValueError) as e:
        raise RuntimeError(
            f"REDIS_URL is set but Redis is unreachable at startup: {type(e).__name__}\n"
            "Fix the connection or unset REDIS_URL to use in-memory state."
        ) from e  # message omits str(e): it can echo the URL, including its password


def make_rate_limiter(tenants: dict, redis_url: str = ""):
    """RedisRateLimiter if redis_url is set (fails fast on a bad URL), else in-memory."""
    from arbiter.budget import RateLimiter

    if redis_url:
        return RedisRateLimiter(tenants, make_redis_client(redis_url))
    return RateLimiter(tenants)


def make_budget_tracker(tenants: dict, redis_url: str = ""):
    """RedisBudgetTracker if redis_url is set (fails fast on a bad URL), else in-memory."""
    from arbiter.budget import BudgetTracker

    if redis_url:
        return RedisBudgetTracker(tenants, make_redis_client(redis_url))
    return BudgetTracker(tenants)
