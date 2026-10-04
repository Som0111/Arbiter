import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime


def _utc_today() -> str:
    return datetime.now(UTC).date().isoformat()


class BudgetTracker:
    """In-memory daily spend per tenant. The date key rolls over at UTC midnight."""

    def __init__(self, tenants: dict, today=_utc_today):
        self._limits = {t: float(c["daily_budget_usd"]) for t, c in tenants.items()}
        self._today = today
        self._spent: dict[str, dict[str, float]] = defaultdict(dict)
        self._lock = threading.Lock()

    def _spent_today(self, tenant_id: str) -> float:
        days = self._spent[tenant_id]
        day = self._today()
        for old in [d for d in days if d != day]:
            del days[old]
        return days.get(day, 0.0)

    def get_remaining(self, tenant_id: str) -> float:
        with self._lock:
            return self._limits[tenant_id] - self._spent_today(tenant_id)

    def check_and_reserve(self, tenant_id: str, estimated_cost: float) -> bool:
        """Reserve estimated_cost against today's budget. False = over budget."""
        with self._lock:
            remaining = self._limits[tenant_id] - self._spent_today(tenant_id)
            if remaining <= 0 or estimated_cost > remaining:
                return False
            self._spent[tenant_id][self._today()] = self._spent_today(tenant_id) + estimated_cost
            return True

    def record_actual(self, tenant_id: str, actual_cost: float, reserved: float = 0.0) -> None:
        """Book the real cost, replacing any `reserved` amount taken by check_and_reserve."""
        with self._lock:
            day = self._today()
            self._spent[tenant_id][day] = self._spent_today(tenant_id) + actual_cost - reserved


class RateLimiter:
    """Sliding 60-second window per tenant."""

    def __init__(self, tenants: dict, clock=time.monotonic):
        self._rpm = {t: int(c["rate_limit_rpm"]) for t, c in tenants.items()}
        self._clock = clock
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, tenant_id: str) -> bool:
        """Count this request. False = over the tenant's rpm limit."""
        with self._lock:
            now = self._clock()
            hits = self._hits[tenant_id]
            while hits and now - hits[0] >= 60:
                hits.popleft()
            if len(hits) >= self._rpm[tenant_id]:
                return False
            hits.append(now)
            return True
