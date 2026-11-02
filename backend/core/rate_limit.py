"""In-process sliding-window rate limiter exposed as a FastAPI dependency.

Requests are bucketed per API key (when auth is on) or per client IP.
This is intentionally dependency-free; for multi-instance deployments swap
:class:`SlidingWindowLimiter` for a Redis-backed implementation with the
same ``hit`` signature.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable

from fastapi import Depends, HTTPException, Request, Response, status

from backend.core.config import Settings, get_settings
from backend.core.security import require_api_key


class SlidingWindowLimiter:
    """Thread-safe sliding-window-log limiter."""

    def __init__(self, window_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int) -> tuple[bool, int, float]:
        """Record a hit. Returns ``(allowed, remaining, seconds_until_reset)``."""
        now = self._clock()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            cutoff = now - self.window
            while q and q[0] <= cutoff:
                q.popleft()
            if len(q) >= limit:
                return False, 0, q[0] + self.window - now
            q.append(now)
            reset = q[0] + self.window - now
            # Opportunistic cleanup so idle keys don't accumulate forever.
            if len(self._hits) > 10_000:
                self._hits = {k: v for k, v in self._hits.items() if v and v[-1] > cutoff}
            return True, limit - len(q), reset

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = SlidingWindowLimiter()


def rate_limit(scope: str, limit_of: Callable[[Settings], int]):
    """Dependency factory: ``Depends(rate_limit("predict", lambda s: s.rate_limit_per_minute))``."""

    def dependency(
        request: Request,
        response: Response,
        key_id: str | None = Depends(require_api_key),
        settings: Settings = Depends(get_settings),
    ) -> None:
        limit = limit_of(settings)
        who = key_id or (request.client.host if request.client else "anonymous")
        allowed, remaining, reset = limiter.hit(f"{scope}:{who}", limit)
        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(math.ceil(reset)),
        }
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {limit} requests/minute for '{scope}'",
                headers={**headers, "Retry-After": str(max(1, math.ceil(reset)))},
            )
        response.headers.update(headers)

    return dependency
