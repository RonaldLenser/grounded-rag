"""Per-client rate limiting (sliding window, in memory).

Why: /query and /search call paid APIs. On a public demo, one script hitting
them in a loop could run up the bill. This caps each client IP.

How it works (sliding window log): for each client we keep the timestamps of
their recent requests. A new request is allowed if fewer than `limit`
timestamps fall inside the last `window` seconds.

Tradeoffs, stated honestly:
- In memory, per instance: with N Cloud Run instances a client could get up
  to N x limit. Fine for a demo with max 2 instances; a shared store (Redis)
  would make it exact. The global DAILY cap (answerer.py) is DB-backed and
  exact, so total spend stays bounded regardless.
- Keyed by client IP. Behind Cloud Run's proxy the TCP peer is Google's front
  end, so the real client IP comes from X-Forwarded-For, read in a way clients
  can't spoof (see client_key and the TRUSTED_PROXY_HOPS setting).
"""

import math
import time
from collections import deque
from collections.abc import Callable

import structlog
from fastapi import Request

from grounded.errors import RateLimitedError

MAX_TRACKED_CLIENTS = 10_000

logger = structlog.get_logger(__name__)


class SlidingWindowRateLimiter:
    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> None:
        """Record a request for `key`, or raise RateLimitedError if over the limit."""
        now = self._clock()
        hits = self._hits.setdefault(key, deque())
        while hits and hits[0] <= now - self.window_seconds:
            hits.popleft()

        if len(hits) >= self.limit:
            retry_after = math.ceil(hits[0] + self.window_seconds - now)
            raise RateLimitedError(max(retry_after, 1))

        hits.append(now)
        if len(self._hits) > MAX_TRACKED_CLIENTS:
            self._evict_idle(now)

    def _evict_idle(self, now: float) -> None:
        """Bound memory: forget clients with no requests inside the window."""
        cutoff = now - self.window_seconds
        for key in [k for k, hits in self._hits.items() if not hits or hits[-1] <= cutoff]:
            del self._hits[key]


def client_key(request: Request, trusted_proxy_hops: int) -> str:
    """The client's IP, resistant to X-Forwarded-For spoofing.

    Each proxy APPENDS the address it received the request from. With N trusted
    proxies, the Nth entry from the right was written by our own infrastructure;
    everything to its left could have been sent by the client and is ignored.
    """
    peer = request.client.host if request.client else "unknown"
    if trusted_proxy_hops == 0:
        return peer
    forwarded = request.headers.get("x-forwarded-for", "")
    entries = [entry.strip() for entry in forwarded.split(",") if entry.strip()]
    return entries[-trusted_proxy_hops] if len(entries) >= trusted_proxy_hops else peer


def rate_limit(name: str) -> Callable[[Request], None]:
    """FastAPI dependency: enforce the limiter stored at app.state.rate_limiters[name]."""

    def dependency(request: Request) -> None:
        limiter: SlidingWindowRateLimiter = request.app.state.rate_limiters[name]
        key = client_key(request, request.app.state.settings.trusted_proxy_hops)
        try:
            limiter.check(key)
        except RateLimitedError:
            logger.warning("rate_limited", limiter=name, client=key)
            raise

    return dependency
