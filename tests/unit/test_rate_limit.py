import pytest
from starlette.requests import Request

from grounded.api.rate_limit import SlidingWindowRateLimiter, client_key
from grounded.errors import RateLimitedError
from grounded.logging_config import add_severity


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_allows_up_to_the_limit_then_blocks() -> None:
    limiter = SlidingWindowRateLimiter(limit=3, window_seconds=60, clock=FakeClock())

    for _ in range(3):
        limiter.check("1.2.3.4")
    with pytest.raises(RateLimitedError) as excinfo:
        limiter.check("1.2.3.4")

    assert excinfo.value.status_code == 429
    assert excinfo.value.headers == {"Retry-After": "60"}


def test_window_slides_instead_of_resetting_all_at_once() -> None:
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60, clock=clock)
    limiter.check("ip")  # t=1000
    clock.now = 1030
    limiter.check("ip")  # t=1030

    clock.now = 1059
    with pytest.raises(RateLimitedError) as excinfo:
        limiter.check("ip")
    assert excinfo.value.headers["Retry-After"] == "1"  # first hit expires at t=1060

    clock.now = 1061  # first request left the window; one slot free
    limiter.check("ip")
    with pytest.raises(RateLimitedError):
        limiter.check("ip")


def test_clients_are_limited_independently() -> None:
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60, clock=FakeClock())

    limiter.check("alice")
    limiter.check("bob")  # bob is unaffected by alice
    with pytest.raises(RateLimitedError):
        limiter.check("alice")


def test_rejected_requests_do_not_extend_the_block() -> None:
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60, clock=clock)
    limiter.check("ip")
    for _ in range(5):
        with pytest.raises(RateLimitedError):
            limiter.check("ip")

    clock.now = 1061
    limiter.check("ip")  # hammering while blocked didn't push the window back


def test_idle_clients_are_evicted_to_bound_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("grounded.api.rate_limit.MAX_TRACKED_CLIENTS", 3)
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(limit=5, window_seconds=60, clock=clock)
    for ip in ("a", "b", "c"):
        limiter.check(ip)

    clock.now = 2000  # everyone is idle now
    limiter.check("d")
    limiter.check("e")

    assert set(limiter._hits) <= {"d", "e"}


def test_json_logs_get_a_google_cloud_severity() -> None:
    assert add_severity(None, "info", {"level": "warning"})["severity"] == "WARNING"


def _request(peer: str, forwarded: str | None = None) -> Request:
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded else []
    return Request({"type": "http", "headers": headers, "client": (peer, 1234)})


def test_without_trusted_proxies_the_header_is_ignored() -> None:
    assert client_key(_request("10.0.0.1", "6.6.6.6"), trusted_proxy_hops=0) == "10.0.0.1"


def test_behind_one_proxy_the_last_entry_is_used() -> None:
    assert client_key(_request("35.1.1.1", "203.0.113.7"), trusted_proxy_hops=1) == "203.0.113.7"


def test_spoofed_entries_on_the_left_are_ignored() -> None:
    """An attacker sends a fake X-Forwarded-For; the proxy appends the real IP after it."""
    request = _request("35.1.1.1", "1.1.1.1, 2.2.2.2, 203.0.113.7")

    assert client_key(request, trusted_proxy_hops=1) == "203.0.113.7"


def test_missing_header_falls_back_to_the_peer() -> None:
    assert client_key(_request("35.1.1.1"), trusted_proxy_hops=1) == "35.1.1.1"
