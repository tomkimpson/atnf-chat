"""Rate limiting middleware for ATNF-Chat API.

Provides per-IP rate limiting to prevent abuse when using a server-side API key.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


@dataclass
class RateLimitState:
    """Tracks rate limit state for a single client."""

    requests: list[float] = field(default_factory=list)

    def cleanup(self, window_seconds: float) -> None:
        """Remove requests outside the time window."""
        cutoff = time.time() - window_seconds
        self.requests = [t for t in self.requests if t > cutoff]

    def add_request(self) -> None:
        """Record a new request."""
        self.requests.append(time.time())

    def count(self) -> int:
        """Get number of requests in current window."""
        return len(self.requests)


class RateLimiter:
    """In-memory rate limiter with per-IP tracking.

    For production at scale, consider using Redis-based rate limiting.
    This implementation is suitable for moderate traffic levels.
    """

    def __init__(
        self,
        requests_per_minute: int = 20,
        requests_per_hour: int = 200,
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self._minute_state: dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._hour_state: dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # Cleanup every 5 minutes

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request, handling proxies."""
        # Check for forwarded headers (common with reverse proxies)
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # Take the first IP in the chain
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fall back to direct client
        return request.client.host if request.client else "unknown"

    def _maybe_cleanup(self) -> None:
        """Periodically cleanup old entries to prevent memory growth."""
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            # Clean up stale entries
            stale_minute_keys = [
                k for k, v in self._minute_state.items()
                if not v.requests or max(v.requests) < now - 120
            ]
            for k in stale_minute_keys:
                del self._minute_state[k]

            stale_hour_keys = [
                k for k, v in self._hour_state.items()
                if not v.requests or max(v.requests) < now - 7200
            ]
            for k in stale_hour_keys:
                del self._hour_state[k]

            self._last_cleanup = now

    def check_rate_limit(self, request: Request) -> tuple[bool, str | None]:
        """Check if request is within rate limits.

        Returns:
            Tuple of (is_allowed, error_message)
        """
        self._maybe_cleanup()

        client_ip = self._get_client_ip(request)

        # Check minute limit
        minute_state = self._minute_state[client_ip]
        minute_state.cleanup(60)
        if minute_state.count() >= self.requests_per_minute:
            return False, f"Rate limit exceeded: {self.requests_per_minute} requests per minute"

        # Check hour limit
        hour_state = self._hour_state[client_ip]
        hour_state.cleanup(3600)
        if hour_state.count() >= self.requests_per_hour:
            return False, f"Rate limit exceeded: {self.requests_per_hour} requests per hour"

        # Record the request
        minute_state.add_request()
        hour_state.add_request()

        return True, None

    def get_remaining(self, request: Request) -> dict[str, int]:
        """Get remaining requests for rate limit headers."""
        client_ip = self._get_client_ip(request)

        minute_state = self._minute_state.get(client_ip, RateLimitState())
        minute_state.cleanup(60)

        hour_state = self._hour_state.get(client_ip, RateLimitState())
        hour_state.cleanup(3600)

        return {
            "minute": max(0, self.requests_per_minute - minute_state.count()),
            "hour": max(0, self.requests_per_hour - hour_state.count()),
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting chat endpoints."""

    def __init__(
        self,
        app,
        rate_limiter: RateLimiter,
        protected_paths: list[str] | None = None,
    ):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.protected_paths = protected_paths or ["/chat/"]

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request through rate limiter."""
        # Only rate limit specific paths
        path = request.url.path
        should_limit = any(path.startswith(p) for p in self.protected_paths)

        # Skip rate limiting if user provides their own API key
        has_user_key = request.headers.get("X-API-Key") is not None

        if should_limit and not has_user_key:
            allowed, error = self.rate_limiter.check_rate_limit(request)
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail=error,
                    headers={"Retry-After": "60"},
                )

        response = await call_next(request)

        # Add rate limit headers for transparency
        if should_limit and not has_user_key:
            remaining = self.rate_limiter.get_remaining(request)
            response.headers["X-RateLimit-Remaining-Minute"] = str(remaining["minute"])
            response.headers["X-RateLimit-Remaining-Hour"] = str(remaining["hour"])

        return response


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        from atnf_chat.config import get_settings
        settings = get_settings()
        _rate_limiter = RateLimiter(
            requests_per_minute=settings.max_api_calls_per_minute,
            requests_per_hour=settings.max_api_calls_per_minute * 10,  # 10x minute limit
        )
    return _rate_limiter
