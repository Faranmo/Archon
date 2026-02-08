"""
Archon Rate Limiter Module

Token bucket rate limiting for API protection.
Prevents abuse and ensures fair usage.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from src.core.config import settings
from src.core.errors import RateLimitError
from src.monitoring.logger import get_logger

logger = get_logger("security.rate_limiter")


# =============================================================================
# Token Bucket Implementation
# =============================================================================

@dataclass
class TokenBucket:
    """
    Token bucket rate limiter.

    Allows burst traffic while maintaining average rate.
    """
    capacity: int  # Maximum tokens
    refill_rate: float  # Tokens per second
    tokens: float = field(default=0.0, init=False)
    last_refill: float = field(default_factory=time.time, init=False)

    def __post_init__(self):
        self.tokens = float(self.capacity)

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if not enough
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def wait_time(self, tokens: int = 1) -> float:
        """
        Calculate wait time until tokens are available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Seconds to wait (0 if tokens available now)
        """
        self._refill()

        if self.tokens >= tokens:
            return 0.0

        needed = tokens - self.tokens
        return needed / self.refill_rate

    async def wait_and_consume(self, tokens: int = 1) -> bool:
        """
        Wait until tokens available, then consume.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True when consumed
        """
        wait = self.wait_time(tokens)
        if wait > 0:
            await asyncio.sleep(wait)
        return self.consume(tokens)


# =============================================================================
# Sliding Window Counter
# =============================================================================

@dataclass
class SlidingWindowCounter:
    """
    Sliding window rate limiter.

    More accurate than fixed windows, counts requests
    over a rolling time window.
    """
    window_size: int  # Window size in seconds
    max_requests: int  # Maximum requests per window
    _requests: list[float] = field(default_factory=list, init=False)

    def _cleanup(self):
        """Remove expired requests."""
        cutoff = time.time() - self.window_size
        self._requests = [t for t in self._requests if t > cutoff]

    def allow(self) -> bool:
        """
        Check if request is allowed.

        Returns:
            True if allowed, False if rate limited
        """
        self._cleanup()

        if len(self._requests) < self.max_requests:
            self._requests.append(time.time())
            return True
        return False

    def remaining(self) -> int:
        """Get remaining requests in current window."""
        self._cleanup()
        return max(0, self.max_requests - len(self._requests))

    def reset_time(self) -> float:
        """Get seconds until oldest request expires."""
        self._cleanup()

        if not self._requests:
            return 0.0

        oldest = min(self._requests)
        return max(0.0, oldest + self.window_size - time.time())


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """
    Multi-tier rate limiter.

    Supports rate limiting by:
    - User ID
    - IP address
    - API key
    - Endpoint
    - Global
    """

    def __init__(
        self,
        requests_per_minute: int = settings.requests_per_minute,
        requests_per_hour: int = settings.requests_per_hour,
        burst_multiplier: float = 2.0,
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_multiplier = burst_multiplier

        # Per-key rate limiters
        self._minute_limiters: dict[str, SlidingWindowCounter] = defaultdict(
            lambda: SlidingWindowCounter(
                window_size=60,
                max_requests=self.requests_per_minute
            )
        )
        self._hour_limiters: dict[str, SlidingWindowCounter] = defaultdict(
            lambda: SlidingWindowCounter(
                window_size=3600,
                max_requests=self.requests_per_hour
            )
        )

        # Token bucket for burst handling
        self._burst_buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(
                capacity=int(self.requests_per_minute * self.burst_multiplier),
                refill_rate=self.requests_per_minute / 60
            )
        )

        # Global limiter
        self._global_limiter = SlidingWindowCounter(
            window_size=60,
            max_requests=requests_per_minute * 10  # 10x individual limit
        )

    def _get_key(
        self,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> str:
        """Generate a rate limit key from identifiers."""
        parts = []
        if user_id:
            parts.append(f"user:{user_id}")
        if api_key:
            # Hash API key for privacy
            parts.append(f"key:{hash(api_key) % 10000}")
        if ip_address:
            parts.append(f"ip:{ip_address}")
        if endpoint:
            parts.append(f"ep:{endpoint}")

        return ":".join(parts) if parts else "anonymous"

    def check(
        self,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
        cost: int = 1,
    ) -> tuple[bool, dict]:
        """
        Check if request is allowed.

        Args:
            user_id: User identifier
            api_key: API key
            ip_address: Client IP
            endpoint: API endpoint
            cost: Request cost (for weighted limiting)

        Returns:
            Tuple of (allowed, headers_dict)
        """
        key = self._get_key(user_id, api_key, ip_address, endpoint)

        # Check global limit first
        if not self._global_limiter.allow():
            logger.warning("Global rate limit exceeded")
            return False, self._get_headers(key, limited=True)

        # Check minute limit
        minute_limiter = self._minute_limiters[key]
        if not minute_limiter.allow():
            logger.warning(
                f"Rate limit exceeded for {key}",
                metadata={"window": "minute", "key": key}
            )
            return False, self._get_headers(key, limited=True)

        # Check hour limit
        hour_limiter = self._hour_limiters[key]
        if not hour_limiter.allow():
            logger.warning(
                f"Rate limit exceeded for {key}",
                metadata={"window": "hour", "key": key}
            )
            return False, self._get_headers(key, limited=True)

        return True, self._get_headers(key, limited=False)

    def _get_headers(self, key: str, limited: bool) -> dict:
        """Generate rate limit headers."""
        minute_limiter = self._minute_limiters[key]

        headers = {
            "X-RateLimit-Limit": str(self.requests_per_minute),
            "X-RateLimit-Remaining": str(minute_limiter.remaining()),
            "X-RateLimit-Reset": str(int(minute_limiter.reset_time())),
        }

        if limited:
            headers["Retry-After"] = str(int(minute_limiter.reset_time()) + 1)

        return headers

    def check_or_raise(
        self,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        """
        Check rate limit and raise if exceeded.

        Raises:
            RateLimitError: If rate limit exceeded
        """
        allowed, headers = self.check(user_id, api_key, ip_address, endpoint)

        if not allowed:
            retry_after = int(headers.get("Retry-After", 60))
            raise RateLimitError(
                message="Rate limit exceeded. Please slow down.",
                retry_after=retry_after,
                details={
                    "limit": self.requests_per_minute,
                    "remaining": headers.get("X-RateLimit-Remaining"),
                    "reset": headers.get("X-RateLimit-Reset"),
                }
            )

    async def wait_if_needed(
        self,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        """
        Wait if rate limited, then proceed.

        For graceful rate limiting without errors.
        """
        key = self._get_key(user_id, api_key, ip_address, endpoint)
        bucket = self._burst_buckets[key]

        await bucket.wait_and_consume(1)

    def reset(self, key: Optional[str] = None):
        """
        Reset rate limits.

        Args:
            key: Specific key to reset, or None for all
        """
        if key:
            self._minute_limiters.pop(key, None)
            self._hour_limiters.pop(key, None)
            self._burst_buckets.pop(key, None)
        else:
            self._minute_limiters.clear()
            self._hour_limiters.clear()
            self._burst_buckets.clear()

        logger.info(f"Rate limits reset: {key or 'all'}")


# =============================================================================
# Global Instance
# =============================================================================

_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


# =============================================================================
# Middleware Helper
# =============================================================================

async def rate_limit_middleware_helper(
    user_id: Optional[str] = None,
    api_key: Optional[str] = None,
    ip_address: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> dict:
    """
    Helper for rate limiting middleware.

    Returns headers to add to response.
    Raises RateLimitError if exceeded.
    """
    limiter = get_rate_limiter()
    limiter.check_or_raise(
        user_id=user_id,
        api_key=api_key,
        ip_address=ip_address,
        endpoint=endpoint,
    )

    _, headers = limiter.check(user_id, api_key, ip_address, endpoint)
    return headers
