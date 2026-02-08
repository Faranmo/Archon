"""
Archon Retry & Circuit Breaker Module

Production-grade retry logic with exponential backoff,
jitter, and circuit breaker pattern for resilient systems.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, Type, TypeVar, Union

from src.core.errors import (
    ArchonError,
    ErrorCategory,
    RateLimitError,
    classify_exception,
)
from src.monitoring.logger import get_logger

logger = get_logger("core.retry")

T = TypeVar("T")


# =============================================================================
# Retry Configuration
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1

    # Which error categories to retry
    retryable_categories: set[ErrorCategory] = field(default_factory=lambda: {
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.TIMEOUT,
        ErrorCategory.NETWORK_ERROR,
        ErrorCategory.MODEL_ERROR,
    })

    # Specific exceptions to retry (in addition to ArchonError)
    retryable_exceptions: tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
    )


# Default configs for different scenarios
DEFAULT_RETRY_CONFIG = RetryConfig()

AGGRESSIVE_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    base_delay=0.5,
    max_delay=30.0,
)

CONSERVATIVE_RETRY_CONFIG = RetryConfig(
    max_retries=2,
    base_delay=2.0,
    max_delay=10.0,
)


# =============================================================================
# Retry Logic
# =============================================================================

def calculate_delay(
    attempt: int,
    config: RetryConfig,
    retry_after: Optional[int] = None,
) -> float:
    """
    Calculate delay before next retry.

    Uses exponential backoff with optional jitter.
    Respects retry_after from rate limit errors.
    """
    if retry_after:
        # Use retry_after if provided (e.g., from rate limit header)
        delay = float(retry_after)
    else:
        # Exponential backoff
        delay = config.base_delay * (config.exponential_base ** attempt)

    # Cap at max delay
    delay = min(delay, config.max_delay)

    # Add jitter to prevent thundering herd
    if config.jitter:
        jitter_range = delay * config.jitter_factor
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0, delay)


def should_retry(
    error: Exception,
    attempt: int,
    config: RetryConfig,
) -> bool:
    """Determine if we should retry based on the error and attempt count."""
    if attempt >= config.max_retries:
        return False

    # Check if it's an ArchonError with recoverable flag
    if isinstance(error, ArchonError):
        if not error.recoverable:
            return False
        if error.category in config.retryable_categories:
            return True

    # Check if it's a retryable exception type
    if isinstance(error, config.retryable_exceptions):
        return True

    # Classify unknown exceptions
    classified = classify_exception(error)
    return classified.recoverable and classified.category in config.retryable_categories


def with_retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorator for adding retry logic to synchronous functions.

    Args:
        config: Retry configuration
        on_retry: Callback called before each retry (error, attempt)

    Example:
        @with_retry(config=RetryConfig(max_retries=3))
        def call_api():
            ...
    """
    config = config or DEFAULT_RETRY_CONFIG

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error: Optional[Exception] = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    if not should_retry(e, attempt, config):
                        raise

                    # Get retry_after if available
                    retry_after = None
                    if isinstance(e, ArchonError):
                        retry_after = e.retry_after

                    delay = calculate_delay(attempt, config, retry_after)

                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_retries} for {func.__name__}",
                        metadata={
                            "error": str(e),
                            "delay_seconds": delay,
                            "attempt": attempt + 1,
                        }
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    time.sleep(delay)

            # Should not reach here, but just in case
            raise last_error  # type: ignore

        return wrapper
    return decorator


def with_async_retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorator for adding retry logic to async functions.

    Args:
        config: Retry configuration
        on_retry: Callback called before each retry (error, attempt)

    Example:
        @with_async_retry(config=RetryConfig(max_retries=3))
        async def call_api():
            ...
    """
    config = config or DEFAULT_RETRY_CONFIG

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_error: Optional[Exception] = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    if not should_retry(e, attempt, config):
                        raise

                    # Get retry_after if available
                    retry_after = None
                    if isinstance(e, ArchonError):
                        retry_after = e.retry_after

                    delay = calculate_delay(attempt, config, retry_after)

                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_retries} for {func.__name__}",
                        metadata={
                            "error": str(e),
                            "delay_seconds": delay,
                            "attempt": attempt + 1,
                        }
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    await asyncio.sleep(delay)

            raise last_error  # type: ignore

        return wrapper
    return decorator


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascade failures by stopping requests to failing services.

    States:
    - CLOSED: Normal operation, requests flow through
    - OPEN: Service is down, reject requests immediately
    - HALF_OPEN: Testing if service recovered

    Example:
        breaker = CircuitBreaker(name="openai", failure_threshold=5)

        try:
            result = breaker.call(lambda: call_openai())
        except CircuitOpenError:
            # Use fallback
            result = fallback_response()
    """

    name: str
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    half_open_max_calls: int = 3

    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: Optional[datetime] = field(default=None, init=False)
    _half_open_calls: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for recovery timeout."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time:
                elapsed = datetime.now() - self._last_failure_time
                if elapsed > timedelta(seconds=self.recovery_timeout):
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(
                        f"Circuit {self.name} entering half-open state",
                        metadata={"circuit": self.name}
                    )
        return self._state

    def _record_success(self):
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls >= self.half_open_max_calls:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info(
                    f"Circuit {self.name} closed after recovery",
                    metadata={"circuit": self.name}
                )
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def _record_failure(self, error: Exception):
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = datetime.now()

        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open goes back to open
            self._state = CircuitState.OPEN
            logger.warning(
                f"Circuit {self.name} re-opened after half-open failure",
                metadata={"circuit": self.name, "error": str(error)}
            )
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.error(
                f"Circuit {self.name} opened after {self._failure_count} failures",
                metadata={"circuit": self.name, "error": str(error)}
            )

    def call(self, func: Callable[[], T]) -> T:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to execute

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open
            Exception: If function fails
        """
        state = self.state

        if state == CircuitState.OPEN:
            raise CircuitOpenError(
                f"Circuit {self.name} is open",
                circuit_name=self.name,
                retry_after=self.recovery_timeout,
            )

        try:
            result = func()
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

    async def call_async(self, func: Callable[[], T]) -> T:
        """
        Execute an async function through the circuit breaker.

        Args:
            func: Async function to execute

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open
            Exception: If function fails
        """
        state = self.state

        if state == CircuitState.OPEN:
            raise CircuitOpenError(
                f"Circuit {self.name} is open",
                circuit_name=self.name,
                retry_after=self.recovery_timeout,
            )

        try:
            result = await func()
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

    def reset(self):
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        logger.info(f"Circuit {self.name} manually reset")


class CircuitOpenError(ArchonError):
    """Raised when circuit breaker is open."""

    def __init__(
        self,
        message: str = "Circuit breaker is open",
        circuit_name: Optional[str] = None,
        retry_after: int = 60,
        **kwargs
    ):
        details = kwargs.pop("details", {})
        if circuit_name:
            details["circuit_name"] = circuit_name

        super().__init__(
            message=message,
            category=ErrorCategory.NETWORK_ERROR,
            recoverable=True,
            retry_after=retry_after,
            details=details,
            **kwargs
        )


# =============================================================================
# Pre-configured Circuit Breakers
# =============================================================================

# Circuit breakers for common services
openai_circuit = CircuitBreaker(name="openai", failure_threshold=5, recovery_timeout=60)
anthropic_circuit = CircuitBreaker(name="anthropic", failure_threshold=5, recovery_timeout=60)
database_circuit = CircuitBreaker(name="database", failure_threshold=3, recovery_timeout=30)
