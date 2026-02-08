"""
Archon Error Handling Module

Centralized error definitions and handling utilities.
Based on production best practices for agentic AI systems.
"""

from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass, field
import traceback


class ErrorCategory(Enum):
    """Categories for classifying errors."""

    # LLM-related errors
    RATE_LIMIT = "rate_limit"
    TOKEN_LIMIT = "token_limit"
    MODEL_ERROR = "model_error"
    INVALID_RESPONSE = "invalid_response"

    # Infrastructure errors
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    DATABASE_ERROR = "database_error"
    CACHE_ERROR = "cache_error"

    # Input/Output errors
    INVALID_INPUT = "invalid_input"
    VALIDATION_ERROR = "validation_error"
    PARSING_ERROR = "parsing_error"

    # Security errors
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    PROMPT_INJECTION = "prompt_injection"

    # Agent errors
    TOOL_ERROR = "tool_error"
    AGENT_ERROR = "agent_error"
    ORCHESTRATION_ERROR = "orchestration_error"

    # RAG errors
    RETRIEVAL_ERROR = "retrieval_error"
    EMBEDDING_ERROR = "embedding_error"
    INDEXING_ERROR = "indexing_error"

    # Generic
    UNKNOWN = "unknown"


@dataclass
class ArchonError(Exception):
    """
    Base exception for all Archon errors.

    Provides structured error information for logging,
    monitoring, and user feedback.
    """

    message: str
    category: ErrorCategory = ErrorCategory.UNKNOWN
    recoverable: bool = True
    retry_after: Optional[int] = None  # seconds
    details: dict[str, Any] = field(default_factory=dict)
    original_exception: Optional[Exception] = None

    def __post_init__(self):
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for logging/API responses."""
        return {
            "error": self.message,
            "category": self.category.value,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "details": self.details,
        }

    def __str__(self) -> str:
        return f"[{self.category.value}] {self.message}"


# =============================================================================
# Specific Error Classes
# =============================================================================

class RateLimitError(ArchonError):
    """Raised when API rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int = 60,
        **kwargs
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.RATE_LIMIT,
            recoverable=True,
            retry_after=retry_after,
            **kwargs
        )


class TokenLimitError(ArchonError):
    """Raised when token limit is exceeded."""

    def __init__(
        self,
        message: str = "Token limit exceeded",
        tokens_used: Optional[int] = None,
        token_limit: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.pop("details", {})
        if tokens_used:
            details["tokens_used"] = tokens_used
        if token_limit:
            details["token_limit"] = token_limit

        super().__init__(
            message=message,
            category=ErrorCategory.TOKEN_LIMIT,
            recoverable=False,
            details=details,
            **kwargs
        )


class ModelError(ArchonError):
    """Raised when LLM returns an error."""

    def __init__(
        self,
        message: str = "Model error",
        model: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop("details", {})
        if model:
            details["model"] = model

        super().__init__(
            message=message,
            category=ErrorCategory.MODEL_ERROR,
            recoverable=True,
            details=details,
            **kwargs
        )


class ValidationError(ArchonError):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str = "Validation error",
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ):
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)[:100]  # Truncate for safety

        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION_ERROR,
            recoverable=False,
            details=details,
            **kwargs
        )


class PromptInjectionError(ArchonError):
    """Raised when prompt injection is detected."""

    def __init__(
        self,
        message: str = "Potential prompt injection detected",
        **kwargs
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.PROMPT_INJECTION,
            recoverable=False,
            **kwargs
        )


class ToolError(ArchonError):
    """Raised when a tool execution fails."""

    def __init__(
        self,
        message: str = "Tool execution failed",
        tool_name: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop("details", {})
        if tool_name:
            details["tool_name"] = tool_name

        super().__init__(
            message=message,
            category=ErrorCategory.TOOL_ERROR,
            recoverable=True,
            details=details,
            **kwargs
        )


class RetrievalError(ArchonError):
    """Raised when RAG retrieval fails."""

    def __init__(
        self,
        message: str = "Retrieval failed",
        query: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop("details", {})
        if query:
            details["query"] = query[:100]  # Truncate

        super().__init__(
            message=message,
            category=ErrorCategory.RETRIEVAL_ERROR,
            recoverable=True,
            details=details,
            **kwargs
        )


class AuthenticationError(ArchonError):
    """Raised when authentication fails."""

    def __init__(
        self,
        message: str = "Authentication failed",
        **kwargs
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHENTICATION_ERROR,
            recoverable=False,
            **kwargs
        )


class AuthorizationError(ArchonError):
    """Raised when authorization fails."""

    def __init__(
        self,
        message: str = "Not authorized",
        required_permission: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop("details", {})
        if required_permission:
            details["required_permission"] = required_permission

        super().__init__(
            message=message,
            category=ErrorCategory.AUTHORIZATION_ERROR,
            recoverable=False,
            details=details,
            **kwargs
        )


# =============================================================================
# Error Classification Utility
# =============================================================================

def classify_exception(exc: Exception) -> ArchonError:
    """
    Classify a generic exception into an ArchonError.

    Useful for converting third-party exceptions into
    our structured error format.

    Args:
        exc: The exception to classify

    Returns:
        ArchonError with appropriate category
    """
    if isinstance(exc, ArchonError):
        return exc

    error_str = str(exc).lower()
    exc_type = type(exc).__name__.lower()

    # Rate limiting
    if "rate limit" in error_str or "429" in error_str or "too many requests" in error_str:
        return RateLimitError(
            message=str(exc),
            original_exception=exc
        )

    # Token limits
    if "token" in error_str and ("limit" in error_str or "exceeded" in error_str):
        return TokenLimitError(
            message=str(exc),
            original_exception=exc
        )

    # Timeout
    if "timeout" in error_str or "timed out" in error_str:
        return ArchonError(
            message=str(exc),
            category=ErrorCategory.TIMEOUT,
            recoverable=True,
            original_exception=exc
        )

    # Network errors
    if "connection" in error_str or "network" in error_str or "dns" in error_str:
        return ArchonError(
            message=str(exc),
            category=ErrorCategory.NETWORK_ERROR,
            recoverable=True,
            original_exception=exc
        )

    # Validation errors
    if "validation" in error_str or "invalid" in error_str:
        return ValidationError(
            message=str(exc),
            original_exception=exc
        )

    # Authentication
    if "authentication" in error_str or "401" in error_str or "api key" in error_str:
        return AuthenticationError(
            message=str(exc),
            original_exception=exc
        )

    # Default: unknown error
    return ArchonError(
        message=str(exc),
        category=ErrorCategory.UNKNOWN,
        recoverable=False,
        original_exception=exc,
        details={"traceback": traceback.format_exc()}
    )
