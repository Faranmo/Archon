"""
Tests for core module.
"""

import pytest
from src.core.errors import (
    ArchonError,
    ErrorCategory,
    RateLimitError,
    TokenLimitError,
    ValidationError,
    classify_exception,
)
from src.core.types import (
    Message,
    MessageRole,
    Document,
    DocumentType,
    Chunk,
)
from src.core.retry import (
    RetryConfig,
    calculate_delay,
    should_retry,
)


class TestErrors:
    """Test error handling."""

    def test_archon_error_basic(self):
        """Test basic ArchonError creation."""
        error = ArchonError(
            message="Test error",
            category=ErrorCategory.UNKNOWN,
        )
        assert str(error) == "[unknown] Test error"
        assert error.recoverable is True

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError(retry_after=30)
        assert error.category == ErrorCategory.RATE_LIMIT
        assert error.retry_after == 30
        assert error.recoverable is True

    def test_token_limit_error(self):
        """Test TokenLimitError."""
        error = TokenLimitError(tokens_used=10000, token_limit=8192)
        assert error.category == ErrorCategory.TOKEN_LIMIT
        assert error.recoverable is False
        assert error.details["tokens_used"] == 10000

    def test_classify_exception_rate_limit(self):
        """Test exception classification for rate limits."""
        exc = Exception("Rate limit exceeded")
        classified = classify_exception(exc)
        assert classified.category == ErrorCategory.RATE_LIMIT

    def test_classify_exception_timeout(self):
        """Test exception classification for timeouts."""
        exc = Exception("Connection timed out")
        classified = classify_exception(exc)
        assert classified.category == ErrorCategory.TIMEOUT


class TestTypes:
    """Test type definitions."""

    def test_message_creation(self):
        """Test Message creation."""
        msg = Message(
            role=MessageRole.USER,
            content="Hello",
        )
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"

    def test_message_to_dict(self):
        """Test Message to_dict."""
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Hi there",
        )
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "Hi there"

    def test_document_properties(self):
        """Test Document properties."""
        doc = Document(
            content="Hello world test",
            doc_type=DocumentType.TXT,
        )
        assert doc.char_count == 16
        assert doc.word_count == 3

    def test_chunk_creation(self):
        """Test Chunk creation."""
        chunk = Chunk(
            content="Test chunk content",
            document_id="doc-1",
            chunk_index=0,
        )
        assert chunk.content == "Test chunk content"
        assert chunk.embedding is None


class TestRetry:
    """Test retry logic."""

    def test_calculate_delay_basic(self):
        """Test basic delay calculation."""
        config = RetryConfig(base_delay=1.0, jitter=False)
        delay = calculate_delay(0, config)
        assert delay == 1.0

    def test_calculate_delay_exponential(self):
        """Test exponential backoff."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        delay = calculate_delay(2, config)
        assert delay == 4.0  # 1 * 2^2

    def test_calculate_delay_max(self):
        """Test max delay cap."""
        config = RetryConfig(base_delay=1.0, max_delay=10.0, jitter=False)
        delay = calculate_delay(10, config)
        assert delay == 10.0

    def test_should_retry_max_attempts(self):
        """Test retry limit."""
        config = RetryConfig(max_retries=3)
        error = RateLimitError()
        assert should_retry(error, 2, config) is True
        assert should_retry(error, 3, config) is False

    def test_should_retry_non_recoverable(self):
        """Test non-recoverable error."""
        config = RetryConfig()
        error = ValidationError(message="Invalid input")
        assert should_retry(error, 0, config) is False
