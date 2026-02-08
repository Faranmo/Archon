"""
Archon Test Configuration

Pytest fixtures and configuration.
"""

import pytest
import asyncio
from typing import Generator
from unittest.mock import MagicMock, AsyncMock

# Configure async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Mock LLM client
@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    client = MagicMock()
    client.complete = AsyncMock(return_value=MagicMock(
        content="Test response",
        model="gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        latency_ms=500,
        cost_usd=0.001,
        tool_calls=[],
        has_tool_calls=False,
    ))
    return client


# Mock retriever
@pytest.fixture
def mock_retriever():
    """Mock retriever for testing."""
    from src.core.types import Chunk, RetrievalResult
    
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=[
        RetrievalResult(
            chunk=Chunk(
                id="test-1",
                content="Test content",
                document_id="doc-1",
            ),
            score=0.9,
        )
    ])
    return retriever


# Test document
@pytest.fixture
def sample_document():
    """Sample document for testing."""
    from src.core.types import Document, DocumentType
    
    return Document(
        id="test-doc-1",
        content="""
# Test Document

This is a test document for unit testing.

## Section 1

Some content in section 1.

## Section 2

More content in section 2.
""",
        doc_type=DocumentType.MD,
        source="test.md",
    )


# Test message
@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    from src.core.types import Message, MessageRole
    
    return [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Hello, world!"),
    ]
