"""
Archon LLM Client Module

Unified interface for multiple LLM providers.
Supports OpenAI, Anthropic, and model consortium patterns.
"""

from src.llm.client import (
    LLMClient,
    get_llm_client,
    complete,
    complete_with_tools,
)
from src.llm.providers import (
    OpenAIProvider,
    AnthropicProvider,
    get_provider,
)

__all__ = [
    "LLMClient",
    "get_llm_client",
    "complete",
    "complete_with_tools",
    "OpenAIProvider",
    "AnthropicProvider",
    "get_provider",
]
