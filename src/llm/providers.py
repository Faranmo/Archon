"""
Archon LLM Providers Module

Abstract provider interface and implementations for OpenAI, Anthropic.
Follows arXiv 2512.08769 model consortium pattern.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, AsyncIterator

from src.core.config import settings
from src.core.types import LLMResponse, ToolCall, Message, MessageRole
from src.core.errors import ModelError, RateLimitError, TokenLimitError
from src.core.retry import with_async_retry, RetryConfig, AGGRESSIVE_RETRY_CONFIG
from src.monitoring.logger import get_logger
from src.monitoring.cost_tracker import calculate_cost, track_usage

logger = get_logger("llm.providers")


# =============================================================================
# Provider Interface
# =============================================================================

@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60
    base_url: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._client: Any = None

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        **kwargs,
    ) -> LLMResponse:
        """Generate a completion."""
        pass

    @abstractmethod
    async def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[dict],
        **kwargs,
    ) -> LLMResponse:
        """Generate a completion with tool calling."""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a completion."""
        pass

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert Message objects to provider format."""
        return [m.to_dict() for m in messages]


# =============================================================================
# OpenAI Provider
# =============================================================================

class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT-4, GPT-4o, etc.)."""

    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig(
                api_key=settings.openai_api_key,
                model=settings.default_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
        super().__init__(config)
        self._init_client()

    def _init_client(self):
        """Initialize the OpenAI client."""
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    @with_async_retry(config=AGGRESSIVE_RETRY_CONFIG)
    async def complete(
        self,
        messages: list[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a completion using OpenAI."""
        model = model or self.config.model
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens

        start_time = time.time()

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=self._convert_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract usage
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens

            # Calculate cost
            cost = calculate_cost(model, prompt_tokens, completion_tokens)

            # Track usage
            track_usage(
                model=model,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
            )

            # Log the call
            logger.log_llm_call(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
            )

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
                finish_reason=response.choices[0].finish_reason,
            )

        except Exception as e:
            self._handle_error(e)
            raise

    @with_async_retry(config=AGGRESSIVE_RETRY_CONFIG)
    async def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: str = "auto",
        **kwargs,
    ) -> LLMResponse:
        """Generate a completion with tool calling."""
        model = model or self.config.model
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens

        start_time = time.time()

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=self._convert_messages(messages),
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract usage
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens

            # Calculate cost
            cost = calculate_cost(model, prompt_tokens, completion_tokens)

            # Track usage
            track_usage(
                model=model,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
            )

            # Parse tool calls
            tool_calls = []
            message = response.choices[0].message

            if message.tool_calls:
                import json
                for tc in message.tool_calls:
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    ))

            # Log the call
            logger.log_llm_call(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
                metadata={"tool_calls": len(tool_calls)},
            )

            return LLMResponse(
                content=message.content or "",
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
                tool_calls=tool_calls,
                finish_reason=response.choices[0].finish_reason,
            )

        except Exception as e:
            self._handle_error(e)
            raise

    async def stream(
        self,
        messages: list[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a completion."""
        model = model or self.config.model
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens

        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=self._convert_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            self._handle_error(e)
            raise

    def _handle_error(self, error: Exception):
        """Handle OpenAI-specific errors."""
        error_str = str(error).lower()

        if "rate limit" in error_str or "429" in error_str:
            raise RateLimitError(
                message=str(error),
                retry_after=60,
                original_exception=error,
            )
        elif "context length" in error_str or "token" in error_str:
            raise TokenLimitError(
                message=str(error),
                original_exception=error,
            )
        else:
            raise ModelError(
                message=str(error),
                model=self.config.model,
                original_exception=error,
            )


# =============================================================================
# Anthropic Provider
# =============================================================================

class AnthropicProvider(LLMProvider):
    """Anthropic API provider (Claude models)."""

    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig(
                api_key=settings.anthropic_api_key,
                model=settings.anthropic_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
        super().__init__(config)
        self._init_client()

    def _init_client(self):
        """Initialize the Anthropic client."""
        try:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    def _convert_messages(self, messages: list[Message]) -> tuple[Optional[str], list[dict]]:
        """Convert messages to Anthropic format (separate system prompt)."""
        system_prompt = None
        converted = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            else:
                converted.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })

        return system_prompt, converted

    @with_async_retry(config=AGGRESSIVE_RETRY_CONFIG)
    async def complete(
        self,
        messages: list[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a completion using Anthropic."""
        model = model or self.config.model
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens

        start_time = time.time()
        system_prompt, converted_messages = self._convert_messages(messages)

        try:
            response = await self._client.messages.create(
                model=model,
                messages=converted_messages,
                system=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract usage
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens

            # Calculate cost
            cost = calculate_cost(model, prompt_tokens, completion_tokens)

            # Track usage
            track_usage(
                model=model,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
            )

            # Get content
            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

            # Log the call
            logger.log_llm_call(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
            )

            return LLMResponse(
                content=content,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
                finish_reason=response.stop_reason or "stop",
            )

        except Exception as e:
            self._handle_error(e)
            raise

    @with_async_retry(config=AGGRESSIVE_RETRY_CONFIG)
    async def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a completion with tool calling."""
        model = model or self.config.model
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens

        start_time = time.time()
        system_prompt, converted_messages = self._convert_messages(messages)

        # Convert OpenAI tool format to Anthropic format
        anthropic_tools = []
        for tool in tools:
            if "function" in tool:
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })
            else:
                anthropic_tools.append(tool)

        try:
            response = await self._client.messages.create(
                model=model,
                messages=converted_messages,
                system=system_prompt,
                tools=anthropic_tools,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract usage
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens

            # Calculate cost
            cost = calculate_cost(model, prompt_tokens, completion_tokens)

            # Track usage
            track_usage(
                model=model,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
            )

            # Parse content and tool calls
            content = ""
            tool_calls = []

            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append(ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    ))

            # Log the call
            logger.log_llm_call(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
                metadata={"tool_calls": len(tool_calls)},
            )

            return LLMResponse(
                content=content,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
                tool_calls=tool_calls,
                finish_reason=response.stop_reason or "stop",
            )

        except Exception as e:
            self._handle_error(e)
            raise

    async def stream(
        self,
        messages: list[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a completion."""
        model = model or self.config.model
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens

        system_prompt, converted_messages = self._convert_messages(messages)

        try:
            async with self._client.messages.stream(
                model=model,
                messages=converted_messages,
                system=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            self._handle_error(e)
            raise

    def _handle_error(self, error: Exception):
        """Handle Anthropic-specific errors."""
        error_str = str(error).lower()

        if "rate limit" in error_str or "429" in error_str:
            raise RateLimitError(
                message=str(error),
                retry_after=60,
                original_exception=error,
            )
        elif "token" in error_str and "limit" in error_str:
            raise TokenLimitError(
                message=str(error),
                original_exception=error,
            )
        else:
            raise ModelError(
                message=str(error),
                model=self.config.model,
                original_exception=error,
            )


# =============================================================================
# Provider Factory
# =============================================================================

_providers: dict[str, LLMProvider] = {}


def get_provider(provider_name: str = "openai") -> LLMProvider:
    """
    Get an LLM provider by name.

    Args:
        provider_name: "openai" or "anthropic"

    Returns:
        LLMProvider instance
    """
    if provider_name not in _providers:
        if provider_name == "openai":
            _providers[provider_name] = OpenAIProvider()
        elif provider_name == "anthropic":
            _providers[provider_name] = AnthropicProvider()
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

    return _providers[provider_name]
