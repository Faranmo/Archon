"""
Archon LLM Client Module

High-level client for LLM interactions.
Implements model consortium and fallback patterns.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional, AsyncIterator
from enum import Enum

from src.core.config import settings
from src.core.types import LLMResponse, Message, MessageRole
from src.core.errors import ModelError
from src.core.retry import CircuitBreaker, CircuitOpenError
from src.llm.providers import LLMProvider, get_provider, OpenAIProvider, AnthropicProvider
from src.monitoring.logger import get_logger

logger = get_logger("llm.client")


# =============================================================================
# Model Selection Strategy
# =============================================================================

class ModelStrategy(Enum):
    """Strategy for model selection in consortium."""
    FASTEST = "fastest"       # Use fastest available model
    CHEAPEST = "cheapest"     # Use cheapest model
    BEST = "best"            # Use highest quality model
    FALLBACK = "fallback"    # Use fallback chain
    CONSENSUS = "consensus"  # Query multiple, use consensus


@dataclass
class ModelConfig:
    """Configuration for a model in the consortium."""
    provider: str  # "openai" or "anthropic"
    model: str
    priority: int = 0  # Lower = higher priority
    cost_per_1k_tokens: float = 0.01
    latency_ms_estimate: float = 1000
    quality_score: float = 0.8  # 0-1 quality estimate


# Default model consortium
DEFAULT_MODELS: list[ModelConfig] = [
    ModelConfig(
        provider="openai",
        model="gpt-4o",
        priority=0,
        cost_per_1k_tokens=0.01,
        latency_ms_estimate=800,
        quality_score=0.95,
    ),
    ModelConfig(
        provider="openai",
        model="gpt-4o-mini",
        priority=1,
        cost_per_1k_tokens=0.0003,
        latency_ms_estimate=500,
        quality_score=0.85,
    ),
    ModelConfig(
        provider="anthropic",
        model="claude-3-5-sonnet-20240620",
        priority=2,
        cost_per_1k_tokens=0.009,
        latency_ms_estimate=900,
        quality_score=0.93,
    ),
    ModelConfig(
        provider="anthropic",
        model="claude-3-haiku-20240307",
        priority=3,
        cost_per_1k_tokens=0.00175,
        latency_ms_estimate=400,
        quality_score=0.80,
    ),
]


# =============================================================================
# LLM Client
# =============================================================================

class LLMClient:
    """
    High-level LLM client with consortium support.

    Features:
    - Multiple provider support (OpenAI, Anthropic)
    - Model fallback chains
    - Circuit breakers per provider
    - Automatic retries
    - Cost optimization
    """

    def __init__(
        self,
        models: Optional[list[ModelConfig]] = None,
        strategy: ModelStrategy = ModelStrategy.FALLBACK,
    ):
        self.models = models or DEFAULT_MODELS
        self.strategy = strategy

        # Circuit breakers per provider
        self._circuit_breakers: dict[str, CircuitBreaker] = {
            "openai": CircuitBreaker(name="openai", failure_threshold=3),
            "anthropic": CircuitBreaker(name="anthropic", failure_threshold=3),
        }

        # Provider instances
        self._providers: dict[str, LLMProvider] = {}

    def _get_provider(self, provider_name: str) -> LLMProvider:
        """Get or create a provider instance."""
        if provider_name not in self._providers:
            self._providers[provider_name] = get_provider(provider_name)
        return self._providers[provider_name]

    def _select_models(self) -> list[ModelConfig]:
        """Select models based on strategy."""
        models = self.models.copy()

        if self.strategy == ModelStrategy.FASTEST:
            models.sort(key=lambda m: m.latency_ms_estimate)
        elif self.strategy == ModelStrategy.CHEAPEST:
            models.sort(key=lambda m: m.cost_per_1k_tokens)
        elif self.strategy == ModelStrategy.BEST:
            models.sort(key=lambda m: -m.quality_score)
        else:  # FALLBACK or default
            models.sort(key=lambda m: m.priority)

        return models

    async def complete(
        self,
        messages: list[Message],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a completion.

        If model/provider specified, use those directly.
        Otherwise, use the configured strategy.
        """
        # Direct model specification
        if model and provider:
            prov = self._get_provider(provider)
            return await prov.complete(messages, model=model, **kwargs)

        # Use strategy
        models = self._select_models()
        last_error: Optional[Exception] = None

        for model_config in models:
            # Check circuit breaker
            breaker = self._circuit_breakers.get(model_config.provider)
            if breaker and breaker.state.value == "open":
                logger.debug(f"Skipping {model_config.provider} - circuit open")
                continue

            try:
                prov = self._get_provider(model_config.provider)

                # Use circuit breaker
                if breaker:
                    response = await breaker.call_async(
                        lambda: prov.complete(messages, model=model_config.model, **kwargs)
                    )
                else:
                    response = await prov.complete(messages, model=model_config.model, **kwargs)

                logger.info(
                    f"Completed with {model_config.provider}/{model_config.model}",
                    metadata={
                        "model": model_config.model,
                        "latency_ms": response.latency_ms,
                        "tokens": response.total_tokens,
                    }
                )

                return response

            except CircuitOpenError:
                logger.warning(f"Circuit open for {model_config.provider}")
                continue

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Model {model_config.model} failed, trying next",
                    metadata={
                        "model": model_config.model,
                        "error": str(e),
                    }
                )
                continue

        # All models failed
        raise ModelError(
            message=f"All models failed. Last error: {last_error}",
            original_exception=last_error,
        )

    async def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[dict],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a completion with tool calling."""
        # Direct model specification
        if model and provider:
            prov = self._get_provider(provider)
            return await prov.complete_with_tools(messages, tools, model=model, **kwargs)

        # Use strategy (prioritize models with good tool support)
        models = self._select_models()
        last_error: Optional[Exception] = None

        for model_config in models:
            breaker = self._circuit_breakers.get(model_config.provider)
            if breaker and breaker.state.value == "open":
                continue

            try:
                prov = self._get_provider(model_config.provider)

                if breaker:
                    response = await breaker.call_async(
                        lambda: prov.complete_with_tools(
                            messages, tools, model=model_config.model, **kwargs
                        )
                    )
                else:
                    response = await prov.complete_with_tools(
                        messages, tools, model=model_config.model, **kwargs
                    )

                return response

            except CircuitOpenError:
                continue

            except Exception as e:
                last_error = e
                continue

        raise ModelError(
            message=f"All models failed for tool calling. Last error: {last_error}",
            original_exception=last_error,
        )

    async def stream(
        self,
        messages: list[Message],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a completion."""
        # For streaming, we use the first available model
        if model and provider:
            prov = self._get_provider(provider)
            async for chunk in prov.stream(messages, model=model, **kwargs):
                yield chunk
            return

        models = self._select_models()

        for model_config in models:
            breaker = self._circuit_breakers.get(model_config.provider)
            if breaker and breaker.state.value == "open":
                continue

            try:
                prov = self._get_provider(model_config.provider)
                async for chunk in prov.stream(messages, model=model_config.model, **kwargs):
                    yield chunk
                return

            except Exception as e:
                logger.warning(f"Stream failed for {model_config.model}: {e}")
                continue

        raise ModelError(message="All models failed for streaming")

    async def consensus_complete(
        self,
        messages: list[Message],
        num_models: int = 2,
        **kwargs,
    ) -> tuple[LLMResponse, float]:
        """
        Query multiple models and return consensus.

        Returns:
            Tuple of (best_response, agreement_score)
        """
        models = self._select_models()[:num_models]

        # Query all models concurrently
        tasks = []
        for model_config in models:
            prov = self._get_provider(model_config.provider)
            task = prov.complete(messages, model=model_config.model, **kwargs)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful responses
        responses = [r for r in results if isinstance(r, LLMResponse)]

        if not responses:
            raise ModelError(message="All models failed in consensus query")

        # For now, return the response from highest quality model
        # In a real implementation, you'd compare outputs for agreement
        best_response = responses[0]

        # Simple agreement score based on response similarity
        # (placeholder - real implementation would use semantic similarity)
        agreement_score = 1.0 if len(responses) == 1 else 0.8

        return best_response, agreement_score


# =============================================================================
# Global Instance & Convenience Functions
# =============================================================================

_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get the global LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


async def complete(
    messages: list[Message],
    **kwargs,
) -> LLMResponse:
    """Convenience function for completion."""
    client = get_llm_client()
    return await client.complete(messages, **kwargs)


async def complete_with_tools(
    messages: list[Message],
    tools: list[dict],
    **kwargs,
) -> LLMResponse:
    """Convenience function for completion with tools."""
    client = get_llm_client()
    return await client.complete_with_tools(messages, tools, **kwargs)


# =============================================================================
# Helper Functions
# =============================================================================

def create_messages(
    user_message: str,
    system_message: Optional[str] = None,
    history: Optional[list[Message]] = None,
) -> list[Message]:
    """
    Create a message list for completion.

    Args:
        user_message: The user's input
        system_message: Optional system prompt
        history: Optional conversation history

    Returns:
        List of Message objects
    """
    messages = []

    if system_message:
        messages.append(Message(role=MessageRole.SYSTEM, content=system_message))

    if history:
        messages.extend(history)

    messages.append(Message(role=MessageRole.USER, content=user_message))

    return messages
