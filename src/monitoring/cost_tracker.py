"""
Archon Cost Tracking Module (FinOps)

Tracks token usage, costs, and budgets for LLM API calls.
Essential for production cost management.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict
import threading

from src.core.config import settings
from src.monitoring.logger import get_logger

logger = get_logger("monitoring.cost")


# =============================================================================
# Model Pricing (as of 2024)
# =============================================================================

@dataclass
class ModelPricing:
    """Pricing per 1K tokens."""
    input_per_1k: float
    output_per_1k: float
    cached_input_per_1k: Optional[float] = None  # For cached prompts


# Pricing data - update as needed
MODEL_PRICING: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o": ModelPricing(0.005, 0.015),
    "gpt-4o-mini": ModelPricing(0.00015, 0.0006),
    "gpt-4-turbo": ModelPricing(0.01, 0.03),
    "gpt-4": ModelPricing(0.03, 0.06),
    "gpt-3.5-turbo": ModelPricing(0.0005, 0.0015),

    # OpenAI Embeddings
    "text-embedding-3-small": ModelPricing(0.00002, 0.0),
    "text-embedding-3-large": ModelPricing(0.00013, 0.0),
    "text-embedding-ada-002": ModelPricing(0.0001, 0.0),

    # Anthropic Claude
    "claude-3-opus-20240229": ModelPricing(0.015, 0.075),
    "claude-3-sonnet-20240229": ModelPricing(0.003, 0.015),
    "claude-3-haiku-20240307": ModelPricing(0.00025, 0.00125),
    "claude-3-5-sonnet-20240620": ModelPricing(0.003, 0.015),

    # Default fallback
    "default": ModelPricing(0.01, 0.03),
}


def get_model_pricing(model: str) -> ModelPricing:
    """Get pricing for a model, with fallback to default."""
    # Try exact match
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    # Try partial match (for versioned models)
    for key, pricing in MODEL_PRICING.items():
        if key in model or model in key:
            return pricing

    return MODEL_PRICING["default"]


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """
    Calculate cost for an LLM call.

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cached_tokens: Number of cached input tokens (if applicable)

    Returns:
        Cost in USD
    """
    pricing = get_model_pricing(model)

    # Calculate regular input cost
    regular_input = input_tokens - cached_tokens
    input_cost = (regular_input / 1000) * pricing.input_per_1k

    # Calculate cached input cost
    if cached_tokens > 0 and pricing.cached_input_per_1k:
        input_cost += (cached_tokens / 1000) * pricing.cached_input_per_1k

    # Calculate output cost
    output_cost = (output_tokens / 1000) * pricing.output_per_1k

    return input_cost + output_cost


# =============================================================================
# Usage Record
# =============================================================================

@dataclass
class UsageRecord:
    """Record of a single LLM usage."""
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    endpoint: Optional[str] = None


# =============================================================================
# Cost Tracker
# =============================================================================

class CostTracker:
    """
    Tracks LLM usage and costs.

    Features:
    - Real-time cost tracking
    - Budget limits and alerts
    - Per-model, per-user, per-agent breakdowns
    - Daily/hourly aggregations
    """

    def __init__(
        self,
        daily_budget_usd: float = settings.daily_budget_usd,
        alert_threshold_percent: int = settings.alert_threshold_percent,
    ):
        self.daily_budget_usd = daily_budget_usd
        self.alert_threshold_percent = alert_threshold_percent

        # Thread-safe storage
        self._lock = threading.Lock()
        self._records: list[UsageRecord] = []

        # Aggregations
        self._total_cost: float = 0.0
        self._cost_by_model: dict[str, float] = defaultdict(float)
        self._cost_by_user: dict[str, float] = defaultdict(float)
        self._cost_by_agent: dict[str, float] = defaultdict(float)
        self._tokens_by_model: dict[str, dict[str, int]] = defaultdict(
            lambda: {"input": 0, "output": 0}
        )

        # Budget tracking
        self._daily_cost: float = 0.0
        self._last_reset: datetime = datetime.utcnow()
        self._alert_sent: bool = False

    def _maybe_reset_daily(self):
        """Reset daily tracking if new day."""
        now = datetime.utcnow()
        if now.date() > self._last_reset.date():
            self._daily_cost = 0.0
            self._last_reset = now
            self._alert_sent = False
            logger.info("Daily cost tracking reset")

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> float:
        """
        Record LLM usage and return the cost.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            request_id: Request identifier
            user_id: User identifier
            agent_id: Agent identifier
            endpoint: API endpoint

        Returns:
            Cost in USD
        """
        cost = calculate_cost(model, input_tokens, output_tokens)

        record = UsageRecord(
            timestamp=datetime.utcnow(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            request_id=request_id,
            user_id=user_id,
            agent_id=agent_id,
            endpoint=endpoint,
        )

        with self._lock:
            self._maybe_reset_daily()

            self._records.append(record)
            self._total_cost += cost
            self._daily_cost += cost
            self._cost_by_model[model] += cost

            if user_id:
                self._cost_by_user[user_id] += cost
            if agent_id:
                self._cost_by_agent[agent_id] += cost

            self._tokens_by_model[model]["input"] += input_tokens
            self._tokens_by_model[model]["output"] += output_tokens

            # Check budget
            self._check_budget()

        logger.debug(
            f"Recorded usage: {model}",
            metadata={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "daily_total": self._daily_cost,
            }
        )

        return cost

    def _check_budget(self):
        """Check if budget threshold is reached."""
        if self._alert_sent:
            return

        percent_used = (self._daily_cost / self.daily_budget_usd) * 100

        if percent_used >= self.alert_threshold_percent:
            logger.warning(
                f"Budget alert: {percent_used:.1f}% of daily budget used",
                metadata={
                    "daily_cost": self._daily_cost,
                    "daily_budget": self.daily_budget_usd,
                    "percent_used": percent_used,
                }
            )
            self._alert_sent = True

        if self._daily_cost >= self.daily_budget_usd:
            logger.error(
                f"Daily budget exceeded: ${self._daily_cost:.4f} / ${self.daily_budget_usd:.2f}",
                metadata={
                    "daily_cost": self._daily_cost,
                    "daily_budget": self.daily_budget_usd,
                }
            )

    def is_budget_exceeded(self) -> bool:
        """Check if daily budget is exceeded."""
        with self._lock:
            self._maybe_reset_daily()
            return self._daily_cost >= self.daily_budget_usd

    def get_remaining_budget(self) -> float:
        """Get remaining daily budget."""
        with self._lock:
            self._maybe_reset_daily()
            return max(0, self.daily_budget_usd - self._daily_cost)

    def get_report(self, hours: int = 24) -> dict:
        """
        Get cost report for the specified time period.

        Args:
            hours: Number of hours to include

        Returns:
            Report dictionary
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        with self._lock:
            recent_records = [r for r in self._records if r.timestamp > cutoff]

            total_cost = sum(r.cost_usd for r in recent_records)
            total_input = sum(r.input_tokens for r in recent_records)
            total_output = sum(r.output_tokens for r in recent_records)

            cost_by_model: dict[str, float] = defaultdict(float)
            cost_by_user: dict[str, float] = defaultdict(float)
            cost_by_agent: dict[str, float] = defaultdict(float)

            for r in recent_records:
                cost_by_model[r.model] += r.cost_usd
                if r.user_id:
                    cost_by_user[r.user_id] += r.cost_usd
                if r.agent_id:
                    cost_by_agent[r.agent_id] += r.cost_usd

        return {
            "period_hours": hours,
            "total_requests": len(recent_records),
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "avg_cost_per_request": round(total_cost / max(len(recent_records), 1), 6),
            "cost_by_model": {k: round(v, 6) for k, v in cost_by_model.items()},
            "cost_by_user": {k: round(v, 6) for k, v in cost_by_user.items()},
            "cost_by_agent": {k: round(v, 6) for k, v in cost_by_agent.items()},
            "budget": {
                "daily_limit_usd": self.daily_budget_usd,
                "daily_spent_usd": round(self._daily_cost, 6),
                "daily_remaining_usd": round(self.get_remaining_budget(), 6),
                "percent_used": round((self._daily_cost / self.daily_budget_usd) * 100, 1),
            },
        }


# =============================================================================
# Global Instance
# =============================================================================

# Singleton cost tracker
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


# Convenience function
def track_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    **kwargs
) -> float:
    """Track LLM usage and return cost."""
    return get_cost_tracker().record_usage(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        **kwargs
    )
