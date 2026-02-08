"""
Archon Structured Logging Module

Production-grade logging with structured JSON output,
request context propagation, and LLM-specific fields.
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Optional
from dataclasses import dataclass, asdict, field
from uuid import uuid4

from src.core.config import settings


# =============================================================================
# Context Variables (for distributed tracing)
# =============================================================================

request_id_var: ContextVar[str] = ContextVar("request_id", default="")
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")
session_id_var: ContextVar[str] = ContextVar("session_id", default="")


def set_request_context(
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict[str, str]:
    """
    Set request context for logging.

    Returns the context dict for convenience.
    """
    rid = request_id or str(uuid4())
    tid = trace_id or str(uuid4())

    request_id_var.set(rid)
    trace_id_var.set(tid)

    if user_id:
        user_id_var.set(user_id)
    if session_id:
        session_id_var.set(session_id)

    return {
        "request_id": rid,
        "trace_id": tid,
        "user_id": user_id or "",
        "session_id": session_id or "",
    }


def clear_request_context():
    """Clear request context after request completes."""
    request_id_var.set("")
    trace_id_var.set("")
    user_id_var.set("")
    session_id_var.set("")


# =============================================================================
# Log Entry Schema
# =============================================================================

@dataclass
class LogEntry:
    """Structured log entry with all relevant fields."""

    # Required fields
    timestamp: str
    level: str
    message: str
    component: str

    # Context fields
    request_id: str = ""
    trace_id: str = ""
    user_id: str = ""
    session_id: str = ""

    # LLM-specific fields (optional)
    agent_id: Optional[str] = None
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None

    # Tool-specific fields (optional)
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[str] = None

    # Error fields (optional)
    error: Optional[str] = None
    error_category: Optional[str] = None
    error_traceback: Optional[str] = None

    # Additional metadata
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None and v != "" and v != {}}

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


# =============================================================================
# JSON Formatter
# =============================================================================

class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs JSON log lines."""

    def format(self, record: logging.LogRecord) -> str:
        # Get extra fields if present
        extra = getattr(record, "extra", {})

        entry = LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=record.levelname,
            message=record.getMessage(),
            component=record.name,
            request_id=request_id_var.get(),
            trace_id=trace_id_var.get(),
            user_id=user_id_var.get(),
            session_id=session_id_var.get(),
            **extra
        )

        # Add exception info if present
        if record.exc_info:
            import traceback
            entry.error_traceback = "".join(traceback.format_exception(*record.exc_info))

        return entry.to_json()


# =============================================================================
# Archon Logger
# =============================================================================

class ArchonLogger:
    """
    Production-grade logger for AI agent systems.

    Features:
    - Structured JSON output
    - Request context propagation
    - LLM-specific logging (tokens, latency, cost)
    - Tool execution logging
    - Error categorization
    """

    def __init__(self, component: str):
        """
        Initialize logger for a component.

        Args:
            component: Name of the component (e.g., "agents.planner", "rag.retriever")
        """
        self.component = component
        self.logger = logging.getLogger(component)

        # Only configure if not already configured
        if not self.logger.handlers:
            self._configure()

    def _configure(self):
        """Configure the logger with JSON formatter."""
        self.logger.setLevel(getattr(logging, settings.log_level))

        # Console handler with JSON output
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)

        # Prevent propagation to root logger
        self.logger.propagate = False

    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method with extra fields."""
        # Store extra fields in a way the formatter can access
        extra = {"extra": kwargs}
        self.logger.log(level, message, extra=extra)

    # -------------------------------------------------------------------------
    # Standard logging methods
    # -------------------------------------------------------------------------

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)

    # -------------------------------------------------------------------------
    # LLM-specific logging
    # -------------------------------------------------------------------------

    def log_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        cost_usd: float,
        success: bool = True,
        error: Optional[str] = None,
        agent_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Log an LLM API call with all relevant metrics.

        Args:
            model: Model name (e.g., "gpt-4o-mini")
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            latency_ms: Request latency in milliseconds
            cost_usd: Estimated cost in USD
            success: Whether the call succeeded
            error: Error message if failed
            agent_id: Agent that made the call
            metadata: Additional metadata
        """
        level = logging.INFO if success else logging.ERROR
        message = f"LLM call {'completed' if success else 'failed'}: {model}"

        self._log(
            level,
            message,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            agent_id=agent_id,
            error=error,
            metadata=metadata or {},
        )

    # -------------------------------------------------------------------------
    # Tool execution logging
    # -------------------------------------------------------------------------

    def log_tool_call(
        self,
        tool_name: str,
        latency_ms: float,
        success: bool = True,
        tool_input: Optional[dict] = None,
        tool_output: Optional[str] = None,
        error: Optional[str] = None,
        agent_id: Optional[str] = None,
    ):
        """
        Log a tool execution.

        Args:
            tool_name: Name of the tool
            latency_ms: Execution time in milliseconds
            success: Whether the tool succeeded
            tool_input: Input parameters (will be truncated)
            tool_output: Output (will be truncated)
            error: Error message if failed
            agent_id: Agent that called the tool
        """
        level = logging.INFO if success else logging.ERROR
        message = f"Tool {'executed' if success else 'failed'}: {tool_name}"

        # Truncate large inputs/outputs
        if tool_input:
            tool_input = {k: str(v)[:500] for k, v in tool_input.items()}
        if tool_output and len(tool_output) > 500:
            tool_output = tool_output[:500] + "..."

        self._log(
            level,
            message,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            latency_ms=latency_ms,
            error=error,
            agent_id=agent_id,
        )

    # -------------------------------------------------------------------------
    # Agent logging
    # -------------------------------------------------------------------------

    def log_agent_step(
        self,
        agent_id: str,
        step: str,
        thought: Optional[str] = None,
        action: Optional[str] = None,
        observation: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Log an agent reasoning step (ReAct pattern).

        Args:
            agent_id: Agent identifier
            step: Step number or name
            thought: Agent's reasoning
            action: Action taken
            observation: Result observed
            metadata: Additional context
        """
        self.info(
            f"Agent step: {agent_id} - {step}",
            agent_id=agent_id,
            metadata={
                "step": step,
                "thought": thought[:500] if thought else None,
                "action": action,
                "observation": observation[:500] if observation else None,
                **(metadata or {}),
            },
        )


# =============================================================================
# Logger Factory
# =============================================================================

def get_logger(component: str) -> ArchonLogger:
    """
    Get a logger for a component.

    Args:
        component: Component name (e.g., "agents.planner")

    Returns:
        ArchonLogger instance
    """
    return ArchonLogger(component)


# =============================================================================
# Convenience loggers for common components
# =============================================================================

# Pre-configured loggers
api_logger = get_logger("api")
agent_logger = get_logger("agents")
rag_logger = get_logger("rag")
tool_logger = get_logger("tools")
security_logger = get_logger("security")
