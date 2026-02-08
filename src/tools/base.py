"""
Archon Tool Base Module

Base classes and decorators for tool definitions.
Following arXiv 2512.08769: "Code blocks are preferable to LLM decisions."
"""

import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, Type, TypeVar, Union
from pydantic import BaseModel, Field

from src.core.types import ToolResult as CoreToolResult
from src.core.errors import ToolError
from src.monitoring.logger import get_logger

logger = get_logger("tools.base")

T = TypeVar("T")


# =============================================================================
# Tool Result
# =============================================================================

@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }

    def to_string(self) -> str:
        """Convert to string for LLM consumption."""
        if self.success:
            if isinstance(self.data, (dict, list)):
                return json.dumps(self.data, indent=2)
            return str(self.data)
        return f"Error: {self.error}"


# =============================================================================
# Tool Parameter Schema
# =============================================================================

class ToolParameter(BaseModel):
    """Base model for tool parameters."""
    pass


# =============================================================================
# Tool Definition
# =============================================================================

@dataclass
class Tool:
    """
    Tool definition with metadata and execution.

    Follows OpenAI function calling format for compatibility.
    """
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    function: Callable
    is_async: bool = False
    requires_confirmation: bool = False
    category: str = "general"
    examples: list[dict] = field(default_factory=list)

    def to_openai_format(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def to_anthropic_format(self) -> dict:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool."""
        start_time = time.time()

        try:
            if self.is_async:
                result = await self.function(**kwargs)
            else:
                result = self.function(**kwargs)

            latency_ms = (time.time() - start_time) * 1000

            # Wrap raw result in ToolResult if needed
            if isinstance(result, ToolResult):
                result.latency_ms = latency_ms
                return result

            return ToolResult(
                success=True,
                data=result,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Tool execution failed: {self.name}",
                metadata={"error": str(e), "kwargs": str(kwargs)[:200]}
            )
            return ToolResult(
                success=False,
                error=str(e),
                latency_ms=latency_ms,
            )


# =============================================================================
# Tool Registry
# =============================================================================

class ToolRegistry:
    """Registry for all available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> list[Tool]:
        """List all tools, optionally filtered by category."""
        if category:
            return [t for t in self._tools.values() if t.category == category]
        return list(self._tools.values())

    def get_openai_tools(self, names: Optional[list[str]] = None) -> list[dict]:
        """Get tools in OpenAI format."""
        tools = self._tools.values()
        if names:
            tools = [t for t in tools if t.name in names]
        return [t.to_openai_format() for t in tools]

    def get_anthropic_tools(self, names: Optional[list[str]] = None) -> list[dict]:
        """Get tools in Anthropic format."""
        tools = self._tools.values()
        if names:
            tools = [t for t in tools if t.name in names]
        return [t.to_anthropic_format() for t in tools]


# Global registry
_tool_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _tool_registry


# =============================================================================
# Tool Decorator
# =============================================================================

def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[dict] = None,
    category: str = "general",
    requires_confirmation: bool = False,
    register: bool = True,
):
    """
    Decorator to define a tool.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        parameters: JSON Schema for parameters (auto-generated if not provided)
        category: Tool category
        requires_confirmation: Whether tool needs human approval
        register: Whether to auto-register in global registry

    Example:
        @tool(description="Search the web for information")
        def web_search(query: str, num_results: int = 5) -> list[dict]:
            ...
    """
    def decorator(func: Callable) -> Callable:
        # Determine if async
        import asyncio
        is_async = asyncio.iscoroutinefunction(func)

        # Get tool name
        tool_name = name or func.__name__

        # Get description from docstring if not provided
        tool_description = description or (func.__doc__ or "").strip().split("\n")[0]

        # Auto-generate parameters schema from function signature
        tool_parameters = parameters
        if tool_parameters is None:
            import inspect
            sig = inspect.signature(func)
            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue

                param_type = "string"  # Default
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == list:
                        param_type = "array"
                    elif param.annotation == dict:
                        param_type = "object"

                properties[param_name] = {"type": param_type}

                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

            tool_parameters = {
                "type": "object",
                "properties": properties,
                "required": required,
            }

        # Create Tool instance
        tool_instance = Tool(
            name=tool_name,
            description=tool_description,
            parameters=tool_parameters,
            function=func,
            is_async=is_async,
            requires_confirmation=requires_confirmation,
            category=category,
        )

        # Register if requested
        if register:
            get_tool_registry().register(tool_instance)

        # Attach tool metadata to function
        func._tool = tool_instance

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await tool_instance.execute(**kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            import asyncio
            return asyncio.get_event_loop().run_until_complete(
                tool_instance.execute(**kwargs)
            )

        wrapper = async_wrapper if is_async else sync_wrapper
        wrapper._tool = tool_instance

        return wrapper

    return decorator


# =============================================================================
# Common Tool Schemas
# =============================================================================

# Reusable parameter schemas
STRING_PARAM = {"type": "string"}
INT_PARAM = {"type": "integer"}
BOOL_PARAM = {"type": "boolean"}
ARRAY_PARAM = {"type": "array", "items": {"type": "string"}}


def create_tool_schema(
    properties: dict[str, dict],
    required: Optional[list[str]] = None,
) -> dict:
    """
    Create a JSON Schema for tool parameters.

    Args:
        properties: Property definitions
        required: List of required property names

    Returns:
        JSON Schema dict
    """
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
    }
