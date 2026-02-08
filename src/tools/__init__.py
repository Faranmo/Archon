"""
Archon Tools Module

Deterministic and LLM-invoked tools following arXiv 2512.08769 best practices.
Tools prioritize deterministic operations over LLM calls where possible.
"""

from src.tools.base import (
    Tool,
    ToolResult,
    tool,
    get_tool_registry,
)
from src.tools.executor import (
    ToolExecutor,
    execute_tool,
)
from src.tools.search import (
    web_search,
    semantic_search,
)
from src.tools.document import (
    read_document,
    extract_text,
)

__all__ = [
    "Tool",
    "ToolResult",
    "tool",
    "get_tool_registry",
    "ToolExecutor",
    "execute_tool",
    "web_search",
    "semantic_search",
    "read_document",
    "extract_text",
]
