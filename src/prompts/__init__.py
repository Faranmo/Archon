"""
Archon Prompts Module

Externalized, versioned prompts following arXiv 2512.08769 best practices.
Prompts are stored as templates with variable substitution.
"""

from src.prompts.manager import (
    PromptManager,
    get_prompt_manager,
    get_prompt,
    render_prompt,
)
from src.prompts.templates import (
    AGENT_PROMPTS,
    SYSTEM_PROMPTS,
    TOOL_PROMPTS,
)

__all__ = [
    "PromptManager",
    "get_prompt_manager",
    "get_prompt",
    "render_prompt",
    "AGENT_PROMPTS",
    "SYSTEM_PROMPTS",
    "TOOL_PROMPTS",
]
