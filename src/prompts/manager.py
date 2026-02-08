"""
Archon Prompt Manager

Manages prompt templates with versioning, caching, and rendering.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.prompts.templates import (
    PromptTemplate,
    SYSTEM_PROMPTS,
    AGENT_PROMPTS,
    TOOL_PROMPTS,
    REACT_PROMPTS,
)
from src.monitoring.logger import get_logger

logger = get_logger("prompts.manager")


# =============================================================================
# Prompt Manager
# =============================================================================

class PromptManager:
    """
    Manages prompt templates with versioning and caching.

    Features:
    - Template registry with versioning
    - Variable substitution
    - Prompt caching
    - A/B testing support
    - Metrics collection
    """

    def __init__(self, prompts_dir: Optional[str] = None):
        self.prompts_dir = Path(prompts_dir) if prompts_dir else None

        # Combine all prompt dictionaries
        self._templates: dict[str, PromptTemplate] = {}
        self._register_builtin_prompts()

        # Load external prompts if directory specified
        if self.prompts_dir and self.prompts_dir.exists():
            self._load_external_prompts()

        # Usage metrics
        self._usage_counts: dict[str, int] = {}

    def _register_builtin_prompts(self):
        """Register all built-in prompts."""
        for name, template in SYSTEM_PROMPTS.items():
            self._templates[f"system.{name}"] = template

        for name, template in AGENT_PROMPTS.items():
            self._templates[f"agent.{name}"] = template

        for name, template in TOOL_PROMPTS.items():
            self._templates[f"tool.{name}"] = template

        for name, template in REACT_PROMPTS.items():
            self._templates[f"react.{name}"] = template

        logger.info(f"Registered {len(self._templates)} built-in prompts")

    def _load_external_prompts(self):
        """Load prompts from external JSON files."""
        if not self.prompts_dir:
            return

        for file_path in self.prompts_dir.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)

                template = PromptTemplate(
                    name=data.get("name", file_path.stem),
                    template=data["template"],
                    version=data.get("version", "1.0.0"),
                    description=data.get("description", ""),
                    variables=data.get("variables", []),
                )

                key = data.get("key", f"custom.{file_path.stem}")
                self._templates[key] = template

                logger.info(f"Loaded external prompt: {key}")

            except Exception as e:
                logger.error(f"Failed to load prompt {file_path}: {e}")

    def get(self, name: str) -> Optional[PromptTemplate]:
        """
        Get a prompt template by name.

        Args:
            name: Full prompt name (e.g., "agent.planner", "system.base")

        Returns:
            PromptTemplate or None
        """
        template = self._templates.get(name)

        if template:
            self._usage_counts[name] = self._usage_counts.get(name, 0) + 1

        return template

    def render(self, name: str, **kwargs) -> str:
        """
        Render a prompt template with variables.

        Args:
            name: Prompt name
            **kwargs: Variables to substitute

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If prompt not found
        """
        template = self.get(name)

        if not template:
            raise ValueError(f"Prompt not found: {name}")

        # Add default variables
        defaults = {
            "current_date": datetime.utcnow().strftime("%Y-%m-%d"),
        }

        # Merge with provided kwargs (kwargs take precedence)
        variables = {**defaults, **kwargs}

        rendered = template.render(**variables)

        logger.debug(
            f"Rendered prompt: {name}",
            metadata={
                "template_version": template.version,
                "variables": list(variables.keys()),
                "rendered_length": len(rendered),
            }
        )

        return rendered

    def register(self, key: str, template: PromptTemplate):
        """
        Register a new prompt template.

        Args:
            key: Template key (e.g., "custom.my_prompt")
            template: PromptTemplate instance
        """
        self._templates[key] = template
        logger.info(f"Registered prompt: {key} (v{template.version})")

    def list_prompts(self, prefix: Optional[str] = None) -> list[str]:
        """
        List all registered prompts.

        Args:
            prefix: Optional prefix filter (e.g., "agent.")

        Returns:
            List of prompt names
        """
        if prefix:
            return [k for k in self._templates.keys() if k.startswith(prefix)]
        return list(self._templates.keys())

    def get_usage_stats(self) -> dict[str, int]:
        """Get prompt usage statistics."""
        return self._usage_counts.copy()

    def update_template(
        self,
        name: str,
        new_template: str,
        new_version: Optional[str] = None,
    ):
        """
        Update an existing template.

        Args:
            name: Template name
            new_template: New template content
            new_version: New version string (auto-increments if not provided)
        """
        existing = self._templates.get(name)

        if not existing:
            raise ValueError(f"Template not found: {name}")

        # Auto-increment version if not provided
        if not new_version:
            parts = existing.version.split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            new_version = ".".join(parts)

        updated = PromptTemplate(
            name=existing.name,
            template=new_template,
            version=new_version,
            description=existing.description,
            variables=existing.variables,
            created_at=existing.created_at,
            updated_at=datetime.utcnow(),
        )

        self._templates[name] = updated

        logger.info(
            f"Updated prompt: {name}",
            metadata={
                "old_version": existing.version,
                "new_version": new_version,
            }
        )

    def export_prompts(self, output_dir: str):
        """Export all prompts to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for key, template in self._templates.items():
            filename = key.replace(".", "_") + ".json"
            file_path = output_path / filename

            data = {
                "key": key,
                "name": template.name,
                "template": template.template,
                "version": template.version,
                "description": template.description,
                "variables": template.variables,
            }

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

        logger.info(f"Exported {len(self._templates)} prompts to {output_dir}")


# =============================================================================
# Global Instance
# =============================================================================

_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def get_prompt(name: str) -> Optional[PromptTemplate]:
    """Get a prompt template by name."""
    return get_prompt_manager().get(name)


def render_prompt(name: str, **kwargs) -> str:
    """Render a prompt template with variables."""
    return get_prompt_manager().render(name, **kwargs)
