"""
Archon Planner Agent

Analyzes tasks and creates actionable plans.
Single responsibility: Task decomposition and planning.
"""

from typing import Any, Optional

from src.agents.base import BaseAgent, AgentContext, AgentResult
from src.core.types import AgentType, AgentState
from src.prompts.manager import render_prompt
from src.monitoring.logger import get_logger

logger = get_logger("agents.planner")


class PlannerAgent(BaseAgent):
    """
    Planner agent for task decomposition.

    Responsibilities:
    - Analyze user requests
    - Break down into subtasks
    - Assign to appropriate agents
    - Identify dependencies
    """

    def __init__(self, **kwargs):
        super().__init__(agent_type=AgentType.PLANNER, **kwargs)

    @property
    def system_prompt(self) -> str:
        """Planner system prompt."""
        return render_prompt("agent.planner", user_request="", context="")

    @property
    def available_tools(self) -> list[str]:
        """Planner has limited tools - mainly for research."""
        return ["semantic_search"]

    async def create_plan(
        self,
        request: str,
        context: Optional[dict] = None,
    ) -> dict:
        """
        Create a structured plan for a request.

        Args:
            request: User's request
            context: Additional context

        Returns:
            Structured plan dictionary
        """
        # Build task with context
        context_str = ""
        if context:
            context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())

        task = render_prompt(
            "agent.planner",
            user_request=request,
            context=context_str or "No additional context provided.",
        )

        # Run agent
        result = await self.run(task)

        if not result.success:
            return {
                "success": False,
                "error": result.error,
                "steps": [],
            }

        # Parse the plan from output
        plan = self._parse_plan(result.output)
        plan["success"] = True
        plan["raw_output"] = result.output

        return plan

    def _parse_plan(self, output: str) -> dict:
        """Parse plan structure from LLM output."""
        plan = {
            "steps": [],
            "dependencies": [],
            "risks": [],
        }

        current_section = None
        lines = output.split("\n")

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # Detect sections
            if "PLAN:" in line.upper():
                current_section = "steps"
            elif "DEPENDENCIES:" in line.upper():
                current_section = "dependencies"
            elif "RISKS:" in line.upper():
                current_section = "risks"
            elif current_section and line:
                # Parse line based on section
                if current_section == "steps":
                    # Parse step: "1. [description] - Assigned to: [agent]"
                    step = self._parse_step(line)
                    if step:
                        plan["steps"].append(step)
                elif current_section == "dependencies":
                    plan["dependencies"].append(line.lstrip("- "))
                elif current_section == "risks":
                    plan["risks"].append(line.lstrip("- "))

        return plan

    def _parse_step(self, line: str) -> Optional[dict]:
        """Parse a single step line."""
        import re

        # Try to match: "1. Description - Assigned to: Agent"
        match = re.match(
            r"(\d+)\.\s*(.+?)(?:\s*-\s*Assigned to:\s*(\w+))?$",
            line,
            re.IGNORECASE,
        )

        if match:
            return {
                "number": int(match.group(1)),
                "description": match.group(2).strip(),
                "assigned_to": match.group(3) or "researcher",
            }

        # Simpler format: just numbered item
        match = re.match(r"(\d+)\.\s*(.+)$", line)
        if match:
            return {
                "number": int(match.group(1)),
                "description": match.group(2).strip(),
                "assigned_to": "researcher",
            }

        return None
