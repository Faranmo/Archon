"""
Archon Supervisor Agent

Orchestrates multi-agent workflows.
Single responsibility: Agent coordination.
"""

from typing import Any, Optional, Callable
from enum import Enum

from src.agents.base import BaseAgent, AgentContext, AgentResult
from src.core.types import AgentType, AgentState
from src.prompts.manager import render_prompt
from src.monitoring.logger import get_logger

logger = get_logger("agents.supervisor")


class SupervisorDecision(Enum):
    """Possible supervisor decisions."""
    DELEGATE = "delegate"
    QUERY = "query"
    COMPLETE = "complete"
    ESCALATE = "escalate"


class SupervisorAgent(BaseAgent):
    """
    Supervisor agent for orchestration.

    Responsibilities:
    - Monitor agent progress
    - Coordinate between agents
    - Handle errors and edge cases
    - Make final decisions
    """

    def __init__(self, **kwargs):
        super().__init__(agent_type=AgentType.SUPERVISOR, **kwargs)

        # Track agent states
        self._agent_states: dict[str, dict] = {}

    @property
    def system_prompt(self) -> str:
        """Supervisor system prompt."""
        return render_prompt(
            "agent.supervisor",
            current_state="State will be provided.",
            active_agents="Agent status will be provided.",
            task_progress="Progress will be provided.",
        )

    @property
    def available_tools(self) -> list[str]:
        """Supervisor has broad access."""
        return ["semantic_search"]

    async def coordinate(
        self,
        task: str,
        state: AgentState,
        available_agents: list[str],
    ) -> dict:
        """
        Coordinate a multi-agent task.

        Args:
            task: The overall task
            state: Current workflow state
            available_agents: List of available agent types

        Returns:
            Coordination decision
        """
        # Build status summary
        agent_status = self._format_agent_status(available_agents)
        progress = self._format_progress(state)

        supervisor_task = render_prompt(
            "agent.supervisor",
            current_state=f"Task: {task}\n\nIteration: {state.iteration}/{state.max_iterations}",
            active_agents=agent_status,
            task_progress=progress,
        )

        # Get decision
        result = await self.run(supervisor_task)

        if not result.success:
            return {
                "decision": SupervisorDecision.ESCALATE.value,
                "error": result.error,
            }

        # Parse decision
        decision = self._parse_decision(result.output)

        return decision

    def _format_agent_status(self, agents: list[str]) -> str:
        """Format agent status for prompt."""
        lines = []
        for agent in agents:
            status = self._agent_states.get(agent, {"status": "available"})
            lines.append(f"- {agent}: {status.get('status', 'available')}")
        return "\n".join(lines)

    def _format_progress(self, state: AgentState) -> str:
        """Format progress for prompt."""
        progress = []

        if state.retrieval_results:
            progress.append(f"Retrieved {len(state.retrieval_results)} documents")

        if state.analysis:
            progress.append("Analysis completed")

        if state.final_output:
            progress.append("Final output generated")

        if state.error:
            progress.append(f"Error: {state.error}")

        return "\n".join(progress) if progress else "Task in progress"

    def _parse_decision(self, output: str) -> dict:
        """Parse supervisor decision from output."""
        decision = {
            "decision": SupervisorDecision.COMPLETE.value,
            "next_agent": None,
            "instructions": None,
            "reasoning": output,
        }

        lower = output.lower()

        # Check for decision type
        if "delegate" in lower:
            decision["decision"] = SupervisorDecision.DELEGATE.value

            # Try to find target agent
            for agent_type in AgentType:
                if agent_type.value in lower:
                    decision["next_agent"] = agent_type.value
                    break

        elif "complete" in lower or "done" in lower or "finish" in lower:
            decision["decision"] = SupervisorDecision.COMPLETE.value

        elif "escalate" in lower or "human" in lower:
            decision["decision"] = SupervisorDecision.ESCALATE.value

        return decision

    def update_agent_state(self, agent_id: str, status: str, metadata: dict = None):
        """Update tracked agent state."""
        self._agent_states[agent_id] = {
            "status": status,
            **(metadata or {}),
        }

    def should_continue(self, state: AgentState) -> bool:
        """
        Determine if workflow should continue.

        Used as a conditional edge in LangGraph.
        """
        # Check for completion
        if state.final_output:
            return False

        # Check for errors
        if state.error:
            return False

        # Check iteration limit
        if state.iteration >= state.max_iterations:
            return False

        return True

    def select_next_agent(self, state: AgentState) -> str:
        """
        Select the next agent to run.

        Returns agent type name for graph routing.
        """
        # Simple round-robin for now
        # In production, use supervisor LLM decision

        if not state.retrieval_results:
            return AgentType.RESEARCHER.value

        if not state.analysis:
            return AgentType.ANALYST.value

        if not state.final_output:
            return AgentType.WRITER.value

        return "end"
