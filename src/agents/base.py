"""
Archon Base Agent

Abstract base class for all agents.
Implements ReAct pattern and common functionality.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from src.core.types import (
    AgentState,
    AgentType,
    Message,
    MessageRole,
    LLMResponse,
    ToolCall,
)
from src.core.errors import ArchonError
from src.llm.client import LLMClient, get_llm_client
from src.tools.executor import ToolExecutor, get_tool_executor
from src.prompts.manager import render_prompt
from src.monitoring.logger import get_logger

logger = get_logger("agents.base")


# =============================================================================
# Agent Context
# =============================================================================

@dataclass
class AgentContext:
    """Context passed to agent during execution."""
    run_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    max_iterations: int = 10
    timeout_seconds: float = 300.0
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Agent Result
# =============================================================================

@dataclass
class AgentResult:
    """Result from agent execution."""
    success: bool
    output: Any
    messages: list[Message] = field(default_factory=list)
    tool_calls_made: int = 0
    iterations: int = 0
    latency_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None


# =============================================================================
# Base Agent
# =============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Implements:
    - ReAct pattern (Reasoning + Acting)
    - Tool calling
    - Message management
    - Error handling
    """

    def __init__(
        self,
        agent_type: AgentType,
        llm_client: Optional[LLMClient] = None,
        tool_executor: Optional[ToolExecutor] = None,
    ):
        self.agent_type = agent_type
        self.agent_id = f"{agent_type.value}_{uuid4().hex[:8]}"
        self.llm_client = llm_client or get_llm_client()
        self.tool_executor = tool_executor or get_tool_executor()

        logger.info(f"Initialized agent: {self.agent_id}")

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass

    @property
    def available_tools(self) -> list[str]:
        """Return list of tools available to this agent."""
        return []

    async def run(
        self,
        task: str,
        context: Optional[AgentContext] = None,
        state: Optional[AgentState] = None,
    ) -> AgentResult:
        """
        Execute the agent's task.

        Args:
            task: The task description
            context: Execution context
            state: Current agent state (for graph)

        Returns:
            AgentResult with output and metadata
        """
        context = context or AgentContext()
        state = state or AgentState(query=task)
        start_time = time.time()

        # Initialize tracking
        total_tokens = 0
        total_cost = 0.0
        iterations = 0
        tool_calls_made = 0

        try:
            # Build initial messages
            messages = self._build_messages(task, state)

            # ReAct loop
            while iterations < context.max_iterations:
                iterations += 1

                # Get LLM response
                response = await self._get_completion(messages)
                total_tokens += response.total_tokens
                total_cost += response.cost_usd

                # Log step
                logger.log_agent_step(
                    agent_id=self.agent_id,
                    step=f"iteration_{iterations}",
                    thought=response.content[:200] if response.content else None,
                    metadata={"tokens": response.total_tokens},
                )

                # Check for tool calls
                if response.has_tool_calls:
                    tool_calls_made += len(response.tool_calls)

                    # Execute tools
                    tool_results = await self.tool_executor.execute_tool_calls(
                        tool_calls=response.tool_calls,
                        agent_type=self.agent_type,
                        agent_id=self.agent_id,
                    )

                    # Add tool results to messages
                    messages.append(Message(
                        role=MessageRole.ASSISTANT,
                        content=response.content,
                    ))

                    for tr in tool_results:
                        messages.append(Message(
                            role=MessageRole.TOOL,
                            content=str(tr.result),
                            name=tr.name,
                            tool_call_id=tr.tool_call_id,
                        ))

                    continue

                # No tool calls - check if done
                if self._is_complete(response, state):
                    # Extract final output
                    output = self._extract_output(response, state)

                    latency_ms = (time.time() - start_time) * 1000

                    return AgentResult(
                        success=True,
                        output=output,
                        messages=messages,
                        tool_calls_made=tool_calls_made,
                        iterations=iterations,
                        latency_ms=latency_ms,
                        tokens_used=total_tokens,
                        cost_usd=total_cost,
                    )

                # Add response to messages and continue
                messages.append(Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content,
                ))

            # Max iterations reached
            return AgentResult(
                success=False,
                output=None,
                messages=messages,
                tool_calls_made=tool_calls_made,
                iterations=iterations,
                latency_ms=(time.time() - start_time) * 1000,
                tokens_used=total_tokens,
                cost_usd=total_cost,
                error="Max iterations reached",
            )

        except Exception as e:
            logger.error(
                f"Agent execution failed: {self.agent_id}",
                metadata={"error": str(e), "iterations": iterations}
            )

            return AgentResult(
                success=False,
                output=None,
                iterations=iterations,
                latency_ms=(time.time() - start_time) * 1000,
                tokens_used=total_tokens,
                cost_usd=total_cost,
                error=str(e),
            )

    def _build_messages(self, task: str, state: AgentState) -> list[Message]:
        """Build initial message list."""
        messages = [
            Message(role=MessageRole.SYSTEM, content=self.system_prompt),
        ]

        # Add history if available
        messages.extend(state.messages)

        # Add current task
        messages.append(Message(
            role=MessageRole.USER,
            content=task,
        ))

        return messages

    async def _get_completion(self, messages: list[Message]) -> LLMResponse:
        """Get LLM completion, with or without tools."""
        if self.available_tools:
            tools = self.tool_executor.registry.get_openai_tools(self.available_tools)
            return await self.llm_client.complete_with_tools(messages, tools)
        else:
            return await self.llm_client.complete(messages)

    def _is_complete(self, response: LLMResponse, state: AgentState) -> bool:
        """
        Check if the agent has completed its task.

        Override in subclasses for custom completion logic.
        """
        # Default: complete if no tool calls and response contains content
        return bool(response.content) and not response.has_tool_calls

    def _extract_output(self, response: LLMResponse, state: AgentState) -> Any:
        """
        Extract the final output from the response.

        Override in subclasses for custom extraction.
        """
        return response.content

    async def step(self, state: AgentState) -> AgentState:
        """
        Execute a single step in a graph workflow.

        This is used by LangGraph for multi-agent coordination.
        """
        # Get the last user message as the task
        task = state.query
        for msg in reversed(state.messages):
            if msg.role == MessageRole.USER:
                task = msg.content
                break

        # Run the agent
        result = await self.run(task, state=state)

        # Update state
        state.messages.extend(result.messages)
        state.iteration += 1

        if result.success:
            state.context[f"{self.agent_type.value}_output"] = result.output
        else:
            state.error = result.error

        return state
