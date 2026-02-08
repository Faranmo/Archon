"""
Tests for agents module.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.agents.base import BaseAgent, AgentContext, AgentResult
from src.core.types import AgentType, Message, MessageRole, LLMResponse


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    @property
    def system_prompt(self):
        return "You are a test agent."

    @property
    def available_tools(self):
        return []


class TestBaseAgent:
    """Test base agent functionality."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = MockAgent(agent_type=AgentType.RESEARCHER)
        assert agent.agent_type == AgentType.RESEARCHER
        assert "researcher" in agent.agent_id

    @pytest.mark.asyncio
    async def test_agent_run_success(self, mock_llm_client):
        """Test successful agent run."""
        agent = MockAgent(
            agent_type=AgentType.ANALYST,
            llm_client=mock_llm_client,
        )

        result = await agent.run("Analyze this data")

        assert result.success is True
        assert result.output == "Test response"
        assert result.iterations >= 1

    @pytest.mark.asyncio
    async def test_agent_run_max_iterations(self, mock_llm_client):
        """Test max iterations limit."""
        # Make LLM always request tools
        mock_llm_client.complete = AsyncMock(return_value=MagicMock(
            content="",
            tool_calls=[MagicMock()],  # Non-empty tool calls
            has_tool_calls=True,
            total_tokens=100,
            cost_usd=0.001,
        ))

        agent = MockAgent(
            agent_type=AgentType.RESEARCHER,
            llm_client=mock_llm_client,
        )

        context = AgentContext(max_iterations=2)
        result = await agent.run("Test", context=context)

        assert result.success is False
        assert "Max iterations" in (result.error or "")


class TestAgentContext:
    """Test agent context."""

    def test_context_defaults(self):
        """Test context default values."""
        context = AgentContext()
        assert context.max_iterations == 10
        assert context.timeout_seconds == 300.0
        assert context.run_id is not None
