"""
Archon Agents Module

LangGraph-based multi-agent system following arXiv 2512.08769 best practices:
- Single-responsibility agents
- Tool-first design
- Supervisor coordination
- ReAct reasoning pattern
"""

from src.agents.base import (
    BaseAgent,
    AgentContext,
)
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent
from src.agents.analyst import AnalystAgent
from src.agents.writer import WriterAgent
from src.agents.verifier import VerifierAgent
from src.agents.supervisor import SupervisorAgent
from src.agents.graph import (
    create_research_graph,
    run_research_workflow,
)

__all__ = [
    "BaseAgent",
    "AgentContext",
    "PlannerAgent",
    "ResearcherAgent",
    "AnalystAgent",
    "WriterAgent",
    "VerifierAgent",
    "SupervisorAgent",
    "create_research_graph",
    "run_research_workflow",
]
