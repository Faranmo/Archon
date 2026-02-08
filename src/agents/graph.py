"""
Archon Agent Graph

LangGraph workflow for multi-agent research.
"""

from typing import Any, Callable, Optional, TypedDict, Annotated
from enum import Enum
import operator

from src.core.types import AgentState, AgentType, Message
from src.agents.base import AgentContext
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent
from src.agents.analyst import AnalystAgent
from src.agents.writer import WriterAgent
from src.agents.verifier import VerifierAgent
from src.agents.supervisor import SupervisorAgent
from src.monitoring.logger import get_logger

logger = get_logger("agents.graph")


# =============================================================================
# State Definition
# =============================================================================

class GraphState(TypedDict):
    """State passed through the graph."""
    messages: Annotated[list, operator.add]
    query: str
    plan: Optional[dict]
    research: Optional[dict]
    analysis: Optional[dict]
    draft: Optional[str]
    verification: Optional[dict]
    final_output: Optional[str]
    current_agent: Optional[str]
    iteration: int
    error: Optional[str]


# =============================================================================
# Node Functions
# =============================================================================

async def plan_node(state: GraphState) -> GraphState:
    """Planning node."""
    logger.info("Executing plan node")

    planner = PlannerAgent()
    plan = await planner.create_plan(state["query"])

    return {
        **state,
        "plan": plan,
        "current_agent": "planner",
        "iteration": state.get("iteration", 0) + 1,
    }


async def research_node(state: GraphState) -> GraphState:
    """Research node."""
    logger.info("Executing research node")

    researcher = ResearcherAgent()

    # Use plan if available
    topic = state["query"]
    if state.get("plan") and state["plan"].get("steps"):
        # Research based on first step
        topic = state["plan"]["steps"][0].get("description", topic)

    research = await researcher.research(topic)

    return {
        **state,
        "research": research,
        "current_agent": "researcher",
        "iteration": state.get("iteration", 0) + 1,
    }


async def analyze_node(state: GraphState) -> GraphState:
    """Analysis node."""
    logger.info("Executing analyze node")

    analyst = AnalystAgent()

    # Analyze research findings
    data = state.get("research", {})
    analysis = await analyst.analyze(
        data=data,
        questions=[
            "What are the key findings?",
            "What patterns emerge from the data?",
            "What are the implications?",
        ],
    )

    return {
        **state,
        "analysis": analysis,
        "current_agent": "analyst",
        "iteration": state.get("iteration", 0) + 1,
    }


async def write_node(state: GraphState) -> GraphState:
    """Writing node."""
    logger.info("Executing write node")

    writer = WriterAgent()

    # Combine research and analysis
    import json
    content = {
        "query": state["query"],
        "research": state.get("research", {}),
        "analysis": state.get("analysis", {}),
    }

    result = await writer.create_report(
        research_findings=content,
        title=f"Research Report: {state['query'][:50]}",
    )

    return {
        **state,
        "draft": result.get("content", ""),
        "current_agent": "writer",
        "iteration": state.get("iteration", 0) + 1,
    }


async def verify_node(state: GraphState) -> GraphState:
    """Verification node."""
    logger.info("Executing verify node")

    verifier = VerifierAgent()

    verification = await verifier.verify(
        content=state.get("draft", ""),
        original_request=state["query"],
    )

    # If passed, set as final output
    if verification.get("passed", False):
        return {
            **state,
            "verification": verification,
            "final_output": state.get("draft"),
            "current_agent": "verifier",
            "iteration": state.get("iteration", 0) + 1,
        }

    return {
        **state,
        "verification": verification,
        "current_agent": "verifier",
        "iteration": state.get("iteration", 0) + 1,
    }


# =============================================================================
# Router Function
# =============================================================================

def should_continue(state: GraphState) -> str:
    """Determine next step in the graph."""

    # Check for errors
    if state.get("error"):
        return "end"

    # Check iteration limit
    if state.get("iteration", 0) >= 10:
        return "end"

    # Check completion
    if state.get("final_output"):
        return "end"

    # Determine next step
    if not state.get("plan"):
        return "plan"

    if not state.get("research"):
        return "research"

    if not state.get("analysis"):
        return "analyze"

    if not state.get("draft"):
        return "write"

    if not state.get("verification"):
        return "verify"

    # If verification failed, rewrite
    if state.get("verification") and not state["verification"].get("passed"):
        return "write"

    return "end"


# =============================================================================
# Graph Creation
# =============================================================================

def create_research_graph():
    """
    Create the research workflow graph.

    Returns a compiled LangGraph.
    """
    try:
        from langgraph.graph import StateGraph, END

        # Create graph
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("plan", plan_node)
        workflow.add_node("research", research_node)
        workflow.add_node("analyze", analyze_node)
        workflow.add_node("write", write_node)
        workflow.add_node("verify", verify_node)

        # Add edges
        workflow.add_conditional_edges(
            "plan",
            should_continue,
            {
                "research": "research",
                "end": END,
            }
        )

        workflow.add_conditional_edges(
            "research",
            should_continue,
            {
                "analyze": "analyze",
                "end": END,
            }
        )

        workflow.add_conditional_edges(
            "analyze",
            should_continue,
            {
                "write": "write",
                "end": END,
            }
        )

        workflow.add_conditional_edges(
            "write",
            should_continue,
            {
                "verify": "verify",
                "end": END,
            }
        )

        workflow.add_conditional_edges(
            "verify",
            should_continue,
            {
                "write": "write",  # Rewrite if failed
                "end": END,
            }
        )

        # Set entry point
        workflow.set_entry_point("plan")

        # Compile
        return workflow.compile()

    except ImportError:
        logger.warning("LangGraph not installed, using simple sequential flow")
        return None


# =============================================================================
# Run Workflow
# =============================================================================

async def run_research_workflow(
    query: str,
    context: Optional[AgentContext] = None,
) -> dict:
    """
    Run the research workflow.

    Args:
        query: Research query
        context: Execution context

    Returns:
        Workflow result
    """
    # Try to use LangGraph
    graph = create_research_graph()

    if graph:
        # Run with LangGraph
        initial_state: GraphState = {
            "messages": [],
            "query": query,
            "plan": None,
            "research": None,
            "analysis": None,
            "draft": None,
            "verification": None,
            "final_output": None,
            "current_agent": None,
            "iteration": 0,
            "error": None,
        }

        result = await graph.ainvoke(initial_state)

        return {
            "success": result.get("final_output") is not None,
            "output": result.get("final_output"),
            "plan": result.get("plan"),
            "research": result.get("research"),
            "analysis": result.get("analysis"),
            "verification": result.get("verification"),
            "iterations": result.get("iteration", 0),
        }

    else:
        # Fallback to simple sequential execution
        logger.info("Running sequential workflow (LangGraph not available)")

        state: GraphState = {
            "messages": [],
            "query": query,
            "plan": None,
            "research": None,
            "analysis": None,
            "draft": None,
            "verification": None,
            "final_output": None,
            "current_agent": None,
            "iteration": 0,
            "error": None,
        }

        try:
            state = await plan_node(state)
            state = await research_node(state)
            state = await analyze_node(state)
            state = await write_node(state)
            state = await verify_node(state)

            return {
                "success": state.get("final_output") is not None,
                "output": state.get("final_output"),
                "plan": state.get("plan"),
                "research": state.get("research"),
                "analysis": state.get("analysis"),
                "verification": state.get("verification"),
                "iterations": state.get("iteration", 0),
            }

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }
