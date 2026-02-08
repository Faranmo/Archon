"""
Archon API Routes

REST API endpoints for the research assistant.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from src.core.types import APIResponse
from src.agents.graph import run_research_workflow
from src.agents.base import AgentContext
from src.rag.retriever import get_retriever, retrieve
from src.monitoring.cost_tracker import get_cost_tracker
from src.monitoring.logger import get_logger

logger = get_logger("api.routes")

router = APIRouter()


# =============================================================================
# Request/Response Schemas
# =============================================================================

class ResearchRequest(BaseModel):
    """Research request schema."""
    query: str = Field(..., description="Research query", min_length=1, max_length=10000)
    depth: str = Field("moderate", description="Research depth: quick, moderate, thorough")
    max_iterations: int = Field(10, ge=1, le=20)


class ResearchResponse(BaseModel):
    """Research response schema."""
    success: bool
    output: Optional[str] = None
    plan: Optional[dict] = None
    research: Optional[dict] = None
    analysis: Optional[dict] = None
    verification: Optional[dict] = None
    iterations: int = 0
    error: Optional[str] = None


class SearchRequest(BaseModel):
    """Search request schema."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, ge=1, le=50)
    filter_source: Optional[str] = None


class SearchResult(BaseModel):
    """Search result item."""
    content: str
    score: float
    metadata: dict


class SearchResponse(BaseModel):
    """Search response schema."""
    success: bool
    results: list[SearchResult]
    total: int


class CostReportResponse(BaseModel):
    """Cost report response schema."""
    period_hours: int
    total_requests: int
    total_cost_usd: float
    total_tokens: int
    cost_by_model: dict
    budget: dict


# =============================================================================
# Research Endpoints
# =============================================================================

@router.post("/research", response_model=ResearchResponse, tags=["Research"])
async def run_research(request: ResearchRequest):
    """
    Run a research workflow.

    This endpoint triggers a multi-agent research workflow that:
    1. Plans the research approach
    2. Gathers information from multiple sources
    3. Analyzes the findings
    4. Generates a comprehensive report
    5. Verifies the output for accuracy
    """
    logger.info(f"Research request: {request.query[:100]}...")

    try:
        context = AgentContext(max_iterations=request.max_iterations)

        result = await run_research_workflow(
            query=request.query,
            context=context,
        )

        return ResearchResponse(
            success=result.get("success", False),
            output=result.get("output"),
            plan=result.get("plan"),
            research=result.get("research"),
            analysis=result.get("analysis"),
            verification=result.get("verification"),
            iterations=result.get("iterations", 0),
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"Research failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/research/quick", response_model=ResearchResponse, tags=["Research"])
async def quick_research(request: ResearchRequest):
    """
    Run a quick research query.

    Faster than full research - uses fewer iterations and simpler workflow.
    """
    request.depth = "quick"
    request.max_iterations = 5
    return await run_research(request)


# =============================================================================
# Search Endpoints
# =============================================================================

@router.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_documents(request: SearchRequest):
    """
    Search indexed documents.

    Uses hybrid search (BM25 + vector) to find relevant documents.
    """
    logger.info(f"Search request: {request.query[:100]}...")

    try:
        filter_metadata = None
        if request.filter_source:
            filter_metadata = {"source": request.filter_source}

        results = await retrieve(request.query, top_k=request.top_k)

        return SearchResponse(
            success=True,
            results=[
                SearchResult(
                    content=r.chunk.content,
                    score=r.score,
                    metadata=r.chunk.metadata,
                )
                for r in results
            ],
            total=len(results),
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return SearchResponse(success=False, results=[], total=0)


# =============================================================================
# Monitoring Endpoints
# =============================================================================

@router.get("/costs", response_model=CostReportResponse, tags=["Monitoring"])
async def get_cost_report(hours: int = 24):
    """
    Get cost tracking report.

    Returns usage and cost metrics for the specified time period.
    """
    tracker = get_cost_tracker()
    report = tracker.get_report(hours=hours)

    return CostReportResponse(
        period_hours=report["period_hours"],
        total_requests=report["total_requests"],
        total_cost_usd=report["total_cost_usd"],
        total_tokens=report["total_tokens"],
        cost_by_model=report["cost_by_model"],
        budget=report["budget"],
    )


@router.get("/costs/budget", tags=["Monitoring"])
async def get_budget_status():
    """
    Get current budget status.

    Returns remaining daily budget and whether limit is exceeded.
    """
    tracker = get_cost_tracker()

    return {
        "remaining_usd": tracker.get_remaining_budget(),
        "exceeded": tracker.is_budget_exceeded(),
        "daily_limit_usd": tracker.daily_budget_usd,
    }


# =============================================================================
# Document Endpoints
# =============================================================================

class DocumentUploadRequest(BaseModel):
    """Document upload request."""
    content: str
    filename: str
    metadata: Optional[dict] = None


@router.post("/documents", tags=["Documents"])
async def upload_document(request: DocumentUploadRequest, background_tasks: BackgroundTasks):
    """
    Upload a document for indexing.

    The document will be chunked, embedded, and indexed in the background.
    """
    from src.core.types import Document, DocumentType
    from src.rag.chunking import chunk_document
    from src.rag.embeddings import embed_chunks

    # Create document
    doc = Document(
        content=request.content,
        source=request.filename,
        metadata=request.metadata or {},
    )

    # Schedule background indexing
    async def index_document():
        chunks = chunk_document(doc)
        chunks = await embed_chunks(chunks)
        retriever = get_retriever()
        retriever.index(chunks)
        logger.info(f"Indexed document: {request.filename}")

    background_tasks.add_task(index_document)

    return {
        "success": True,
        "document_id": doc.id,
        "message": "Document queued for indexing",
    }


# =============================================================================
# Agent Endpoints
# =============================================================================

@router.get("/agents", tags=["Agents"])
async def list_agents():
    """
    List available agents.
    """
    from src.core.types import AgentType

    return {
        "agents": [
            {
                "type": agent.value,
                "description": {
                    "planner": "Task decomposition and planning",
                    "researcher": "Information gathering and verification",
                    "analyst": "Data analysis and insights",
                    "writer": "Content generation and synthesis",
                    "verifier": "Quality assurance and fact-checking",
                    "supervisor": "Agent orchestration and coordination",
                }.get(agent.value, ""),
            }
            for agent in AgentType
        ]
    }


class AgentRunRequest(BaseModel):
    """Agent run request."""
    agent_type: str
    task: str
    context: Optional[dict] = None


@router.post("/agents/run", tags=["Agents"])
async def run_agent(request: AgentRunRequest):
    """
    Run a specific agent.
    """
    from src.core.types import AgentType
    from src.agents.planner import PlannerAgent
    from src.agents.researcher import ResearcherAgent
    from src.agents.analyst import AnalystAgent
    from src.agents.writer import WriterAgent
    from src.agents.verifier import VerifierAgent

    agent_map = {
        "planner": PlannerAgent,
        "researcher": ResearcherAgent,
        "analyst": AnalystAgent,
        "writer": WriterAgent,
        "verifier": VerifierAgent,
    }

    if request.agent_type not in agent_map:
        raise HTTPException(status_code=400, detail=f"Unknown agent: {request.agent_type}")

    agent = agent_map[request.agent_type]()
    result = await agent.run(request.task)

    return {
        "success": result.success,
        "output": result.output,
        "iterations": result.iterations,
        "tokens_used": result.tokens_used,
        "cost_usd": result.cost_usd,
        "error": result.error,
    }
