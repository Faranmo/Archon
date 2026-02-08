"""
Archon Shared Type Definitions

Common types used across the application.
Provides type safety and documentation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


# =============================================================================
# Enums
# =============================================================================

class RunStatus(Enum):
    """Status of an agent run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageRole(Enum):
    """Role of a message in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class AgentType(Enum):
    """Types of agents in the system."""
    PLANNER = "planner"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    VERIFIER = "verifier"
    SUPERVISOR = "supervisor"


class ToolPermission(Enum):
    """Permission levels for tool access."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    EXTERNAL_API = "external_api"


class DocumentType(Enum):
    """Types of documents supported."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "markdown"
    HTML = "html"
    CSV = "csv"
    JSON = "json"
    UNKNOWN = "unknown"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Message:
    """A message in a conversation."""
    role: MessageRole
    content: str
    name: Optional[str] = None  # For tool messages
    tool_call_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API calls."""
        result = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


@dataclass
class Document:
    """A document in the system."""
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_type: DocumentType = DocumentType.UNKNOWN
    source: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def word_count(self) -> int:
        return len(self.content.split())


@dataclass
class Chunk:
    """A chunk of a document for RAG."""
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    document_id: str = ""
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None

    # Contextual information
    header: Optional[str] = None  # Section header
    parent_header: Optional[str] = None  # Parent section


@dataclass
class RetrievalResult:
    """Result from RAG retrieval."""
    chunk: Chunk
    score: float  # Similarity score
    rerank_score: Optional[float] = None

    @property
    def content(self) -> str:
        return self.chunk.content


@dataclass
class AgentState:
    """State passed between agents in LangGraph."""
    messages: list[Message] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    current_agent: Optional[AgentType] = None
    next_agent: Optional[AgentType] = None
    iteration: int = 0
    max_iterations: int = 10
    error: Optional[str] = None

    # Task-specific state
    query: str = ""
    documents: list[Document] = field(default_factory=list)
    retrieval_results: list[RetrievalResult] = field(default_factory=list)
    analysis: Optional[str] = None
    final_output: Optional[str] = None

    def add_message(self, role: MessageRole, content: str, **kwargs):
        """Add a message to the state."""
        self.messages.append(Message(role=role, content=content, **kwargs))

    def get_last_message(self) -> Optional[Message]:
        """Get the last message."""
        return self.messages[-1] if self.messages else None


@dataclass
class ToolCall:
    """A tool call request from an LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_call_id: str
    name: str
    result: Any
    success: bool = True
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass
class ResearchReport:
    """Final research report output."""
    id: str = field(default_factory=lambda: str(uuid4()))
    query: str = ""
    summary: str = ""
    sections: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Metrics
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    latency_ms: float = 0.0


# =============================================================================
# API Types
# =============================================================================

@dataclass
class APIResponse:
    """Standard API response format."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    request_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result = {"success": self.success}
        if self.data is not None:
            result["data"] = self.data
        if self.error:
            result["error"] = self.error
        if self.error_code:
            result["error_code"] = self.error_code
        if self.request_id:
            result["request_id"] = self.request_id
        return result
