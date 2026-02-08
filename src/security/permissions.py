"""
Archon Permission Management Module

Controls what actions agents and users can perform.
Implements principle of least privilege for tool access.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Any
from functools import wraps

from src.core.types import ToolPermission, AgentType
from src.core.errors import AuthorizationError
from src.monitoring.logger import get_logger

logger = get_logger("security.permissions")


# =============================================================================
# Permission Levels
# =============================================================================

class RiskLevel(Enum):
    """Risk levels for operations."""
    LOW = "low"           # Read operations, safe computations
    MEDIUM = "medium"     # Write to sandboxed locations
    HIGH = "high"         # External API calls, file writes
    CRITICAL = "critical"  # System commands, credential access


# =============================================================================
# Tool Definitions
# =============================================================================

@dataclass
class ToolDefinition:
    """Definition of a tool with its permissions."""
    name: str
    description: str
    required_permissions: set[ToolPermission]
    risk_level: RiskLevel
    requires_confirmation: bool = False  # Human-in-the-loop
    allowed_agents: Optional[set[AgentType]] = None  # None = all agents
    rate_limit_per_minute: int = 60

    def __hash__(self):
        return hash(self.name)


# =============================================================================
# Default Tool Registry
# =============================================================================

DEFAULT_TOOLS: dict[str, ToolDefinition] = {
    # Low risk - read operations
    "search_documents": ToolDefinition(
        name="search_documents",
        description="Search through indexed documents",
        required_permissions={ToolPermission.READ},
        risk_level=RiskLevel.LOW,
    ),
    "get_current_time": ToolDefinition(
        name="get_current_time",
        description="Get current date and time",
        required_permissions=set(),
        risk_level=RiskLevel.LOW,
    ),
    "calculate": ToolDefinition(
        name="calculate",
        description="Perform mathematical calculations",
        required_permissions=set(),
        risk_level=RiskLevel.LOW,
    ),

    # Medium risk - write to sandboxed areas
    "save_draft": ToolDefinition(
        name="save_draft",
        description="Save a draft document",
        required_permissions={ToolPermission.WRITE},
        risk_level=RiskLevel.MEDIUM,
    ),
    "create_report": ToolDefinition(
        name="create_report",
        description="Create a research report",
        required_permissions={ToolPermission.WRITE},
        risk_level=RiskLevel.MEDIUM,
        allowed_agents={AgentType.WRITER, AgentType.ANALYST},
    ),

    # High risk - external API calls
    "web_search": ToolDefinition(
        name="web_search",
        description="Search the web for information",
        required_permissions={ToolPermission.EXTERNAL_API},
        risk_level=RiskLevel.HIGH,
        rate_limit_per_minute=20,
    ),
    "fetch_url": ToolDefinition(
        name="fetch_url",
        description="Fetch content from a URL",
        required_permissions={ToolPermission.EXTERNAL_API},
        risk_level=RiskLevel.HIGH,
        rate_limit_per_minute=30,
    ),
    "send_email": ToolDefinition(
        name="send_email",
        description="Send an email",
        required_permissions={ToolPermission.EXTERNAL_API, ToolPermission.WRITE},
        risk_level=RiskLevel.HIGH,
        requires_confirmation=True,
    ),

    # Critical risk - system operations
    "execute_code": ToolDefinition(
        name="execute_code",
        description="Execute code in a sandbox",
        required_permissions={ToolPermission.EXECUTE},
        risk_level=RiskLevel.CRITICAL,
        requires_confirmation=True,
        rate_limit_per_minute=10,
    ),
    "file_system_write": ToolDefinition(
        name="file_system_write",
        description="Write to file system",
        required_permissions={ToolPermission.WRITE, ToolPermission.EXECUTE},
        risk_level=RiskLevel.CRITICAL,
        requires_confirmation=True,
    ),
}


# =============================================================================
# Agent Permission Profiles
# =============================================================================

@dataclass
class AgentPermissionProfile:
    """Permission profile for an agent type."""
    agent_type: AgentType
    allowed_permissions: set[ToolPermission]
    max_risk_level: RiskLevel
    requires_supervision: bool = False
    max_tool_calls_per_run: int = 50


DEFAULT_AGENT_PROFILES: dict[AgentType, AgentPermissionProfile] = {
    AgentType.PLANNER: AgentPermissionProfile(
        agent_type=AgentType.PLANNER,
        allowed_permissions={ToolPermission.READ},
        max_risk_level=RiskLevel.LOW,
    ),
    AgentType.RESEARCHER: AgentPermissionProfile(
        agent_type=AgentType.RESEARCHER,
        allowed_permissions={ToolPermission.READ, ToolPermission.EXTERNAL_API},
        max_risk_level=RiskLevel.HIGH,
    ),
    AgentType.ANALYST: AgentPermissionProfile(
        agent_type=AgentType.ANALYST,
        allowed_permissions={ToolPermission.READ, ToolPermission.WRITE},
        max_risk_level=RiskLevel.MEDIUM,
    ),
    AgentType.WRITER: AgentPermissionProfile(
        agent_type=AgentType.WRITER,
        allowed_permissions={ToolPermission.READ, ToolPermission.WRITE},
        max_risk_level=RiskLevel.MEDIUM,
    ),
    AgentType.VERIFIER: AgentPermissionProfile(
        agent_type=AgentType.VERIFIER,
        allowed_permissions={ToolPermission.READ},
        max_risk_level=RiskLevel.LOW,
    ),
    AgentType.SUPERVISOR: AgentPermissionProfile(
        agent_type=AgentType.SUPERVISOR,
        allowed_permissions={ToolPermission.READ, ToolPermission.WRITE, ToolPermission.EXTERNAL_API},
        max_risk_level=RiskLevel.HIGH,
        requires_supervision=False,  # Supervisor doesn't need supervision
    ),
}


# =============================================================================
# Permission Manager
# =============================================================================

class PermissionManager:
    """
    Manages permissions for tools and agents.

    Implements:
    - Role-based access control (RBAC)
    - Principle of least privilege
    - Human-in-the-loop for critical operations
    """

    def __init__(
        self,
        tools: Optional[dict[str, ToolDefinition]] = None,
        agent_profiles: Optional[dict[AgentType, AgentPermissionProfile]] = None,
    ):
        self.tools = tools or DEFAULT_TOOLS.copy()
        self.agent_profiles = agent_profiles or DEFAULT_AGENT_PROFILES.copy()
        self._tool_call_counts: dict[str, dict[str, int]] = {}  # agent_id -> tool_name -> count

    def register_tool(self, tool: ToolDefinition):
        """Register a new tool."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}", metadata={"risk_level": tool.risk_level.value})

    def check_permission(
        self,
        tool_name: str,
        agent_type: AgentType,
        agent_id: str,
        user_permissions: Optional[set[ToolPermission]] = None,
    ) -> tuple[bool, str]:
        """
        Check if an agent can use a tool.

        Args:
            tool_name: Name of the tool
            agent_type: Type of agent requesting
            agent_id: Unique agent identifier
            user_permissions: Additional user-granted permissions

        Returns:
            Tuple of (allowed, reason)
        """
        # Get tool definition
        tool = self.tools.get(tool_name)
        if not tool:
            return False, f"Unknown tool: {tool_name}"

        # Get agent profile
        profile = self.agent_profiles.get(agent_type)
        if not profile:
            return False, f"Unknown agent type: {agent_type}"

        # Check agent type restriction
        if tool.allowed_agents and agent_type not in tool.allowed_agents:
            return False, f"Tool {tool_name} not allowed for {agent_type.value} agents"

        # Check risk level
        risk_order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        if risk_order.index(tool.risk_level) > risk_order.index(profile.max_risk_level):
            return False, f"Tool risk level {tool.risk_level.value} exceeds agent max {profile.max_risk_level.value}"

        # Check permissions
        combined_permissions = profile.allowed_permissions.copy()
        if user_permissions:
            combined_permissions.update(user_permissions)

        missing_permissions = tool.required_permissions - combined_permissions
        if missing_permissions:
            missing_str = ", ".join(p.value for p in missing_permissions)
            return False, f"Missing permissions: {missing_str}"

        # Check tool call limits
        if agent_id not in self._tool_call_counts:
            self._tool_call_counts[agent_id] = {}

        tool_counts = self._tool_call_counts[agent_id]
        current_count = tool_counts.get(tool_name, 0)

        if current_count >= profile.max_tool_calls_per_run:
            return False, f"Tool call limit exceeded for agent {agent_id}"

        return True, "Permission granted"

    def record_tool_call(self, agent_id: str, tool_name: str):
        """Record a tool call for rate limiting."""
        if agent_id not in self._tool_call_counts:
            self._tool_call_counts[agent_id] = {}

        tool_counts = self._tool_call_counts[agent_id]
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

    def reset_agent_counts(self, agent_id: str):
        """Reset tool call counts for an agent (e.g., at start of new run)."""
        self._tool_call_counts.pop(agent_id, None)

    def requires_confirmation(self, tool_name: str) -> bool:
        """Check if a tool requires human confirmation."""
        tool = self.tools.get(tool_name)
        return tool.requires_confirmation if tool else True

    def get_allowed_tools(self, agent_type: AgentType) -> list[str]:
        """Get list of tools allowed for an agent type."""
        profile = self.agent_profiles.get(agent_type)
        if not profile:
            return []

        allowed = []
        for tool_name, tool in self.tools.items():
            # Check agent restriction
            if tool.allowed_agents and agent_type not in tool.allowed_agents:
                continue

            # Check risk level
            risk_order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
            if risk_order.index(tool.risk_level) > risk_order.index(profile.max_risk_level):
                continue

            # Check permissions
            if tool.required_permissions <= profile.allowed_permissions:
                allowed.append(tool_name)

        return allowed


# =============================================================================
# Global Instance
# =============================================================================

_permission_manager: Optional[PermissionManager] = None


def get_permission_manager() -> PermissionManager:
    """Get the global permission manager."""
    global _permission_manager
    if _permission_manager is None:
        _permission_manager = PermissionManager()
    return _permission_manager


# =============================================================================
# Decorators
# =============================================================================

def check_tool_permission(
    tool_name: str,
    agent_type: AgentType,
):
    """
    Decorator to check tool permissions before execution.

    Args:
        tool_name: Name of the tool
        agent_type: Type of agent using the tool
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, agent_id: str = "unknown", **kwargs) -> Any:
            manager = get_permission_manager()

            allowed, reason = manager.check_permission(
                tool_name=tool_name,
                agent_type=agent_type,
                agent_id=agent_id,
            )

            if not allowed:
                logger.warning(
                    f"Permission denied for tool {tool_name}",
                    metadata={
                        "tool": tool_name,
                        "agent_type": agent_type.value,
                        "agent_id": agent_id,
                        "reason": reason,
                    }
                )
                raise AuthorizationError(
                    message=f"Permission denied: {reason}",
                    required_permission=tool_name,
                )

            # Record the tool call
            manager.record_tool_call(agent_id, tool_name)

            logger.debug(
                f"Permission granted for tool {tool_name}",
                metadata={
                    "tool": tool_name,
                    "agent_type": agent_type.value,
                    "agent_id": agent_id,
                }
            )

            return func(*args, **kwargs)

        return wrapper
    return decorator


def require_confirmation(func: Callable) -> Callable:
    """
    Decorator marking a function as requiring human confirmation.

    The actual confirmation logic should be handled by the caller.
    This decorator just sets a flag for introspection.
    """
    func._requires_confirmation = True
    return func
