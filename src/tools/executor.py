"""
Archon Tool Executor

Executes tools with permission checking, logging, and error handling.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Optional

from src.core.types import ToolCall, ToolResult as CoreToolResult, AgentType
from src.core.errors import ToolError, AuthorizationError
from src.tools.base import Tool, ToolResult, get_tool_registry
from src.security.permissions import get_permission_manager
from src.monitoring.logger import get_logger

logger = get_logger("tools.executor")


# =============================================================================
# Tool Executor
# =============================================================================

class ToolExecutor:
    """
    Executes tools with full lifecycle management.

    Features:
    - Permission checking
    - Execution logging
    - Error handling
    - Timeout management
    - Result caching
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 2,
        cache_results: bool = False,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_results = cache_results

        self.registry = get_tool_registry()
        self.permission_manager = get_permission_manager()

        # Simple result cache
        self._cache: dict[str, ToolResult] = {}

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        agent_type: Optional[AgentType] = None,
        agent_id: str = "system",
        check_permissions: bool = True,
    ) -> ToolResult:
        """
        Execute a tool.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            agent_type: Type of calling agent
            agent_id: ID of calling agent
            check_permissions: Whether to check permissions

        Returns:
            ToolResult with execution outcome
        """
        start_time = time.time()

        # Get tool from registry
        tool = self.registry.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool not found: {tool_name}",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Check permissions
        if check_permissions and agent_type:
            allowed, reason = self.permission_manager.check_permission(
                tool_name=tool_name,
                agent_type=agent_type,
                agent_id=agent_id,
            )

            if not allowed:
                logger.warning(
                    f"Permission denied for tool: {tool_name}",
                    metadata={"agent_id": agent_id, "reason": reason}
                )
                return ToolResult(
                    success=False,
                    error=f"Permission denied: {reason}",
                    latency_ms=(time.time() - start_time) * 1000,
                )

        # Check cache
        if self.cache_results:
            cache_key = f"{tool_name}:{hash(str(sorted(arguments.items())))}"
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                cached.metadata["cached"] = True
                return cached

        # Execute with timeout and retries
        for attempt in range(self.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    tool.execute(**arguments),
                    timeout=self.timeout,
                )

                # Log execution
                logger.log_tool_call(
                    tool_name=tool_name,
                    latency_ms=result.latency_ms,
                    success=result.success,
                    tool_input=arguments,
                    tool_output=str(result.data)[:200] if result.data else None,
                    error=result.error,
                    agent_id=agent_id,
                )

                # Cache successful results
                if self.cache_results and result.success:
                    self._cache[cache_key] = result

                return result

            except asyncio.TimeoutError:
                if attempt < self.max_retries:
                    logger.warning(
                        f"Tool timeout, retrying: {tool_name}",
                        metadata={"attempt": attempt + 1}
                    )
                    continue

                return ToolResult(
                    success=False,
                    error=f"Tool execution timed out after {self.timeout}s",
                    latency_ms=(time.time() - start_time) * 1000,
                )

            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(
                        f"Tool error, retrying: {tool_name}",
                        metadata={"attempt": attempt + 1, "error": str(e)}
                    )
                    continue

                return ToolResult(
                    success=False,
                    error=str(e),
                    latency_ms=(time.time() - start_time) * 1000,
                )

        # Should not reach here
        return ToolResult(
            success=False,
            error="Unknown error during tool execution",
            latency_ms=(time.time() - start_time) * 1000,
        )

    async def execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        agent_type: Optional[AgentType] = None,
        agent_id: str = "system",
        parallel: bool = True,
    ) -> list[CoreToolResult]:
        """
        Execute multiple tool calls.

        Args:
            tool_calls: List of ToolCall objects
            agent_type: Calling agent type
            agent_id: Calling agent ID
            parallel: Execute in parallel if True

        Returns:
            List of ToolResult objects
        """
        if parallel:
            # Execute all tools in parallel
            tasks = [
                self.execute(
                    tool_name=tc.name,
                    arguments=tc.arguments,
                    agent_type=agent_type,
                    agent_id=agent_id,
                )
                for tc in tool_calls
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Execute sequentially
            results = []
            for tc in tool_calls:
                result = await self.execute(
                    tool_name=tc.name,
                    arguments=tc.arguments,
                    agent_type=agent_type,
                    agent_id=agent_id,
                )
                results.append(result)

        # Convert to CoreToolResult format
        core_results = []
        for tc, result in zip(tool_calls, results):
            if isinstance(result, Exception):
                core_results.append(CoreToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    result=str(result),
                    success=False,
                    error=str(result),
                ))
            else:
                core_results.append(CoreToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    result=result.data if result.success else result.error,
                    success=result.success,
                    error=result.error,
                    latency_ms=result.latency_ms,
                ))

        return core_results

    def clear_cache(self):
        """Clear the result cache."""
        self._cache.clear()


# =============================================================================
# Global Instance
# =============================================================================

_tool_executor: Optional[ToolExecutor] = None


def get_tool_executor() -> ToolExecutor:
    """Get the global tool executor."""
    global _tool_executor
    if _tool_executor is None:
        _tool_executor = ToolExecutor()
    return _tool_executor


async def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    **kwargs,
) -> ToolResult:
    """Convenience function to execute a tool."""
    executor = get_tool_executor()
    return await executor.execute(tool_name, arguments, **kwargs)
