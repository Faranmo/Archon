"""
Archon Writer Agent

Synthesizes information into polished content.
Single responsibility: Content generation.
"""

from typing import Optional

from src.agents.base import BaseAgent, AgentContext, AgentResult
from src.core.types import AgentType, AgentState
from src.prompts.manager import render_prompt
from src.monitoring.logger import get_logger

logger = get_logger("agents.writer")


class WriterAgent(BaseAgent):
    """
    Writer agent for content generation.

    Responsibilities:
    - Transform research into prose
    - Structure content logically
    - Ensure proper citations
    - Adapt to audience
    """

    def __init__(self, **kwargs):
        super().__init__(agent_type=AgentType.WRITER, **kwargs)

    @property
    def system_prompt(self) -> str:
        """Writer system prompt."""
        return render_prompt(
            "agent.writer",
            format="report",
            audience="general",
            tone="professional",
            length="medium",
            source_material="Source material will be provided.",
            task="Writing task will be specified.",
        )

    @property
    def available_tools(self) -> list[str]:
        """Writer has access to document tools."""
        return ["semantic_search", "read_document"]

    async def write(
        self,
        content: str,
        output_format: str = "report",
        audience: str = "general",
        tone: str = "professional",
        length: str = "medium",
        instructions: Optional[str] = None,
    ) -> dict:
        """
        Generate written content.

        Args:
            content: Source material to synthesize
            output_format: Output format (report, summary, article, etc.)
            audience: Target audience
            tone: Writing tone
            length: Desired length (short, medium, long)
            instructions: Additional writing instructions

        Returns:
            Generated content
        """
        task = render_prompt(
            "agent.writer",
            format=output_format,
            audience=audience,
            tone=tone,
            length=length,
            source_material=content,
            task=instructions or f"Create a {output_format} from the source material.",
        )

        # Run writer
        result = await self.run(task)

        if not result.success:
            return {
                "success": False,
                "error": result.error,
            }

        return {
            "success": True,
            "content": result.output,
            "format": output_format,
            "word_count": len(result.output.split()),
            "tokens_used": result.tokens_used,
            "cost_usd": result.cost_usd,
        }

    async def summarize(
        self,
        content: str,
        max_length: int = 500,
    ) -> dict:
        """
        Generate a summary.

        Args:
            content: Content to summarize
            max_length: Maximum words

        Returns:
            Summary
        """
        return await self.write(
            content=content,
            output_format="summary",
            length="short",
            instructions=f"Summarize the key points in {max_length} words or less.",
        )

    async def create_report(
        self,
        research_findings: dict,
        title: Optional[str] = None,
    ) -> dict:
        """
        Create a research report from findings.

        Args:
            research_findings: Research output from ResearcherAgent
            title: Optional report title

        Returns:
            Formatted report
        """
        import json

        content = json.dumps(research_findings, indent=2)

        instructions = f"""Create a comprehensive research report.
{"Title: " + title if title else ""}

Include:
- Executive Summary
- Key Findings
- Detailed Analysis
- Sources and Citations
- Conclusions and Recommendations
"""

        return await self.write(
            content=content,
            output_format="report",
            length="long",
            instructions=instructions,
        )
