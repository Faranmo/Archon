"""
Archon Researcher Agent

Gathers and verifies information.
Single responsibility: Information retrieval.
"""

from typing import Optional

from src.agents.base import BaseAgent, AgentContext, AgentResult
from src.core.types import AgentType, AgentState
from src.prompts.manager import render_prompt
from src.monitoring.logger import get_logger

logger = get_logger("agents.researcher")


class ResearcherAgent(BaseAgent):
    """
    Researcher agent for information gathering.

    Responsibilities:
    - Search for relevant information
    - Verify facts from multiple sources
    - Document findings with sources
    - Track confidence levels
    """

    def __init__(self, **kwargs):
        super().__init__(agent_type=AgentType.RESEARCHER, **kwargs)

    @property
    def system_prompt(self) -> str:
        """Researcher system prompt."""
        tools_desc = ", ".join(self.available_tools)
        return render_prompt(
            "agent.researcher",
            available_tools=tools_desc,
            task_description="Research and gather information.",
        )

    @property
    def available_tools(self) -> list[str]:
        """Researcher has access to search tools."""
        return [
            "web_search",
            "semantic_search",
            "fetch_url",
            "read_document",
        ]

    async def research(
        self,
        topic: str,
        depth: str = "moderate",
        sources: Optional[list[str]] = None,
    ) -> dict:
        """
        Research a topic and return findings.

        Args:
            topic: The research topic
            depth: Research depth (quick, moderate, thorough)
            sources: Specific sources to check

        Returns:
            Research findings dictionary
        """
        # Build research task
        task = f"""Research the following topic:

Topic: {topic}

Research Depth: {depth}

{f"Specific sources to check: {', '.join(sources)}" if sources else "Use available search tools to find relevant sources."}

Provide your findings with sources and confidence levels.
"""

        # Run research
        result = await self.run(task)

        if not result.success:
            return {
                "success": False,
                "error": result.error,
                "findings": [],
            }

        # Parse findings
        findings = self._parse_findings(result.output)

        return {
            "success": True,
            "topic": topic,
            "findings": findings["findings"],
            "sources": findings["sources"],
            "gaps": findings.get("gaps", []),
            "raw_output": result.output,
            "tokens_used": result.tokens_used,
            "cost_usd": result.cost_usd,
        }

    def _parse_findings(self, output: str) -> dict:
        """Parse research findings from output."""
        findings = {
            "findings": [],
            "sources": [],
            "gaps": [],
        }

        current_section = None
        lines = output.split("\n")

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # Detect sections
            upper_line = line.upper()
            if "FINDINGS:" in upper_line:
                current_section = "findings"
            elif "SOURCES:" in upper_line:
                current_section = "sources"
            elif "GAPS:" in upper_line:
                current_section = "gaps"
            elif current_section and (line.startswith("-") or line.startswith("*") or line[0].isdigit()):
                # Clean the line
                clean = line.lstrip("-*0123456789. ")

                if current_section == "findings":
                    # Parse: "Finding text - Source: X - Confidence: Y"
                    finding = self._parse_finding(clean)
                    findings["findings"].append(finding)
                elif current_section == "sources":
                    findings["sources"].append(clean)
                elif current_section == "gaps":
                    findings["gaps"].append(clean)

        return findings

    def _parse_finding(self, text: str) -> dict:
        """Parse a single finding."""
        import re

        result = {
            "text": text,
            "source": None,
            "confidence": "medium",
        }

        # Try to extract source
        source_match = re.search(r"Source:\s*(.+?)(?:\s*-|$)", text, re.IGNORECASE)
        if source_match:
            result["source"] = source_match.group(1).strip()
            result["text"] = text[:source_match.start()].strip()

        # Try to extract confidence
        conf_match = re.search(r"Confidence:\s*(high|medium|low)", text, re.IGNORECASE)
        if conf_match:
            result["confidence"] = conf_match.group(1).lower()

        return result
