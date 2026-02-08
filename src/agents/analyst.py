"""
Archon Analyst Agent

Analyzes data and extracts insights.
Single responsibility: Data analysis.
"""

from typing import Any, Optional

from src.agents.base import BaseAgent, AgentContext, AgentResult
from src.core.types import AgentType, AgentState
from src.prompts.manager import render_prompt
from src.monitoring.logger import get_logger

logger = get_logger("agents.analyst")


class AnalystAgent(BaseAgent):
    """
    Analyst agent for data analysis.

    Responsibilities:
    - Analyze data and information
    - Identify patterns and trends
    - Draw conclusions from evidence
    - Quantify confidence in analyses
    """

    def __init__(self, **kwargs):
        super().__init__(agent_type=AgentType.ANALYST, **kwargs)

    @property
    def system_prompt(self) -> str:
        """Analyst system prompt."""
        return render_prompt(
            "agent.analyst",
            data="Data will be provided.",
            questions="Analysis questions will be provided.",
        )

    @property
    def available_tools(self) -> list[str]:
        """Analyst has limited tools - focuses on analysis."""
        return ["semantic_search"]

    async def analyze(
        self,
        data: Any,
        questions: Optional[list[str]] = None,
        context: Optional[str] = None,
    ) -> dict:
        """
        Analyze data and answer questions.

        Args:
            data: Data to analyze (dict, list, or string)
            questions: Specific questions to answer
            context: Additional context

        Returns:
            Analysis results
        """
        # Format data
        if isinstance(data, (dict, list)):
            import json
            data_str = json.dumps(data, indent=2)
        else:
            data_str = str(data)

        # Build questions
        questions_str = ""
        if questions:
            questions_str = "\n".join(f"- {q}" for q in questions)
        else:
            questions_str = "Provide a comprehensive analysis."

        task = render_prompt(
            "agent.analyst",
            data=data_str,
            questions=questions_str,
        )

        if context:
            task += f"\n\nAdditional Context: {context}"

        # Run analysis
        result = await self.run(task)

        if not result.success:
            return {
                "success": False,
                "error": result.error,
            }

        # Parse analysis
        analysis = self._parse_analysis(result.output)

        return {
            "success": True,
            "key_findings": analysis.get("findings", []),
            "patterns": analysis.get("patterns", []),
            "conclusions": analysis.get("conclusions", []),
            "limitations": analysis.get("limitations", []),
            "recommendations": analysis.get("recommendations", []),
            "raw_output": result.output,
            "tokens_used": result.tokens_used,
            "cost_usd": result.cost_usd,
        }

    def _parse_analysis(self, output: str) -> dict:
        """Parse analysis from output."""
        analysis = {
            "findings": [],
            "patterns": [],
            "conclusions": [],
            "limitations": [],
            "recommendations": [],
        }

        current_section = None
        lines = output.split("\n")

        section_keywords = {
            "findings": ["finding", "key finding"],
            "patterns": ["pattern"],
            "conclusions": ["conclusion"],
            "limitations": ["limitation", "caveat"],
            "recommendations": ["recommendation"],
        }

        for line in lines:
            line = line.strip()
            if not line:
                continue

            lower = line.lower()

            # Check for section headers
            for section, keywords in section_keywords.items():
                if any(kw in lower for kw in keywords) and ":" in line:
                    current_section = section
                    break

            # Add content to current section
            if current_section and (line.startswith("-") or line.startswith("*") or line[0].isdigit()):
                clean = line.lstrip("-*0123456789. ")
                analysis[current_section].append(clean)

        return analysis
