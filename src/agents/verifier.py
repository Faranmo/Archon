"""
Archon Verifier Agent

Validates outputs and ensures quality.
Single responsibility: Quality assurance.
"""

from typing import Any, Optional

from src.agents.base import BaseAgent, AgentContext, AgentResult
from src.core.types import AgentType, AgentState
from src.prompts.manager import render_prompt
from src.monitoring.logger import get_logger

logger = get_logger("agents.verifier")


class VerifierAgent(BaseAgent):
    """
    Verifier agent for quality assurance.

    Responsibilities:
    - Check factual accuracy
    - Verify logical consistency
    - Identify errors and gaps
    - Assess overall quality
    """

    def __init__(self, **kwargs):
        super().__init__(agent_type=AgentType.VERIFIER, **kwargs)

    @property
    def system_prompt(self) -> str:
        """Verifier system prompt."""
        return render_prompt(
            "agent.verifier",
            content="Content will be provided.",
            original_request="Original request will be provided.",
            criteria="Standard verification criteria.",
        )

    @property
    def available_tools(self) -> list[str]:
        """Verifier has access to search for fact-checking."""
        return ["web_search", "semantic_search"]

    async def verify(
        self,
        content: str,
        original_request: str,
        criteria: Optional[list[str]] = None,
    ) -> dict:
        """
        Verify content quality and accuracy.

        Args:
            content: Content to verify
            original_request: The original user request
            criteria: Specific verification criteria

        Returns:
            Verification report
        """
        criteria_str = ""
        if criteria:
            criteria_str = "\n".join(f"- {c}" for c in criteria)
        else:
            criteria_str = """
- Factual accuracy
- Logical consistency
- Completeness
- Proper citations
- Clear communication
"""

        task = render_prompt(
            "agent.verifier",
            content=content,
            original_request=original_request,
            criteria=criteria_str,
        )

        # Run verification
        result = await self.run(task)

        if not result.success:
            return {
                "success": False,
                "error": result.error,
            }

        # Parse verification report
        report = self._parse_report(result.output)

        return {
            "success": True,
            "passed": report.get("passed", False),
            "confidence": report.get("confidence", 0.0),
            "factual_checks": report.get("factual", []),
            "issues": report.get("issues", []),
            "recommendations": report.get("recommendations", []),
            "raw_output": result.output,
            "tokens_used": result.tokens_used,
            "cost_usd": result.cost_usd,
        }

    def _parse_report(self, output: str) -> dict:
        """Parse verification report."""
        report = {
            "passed": False,
            "confidence": 0.5,
            "factual": [],
            "issues": [],
            "recommendations": [],
        }

        lines = output.split("\n")

        for line in lines:
            line = line.strip()
            lower = line.lower()

            # Check overall assessment
            if "assessment:" in lower:
                if "pass" in lower:
                    report["passed"] = True
                elif "fail" in lower:
                    report["passed"] = False

            # Check confidence
            if "confidence:" in lower:
                import re
                match = re.search(r"(\d+(?:\.\d+)?)\s*%?", line)
                if match:
                    conf = float(match.group(1))
                    if conf > 1:
                        conf /= 100
                    report["confidence"] = conf

            # Parse issues
            if "issue" in lower and ":" in line:
                issue = line.split(":", 1)[1].strip()
                report["issues"].append(issue)

            # Parse recommendations
            if "recommend" in lower and ":" in line:
                rec = line.split(":", 1)[1].strip()
                report["recommendations"].append(rec)

        return report

    async def fact_check(
        self,
        claims: list[str],
    ) -> dict:
        """
        Fact-check a list of claims.

        Args:
            claims: Claims to verify

        Returns:
            Fact-check results
        """
        task = f"""Fact-check the following claims:

{chr(10).join(f'{i+1}. {claim}' for i, claim in enumerate(claims))}

For each claim:
1. Search for supporting/contradicting evidence
2. Rate as VERIFIED, UNVERIFIED, or FALSE
3. Provide sources

Format your response as:
CLAIM 1: [VERIFIED/UNVERIFIED/FALSE]
Evidence: [source and details]
...
"""

        result = await self.run(task)

        if not result.success:
            return {"success": False, "error": result.error}

        return {
            "success": True,
            "results": self._parse_fact_checks(result.output, claims),
            "raw_output": result.output,
        }

    def _parse_fact_checks(self, output: str, claims: list[str]) -> list[dict]:
        """Parse fact-check results."""
        import re

        results = []
        lines = output.split("\n")

        for i, claim in enumerate(claims):
            result = {
                "claim": claim,
                "status": "unverified",
                "evidence": None,
            }

            # Look for claim result
            pattern = rf"CLAIM\s*{i+1}\s*:\s*(VERIFIED|UNVERIFIED|FALSE)"
            for line in lines:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    result["status"] = match.group(1).lower()
                    break

            results.append(result)

        return results
