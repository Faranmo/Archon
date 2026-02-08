"""
Archon LLM Judge

Uses LLM to evaluate response quality.
"""

from dataclasses import dataclass
from typing import Optional

from src.llm.client import get_llm_client
from src.core.types import Message, MessageRole
from src.monitoring.logger import get_logger

logger = get_logger("evaluation.judge")


@dataclass
class JudgmentResult:
    """Result from LLM judge."""
    score: float  # 0.0 to 1.0
    reasoning: str
    criteria_scores: dict[str, float]


JUDGE_PROMPT = """You are an expert evaluator of AI responses.

Evaluate the following response based on these criteria:
1. Relevance (0-10): Does it address the question?
2. Accuracy (0-10): Is the information correct?
3. Completeness (0-10): Is it thorough?
4. Clarity (0-10): Is it well-written?

Question: {question}

Response: {response}

{context_section}

Provide your evaluation in this format:
RELEVANCE: [score]/10 - [brief reason]
ACCURACY: [score]/10 - [brief reason]
COMPLETENESS: [score]/10 - [brief reason]
CLARITY: [score]/10 - [brief reason]

OVERALL: [score]/10
REASONING: [overall assessment]
"""


class LLMJudge:
    """
    Uses LLM to judge response quality.

    More sophisticated than rule-based metrics,
    captures nuance in language quality.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = get_llm_client()

    async def judge(
        self,
        question: str,
        response: str,
        context: Optional[list[str]] = None,
        reference: Optional[str] = None,
    ) -> JudgmentResult:
        """
        Judge a response using LLM.

        Args:
            question: Original question
            response: Response to evaluate
            context: Retrieved contexts
            reference: Reference answer

        Returns:
            JudgmentResult with scores and reasoning
        """
        context_section = ""
        if context:
            context_section = f"Context provided:\n{chr(10).join(context[:3])}"

        prompt = JUDGE_PROMPT.format(
            question=question,
            response=response,
            context_section=context_section,
        )

        messages = [
            Message(role=MessageRole.USER, content=prompt),
        ]

        result = await self.client.complete(messages, model=self.model)

        # Parse the judgment
        return self._parse_judgment(result.content)

    def _parse_judgment(self, text: str) -> JudgmentResult:
        """Parse LLM judgment output."""
        import re

        criteria_scores = {}

        # Parse individual scores
        for criterion in ["RELEVANCE", "ACCURACY", "COMPLETENESS", "CLARITY"]:
            match = re.search(rf"{criterion}:\s*(\d+)/10", text, re.IGNORECASE)
            if match:
                criteria_scores[criterion.lower()] = int(match.group(1)) / 10

        # Parse overall score
        overall_match = re.search(r"OVERALL:\s*(\d+)/10", text, re.IGNORECASE)
        overall = int(overall_match.group(1)) / 10 if overall_match else 0.5

        # Parse reasoning
        reasoning_match = re.search(r"REASONING:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        return JudgmentResult(
            score=overall,
            reasoning=reasoning,
            criteria_scores=criteria_scores,
        )


async def judge_response(
    question: str,
    response: str,
    **kwargs,
) -> JudgmentResult:
    """Convenience function for judging a response."""
    judge = LLMJudge()
    return await judge.judge(question, response, **kwargs)
