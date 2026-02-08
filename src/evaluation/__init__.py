"""
Archon Evaluation Module

Evaluation framework for multi-agent systems.
Includes LLM-as-judge, RAGAS metrics, and test sets.
"""

from src.evaluation.metrics import (
    evaluate_response,
    evaluate_retrieval,
    EvaluationResult,
)
from src.evaluation.judge import (
    LLMJudge,
    judge_response,
)

__all__ = [
    "evaluate_response",
    "evaluate_retrieval",
    "EvaluationResult",
    "LLMJudge",
    "judge_response",
]
