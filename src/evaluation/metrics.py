"""
Archon Evaluation Metrics

Metrics for evaluating RAG and agent performance.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from src.core.types import RetrievalResult
from src.monitoring.logger import get_logger

logger = get_logger("evaluation.metrics")


class MetricType(Enum):
    """Types of evaluation metrics."""
    RELEVANCE = "relevance"
    FAITHFULNESS = "faithfulness"
    COHERENCE = "coherence"
    COMPLETENESS = "completeness"
    CONCISENESS = "conciseness"


@dataclass
class EvaluationResult:
    """Result of an evaluation."""
    metric: MetricType
    score: float  # 0.0 to 1.0
    reasoning: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class RAGEvaluationResult:
    """Comprehensive RAG evaluation result."""
    context_relevance: float
    answer_relevance: float
    faithfulness: float
    context_precision: float
    context_recall: float
    overall_score: float
    details: dict = field(default_factory=dict)


def evaluate_response(
    question: str,
    answer: str,
    reference: Optional[str] = None,
    contexts: Optional[list[str]] = None,
) -> dict[MetricType, EvaluationResult]:
    """
    Evaluate a response across multiple metrics.

    Args:
        question: Original question
        answer: Generated answer
        reference: Reference answer (if available)
        contexts: Retrieved contexts

    Returns:
        Dictionary of metric results
    """
    results = {}

    # Relevance: Does the answer address the question?
    results[MetricType.RELEVANCE] = _evaluate_relevance(question, answer)

    # Coherence: Is the answer well-structured?
    results[MetricType.COHERENCE] = _evaluate_coherence(answer)

    # Completeness: Does it fully answer the question?
    results[MetricType.COMPLETENESS] = _evaluate_completeness(question, answer)

    # Faithfulness: Is it grounded in the context?
    if contexts:
        results[MetricType.FAITHFULNESS] = _evaluate_faithfulness(answer, contexts)

    return results


def _evaluate_relevance(question: str, answer: str) -> EvaluationResult:
    """Evaluate answer relevance to question."""
    # Simple heuristic - check for keyword overlap
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())

    overlap = len(q_words & a_words) / max(len(q_words), 1)
    score = min(1.0, overlap * 2)  # Scale up

    return EvaluationResult(
        metric=MetricType.RELEVANCE,
        score=score,
        reasoning=f"Keyword overlap: {overlap:.2f}",
    )


def _evaluate_coherence(answer: str) -> EvaluationResult:
    """Evaluate answer coherence."""
    # Check for basic structure indicators
    has_structure = any([
        answer.count('.') >= 2,  # Multiple sentences
        '\n' in answer,  # Line breaks
        ':' in answer,  # Lists or definitions
    ])

    score = 0.8 if has_structure else 0.5

    return EvaluationResult(
        metric=MetricType.COHERENCE,
        score=score,
        reasoning="Basic structure check",
    )


def _evaluate_completeness(question: str, answer: str) -> EvaluationResult:
    """Evaluate answer completeness."""
    # Check answer length relative to question complexity
    q_complexity = len(question.split())
    a_length = len(answer.split())

    ratio = a_length / max(q_complexity * 3, 1)  # Expect 3x expansion
    score = min(1.0, ratio)

    return EvaluationResult(
        metric=MetricType.COMPLETENESS,
        score=score,
        reasoning=f"Answer length ratio: {ratio:.2f}",
    )


def _evaluate_faithfulness(answer: str, contexts: list[str]) -> EvaluationResult:
    """Evaluate if answer is grounded in contexts."""
    combined_context = " ".join(contexts).lower()
    answer_words = set(answer.lower().split())
    context_words = set(combined_context.split())

    # Check how many answer words appear in context
    grounded = len(answer_words & context_words) / max(len(answer_words), 1)

    return EvaluationResult(
        metric=MetricType.FAITHFULNESS,
        score=grounded,
        reasoning=f"Word grounding: {grounded:.2f}",
    )


def evaluate_retrieval(
    query: str,
    results: list[RetrievalResult],
    relevant_docs: Optional[list[str]] = None,
) -> dict:
    """
    Evaluate retrieval quality.

    Args:
        query: Search query
        results: Retrieved results
        relevant_docs: Ground truth relevant document IDs

    Returns:
        Retrieval metrics
    """
    metrics = {
        "num_results": len(results),
        "avg_score": sum(r.score for r in results) / max(len(results), 1),
        "max_score": max((r.score for r in results), default=0),
        "min_score": min((r.score for r in results), default=0),
    }

    # Calculate precision/recall if ground truth available
    if relevant_docs:
        retrieved_ids = {r.chunk.document_id for r in results}
        relevant_set = set(relevant_docs)

        true_positives = len(retrieved_ids & relevant_set)
        precision = true_positives / max(len(retrieved_ids), 1)
        recall = true_positives / max(len(relevant_set), 1)

        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = 2 * precision * recall / max(precision + recall, 0.001)

    return metrics
