"""
Archon Reranker Module

Cross-encoder reranking for improved retrieval precision.
"""

from dataclasses import dataclass
from typing import Optional

from src.core.types import RetrievalResult
from src.monitoring.logger import get_logger

logger = get_logger("rag.reranker")


# =============================================================================
# Reranker Configuration
# =============================================================================

@dataclass
class RerankerConfig:
    """Configuration for reranker."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5
    score_threshold: float = 0.0
    use_gpu: bool = False


# =============================================================================
# Cross-Encoder Reranker
# =============================================================================

class CrossEncoderReranker:
    """
    Cross-encoder reranker for improved precision.

    Uses a cross-encoder model to score query-document pairs.
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()
        self._model = None
        self._initialized = False

    def _init_model(self):
        """Initialize the cross-encoder model."""
        if self._initialized:
            return

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.config.model_name,
                device="cuda" if self.config.use_gpu else "cpu",
            )
            self._initialized = True

            logger.info(
                f"Initialized reranker: {self.config.model_name}",
                metadata={"gpu": self.config.use_gpu}
            )

        except ImportError:
            logger.warning("sentence-transformers not installed, using fallback reranker")
            self._initialized = True

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """
        Rerank results using cross-encoder.

        Args:
            query: Search query
            results: Initial retrieval results
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        if not results:
            return []

        self._init_model()
        top_k = top_k or self.config.top_k

        if self._model is None:
            # Fallback: just return top_k by original score
            return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]

        # Create query-document pairs
        pairs = [(query, r.chunk.content) for r in results]

        # Score with cross-encoder
        scores = self._model.predict(pairs)

        # Update results with rerank scores
        for result, score in zip(results, scores):
            result.rerank_score = float(score)

        # Sort by rerank score
        reranked = sorted(
            results,
            key=lambda x: x.rerank_score or 0,
            reverse=True,
        )

        # Filter by threshold and return top_k
        filtered = [
            r for r in reranked
            if (r.rerank_score or 0) >= self.config.score_threshold
        ]

        final_results = filtered[:top_k]

        logger.debug(
            f"Reranked {len(results)} -> {len(final_results)} results",
            metadata={"query": query[:50]}
        )

        return final_results


# =============================================================================
# Cohere Reranker
# =============================================================================

class CohereReranker:
    """
    Cohere rerank API reranker.

    Uses Cohere's hosted reranking service.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "rerank-english-v2.0"):
        self.api_key = api_key
        self.model = model

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Rerank using Cohere API."""
        if not results:
            return []

        try:
            import cohere

            co = cohere.Client(self.api_key)

            documents = [r.chunk.content for r in results]

            response = co.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_n=top_k,
            )

            # Map back to results
            reranked = []
            for item in response.results:
                result = results[item.index]
                result.rerank_score = item.relevance_score
                reranked.append(result)

            return reranked

        except ImportError:
            logger.warning("cohere not installed, returning original order")
            return results[:top_k]


# =============================================================================
# Convenience Function
# =============================================================================

_reranker: Optional[CrossEncoderReranker] = None


def get_reranker() -> CrossEncoderReranker:
    """Get global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def rerank(
    query: str,
    results: list[RetrievalResult],
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Convenience function for reranking."""
    reranker = get_reranker()
    return reranker.rerank(query, results, top_k)
