"""
Archon Retriever Module

Hybrid retrieval with BM25 and vector search.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from src.core.types import Chunk, RetrievalResult
from src.rag.embeddings import embed_text, cosine_similarity
from src.monitoring.logger import get_logger

logger = get_logger("rag.retriever")


# =============================================================================
# Retriever Configuration
# =============================================================================

@dataclass
class RetrieverConfig:
    """Configuration for retriever."""
    top_k: int = 10
    similarity_threshold: float = 0.7
    use_bm25: bool = True
    use_vector: bool = True
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    embedding_model: str = "text-embedding-3-small"


# =============================================================================
# BM25 Implementation
# =============================================================================

class BM25:
    """Simple BM25 implementation for keyword search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._documents: list[list[str]] = []
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0
        self._doc_freqs: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._n_docs: int = 0

    def fit(self, documents: list[str]):
        """Fit BM25 on documents."""
        import math

        self._documents = [doc.lower().split() for doc in documents]
        self._n_docs = len(documents)
        self._doc_lengths = [len(doc) for doc in self._documents]
        self._avg_doc_length = sum(self._doc_lengths) / max(self._n_docs, 1)

        # Calculate document frequencies
        for doc in self._documents:
            unique_terms = set(doc)
            for term in unique_terms:
                self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1

        # Calculate IDF
        for term, df in self._doc_freqs.items():
            self._idf[term] = math.log(
                (self._n_docs - df + 0.5) / (df + 0.5) + 1
            )

    def score(self, query: str) -> list[float]:
        """Score all documents against query."""
        query_terms = query.lower().split()
        scores = []

        for idx, doc in enumerate(self._documents):
            score = 0.0
            doc_len = self._doc_lengths[idx]

            for term in query_terms:
                if term not in self._idf:
                    continue

                tf = doc.count(term)
                idf = self._idf[term]

                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / self._avg_doc_length
                )

                score += idf * numerator / denominator

            scores.append(score)

        return scores


# =============================================================================
# Retriever
# =============================================================================

class Retriever:
    """
    Hybrid retriever combining BM25 and vector search.

    Supports:
    - Keyword (BM25) search
    - Vector (semantic) search
    - Hybrid fusion
    """

    def __init__(self, config: Optional[RetrieverConfig] = None):
        self.config = config or RetrieverConfig()
        self._chunks: list[Chunk] = []
        self._bm25: Optional[BM25] = None
        self._indexed: bool = False

    def index(self, chunks: list[Chunk]):
        """
        Index chunks for retrieval.

        Args:
            chunks: Chunks to index
        """
        self._chunks = chunks

        # Build BM25 index
        if self.config.use_bm25:
            self._bm25 = BM25()
            self._bm25.fit([c.content for c in chunks])

        self._indexed = True

        logger.info(
            f"Indexed {len(chunks)} chunks",
            metadata={"bm25": self.config.use_bm25, "vector": self.config.use_vector}
        )

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query
            top_k: Number of results (overrides config)
            filter_metadata: Filter by metadata

        Returns:
            List of RetrievalResult
        """
        if not self._indexed:
            logger.warning("Retriever not indexed")
            return []

        top_k = top_k or self.config.top_k
        results: list[RetrievalResult] = []

        # Apply metadata filter
        candidates = self._chunks
        if filter_metadata:
            candidates = self._filter_by_metadata(candidates, filter_metadata)

        if not candidates:
            return []

        # BM25 scores
        bm25_scores = {}
        if self.config.use_bm25 and self._bm25:
            scores = self._bm25.score(query)
            for i, chunk in enumerate(self._chunks):
                if chunk in candidates:
                    bm25_scores[chunk.id] = scores[i]

        # Vector scores
        vector_scores = {}
        if self.config.use_vector:
            query_embedding = await embed_text(query, self.config.embedding_model)

            for chunk in candidates:
                if chunk.embedding:
                    score = cosine_similarity(query_embedding, chunk.embedding)
                    vector_scores[chunk.id] = score

        # Combine scores
        final_scores = {}
        for chunk in candidates:
            bm25 = bm25_scores.get(chunk.id, 0.0)
            vector = vector_scores.get(chunk.id, 0.0)

            # Normalize BM25 scores
            max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
            if max_bm25 > 0:
                bm25 = bm25 / max_bm25

            # Weighted combination
            final = (
                self.config.bm25_weight * bm25 +
                self.config.vector_weight * vector
            )

            if final >= self.config.similarity_threshold:
                final_scores[chunk.id] = final

        # Sort and get top-k
        sorted_chunks = sorted(
            [(c, final_scores.get(c.id, 0)) for c in candidates],
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        # Create results
        for chunk, score in sorted_chunks:
            results.append(RetrievalResult(
                chunk=chunk,
                score=score,
            ))

        logger.debug(
            f"Retrieved {len(results)} chunks",
            metadata={"query": query[:50], "top_k": top_k}
        )

        return results

    def _filter_by_metadata(
        self,
        chunks: list[Chunk],
        filter_metadata: dict,
    ) -> list[Chunk]:
        """Filter chunks by metadata."""
        filtered = []

        for chunk in chunks:
            match = True
            for key, value in filter_metadata.items():
                if chunk.metadata.get(key) != value:
                    match = False
                    break

            if match:
                filtered.append(chunk)

        return filtered


# =============================================================================
# Global Retriever
# =============================================================================

_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """Get global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


async def retrieve(
    query: str,
    top_k: int = 10,
) -> list[RetrievalResult]:
    """Convenience function for retrieval."""
    retriever = get_retriever()
    return await retriever.retrieve(query, top_k)
