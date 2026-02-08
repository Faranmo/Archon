"""
Tests for RAG module.
"""

import pytest
from src.rag.chunking import (
    chunk_by_tokens,
    chunk_by_sentences,
    chunk_by_paragraphs,
    SemanticChunker,
    ChunkConfig,
)
from src.rag.embeddings import cosine_similarity, normalize_embedding
from src.rag.retriever import BM25


class TestChunking:
    """Test document chunking."""

    def test_chunk_by_tokens_basic(self):
        """Test basic token chunking."""
        text = " ".join(["word"] * 100)
        chunks = chunk_by_tokens(text, chunk_size=50, overlap=0)
        assert len(chunks) > 1

    def test_chunk_by_sentences(self):
        """Test sentence-based chunking."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_by_sentences(text, max_chunk_size=50, min_chunk_size=10)
        assert len(chunks) >= 1

    def test_chunk_by_paragraphs(self):
        """Test paragraph-based chunking."""
        text = """First paragraph with some content.

Second paragraph with more content.

Third paragraph."""
        chunks = chunk_by_paragraphs(text)
        assert len(chunks) >= 1

    def test_semantic_chunker(self, sample_document):
        """Test semantic chunker with document."""
        chunker = SemanticChunker()
        chunks = chunker.chunk(sample_document)
        assert len(chunks) > 0
        assert all(c.document_id == sample_document.id for c in chunks)


class TestEmbeddings:
    """Test embedding utilities."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        vec = [1.0, 2.0, 3.0]
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.0001

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim) < 0.0001

    def test_normalize_embedding(self):
        """Test embedding normalization."""
        vec = [3.0, 4.0]
        normalized = normalize_embedding(vec)
        # Should have unit length
        length = sum(x * x for x in normalized) ** 0.5
        assert abs(length - 1.0) < 0.0001


class TestBM25:
    """Test BM25 implementation."""

    def test_bm25_fit(self):
        """Test BM25 fitting."""
        bm25 = BM25()
        docs = ["hello world", "world is great", "hello again"]
        bm25.fit(docs)
        assert bm25._n_docs == 3

    def test_bm25_score(self):
        """Test BM25 scoring."""
        bm25 = BM25()
        docs = ["python programming", "java programming", "python is fun"]
        bm25.fit(docs)
        scores = bm25.score("python")
        # Doc 0 and Doc 2 should score higher than Doc 1
        assert scores[0] > scores[1]
        assert scores[2] > scores[1]
