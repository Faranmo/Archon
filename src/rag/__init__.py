"""
Archon RAG Module

Retrieval-Augmented Generation pipeline with:
- Semantic chunking
- Hybrid search (BM25 + vector)
- Cross-encoder reranking
- Contextual retrieval
"""

from src.rag.chunking import (
    chunk_document,
    SemanticChunker,
)
from src.rag.embeddings import (
    embed_text,
    embed_chunks,
    EmbeddingModel,
)
from src.rag.retriever import (
    Retriever,
    retrieve,
)
from src.rag.reranker import (
    rerank,
    CrossEncoderReranker,
)

__all__ = [
    "chunk_document",
    "SemanticChunker",
    "embed_text",
    "embed_chunks",
    "EmbeddingModel",
    "Retriever",
    "retrieve",
    "rerank",
    "CrossEncoderReranker",
]
