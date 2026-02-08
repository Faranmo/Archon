"""
Archon Embedding Module

Text embedding generation for semantic search.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from src.core.config import settings
from src.core.types import Chunk
from src.monitoring.logger import get_logger
from src.monitoring.cost_tracker import track_usage

logger = get_logger("rag.embeddings")


# =============================================================================
# Embedding Models
# =============================================================================

class EmbeddingProvider(Enum):
    """Available embedding providers."""
    OPENAI = "openai"
    COHERE = "cohere"
    LOCAL = "local"


@dataclass
class EmbeddingModel:
    """Embedding model configuration."""
    provider: EmbeddingProvider
    model_name: str
    dimensions: int
    max_tokens: int = 8192
    batch_size: int = 100


# Default models
EMBEDDING_MODELS = {
    "text-embedding-3-small": EmbeddingModel(
        provider=EmbeddingProvider.OPENAI,
        model_name="text-embedding-3-small",
        dimensions=1536,
    ),
    "text-embedding-3-large": EmbeddingModel(
        provider=EmbeddingProvider.OPENAI,
        model_name="text-embedding-3-large",
        dimensions=3072,
    ),
    "text-embedding-ada-002": EmbeddingModel(
        provider=EmbeddingProvider.OPENAI,
        model_name="text-embedding-ada-002",
        dimensions=1536,
    ),
}


# =============================================================================
# Embedding Functions
# =============================================================================

async def embed_text(
    text: str,
    model: str = "text-embedding-3-small",
) -> list[float]:
    """
    Generate embedding for a single text.

    Args:
        text: Text to embed
        model: Embedding model name

    Returns:
        Embedding vector
    """
    model_config = EMBEDDING_MODELS.get(model)

    if not model_config or model_config.provider == EmbeddingProvider.OPENAI:
        return await _embed_openai([text], model)
    else:
        raise ValueError(f"Unsupported embedding model: {model}")


async def embed_texts(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> list[list[float]]:
    """
    Generate embeddings for multiple texts.

    Args:
        texts: Texts to embed
        model: Embedding model name
        batch_size: Batch size for API calls

    Returns:
        List of embedding vectors
    """
    model_config = EMBEDDING_MODELS.get(model)

    if not model_config or model_config.provider == EmbeddingProvider.OPENAI:
        # Process in batches
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = await _embed_openai(batch, model)
            all_embeddings.extend(embeddings)

        return all_embeddings

    else:
        raise ValueError(f"Unsupported embedding model: {model}")


async def embed_chunks(
    chunks: list[Chunk],
    model: str = "text-embedding-3-small",
) -> list[Chunk]:
    """
    Generate embeddings for chunks and attach to them.

    Args:
        chunks: Chunks to embed
        model: Embedding model name

    Returns:
        Chunks with embeddings attached
    """
    texts = [chunk.content for chunk in chunks]
    embeddings = await embed_texts(texts, model)

    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding

    logger.info(
        f"Embedded {len(chunks)} chunks",
        metadata={"model": model}
    )

    return chunks


# =============================================================================
# Provider Implementations
# =============================================================================

async def _embed_openai(
    texts: list[str],
    model: str = "text-embedding-3-small",
) -> list[list[float]]:
    """Generate embeddings using OpenAI."""
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.openai_api_key)

        response = await client.embeddings.create(
            model=model,
            input=texts,
        )

        # Track usage
        track_usage(
            model=model,
            input_tokens=response.usage.total_tokens,
            output_tokens=0,
        )

        # Extract embeddings in order
        embeddings = [None] * len(texts)
        for item in response.data:
            embeddings[item.index] = item.embedding

        return embeddings

    except ImportError:
        raise ImportError("openai package not installed")
    except Exception as e:
        logger.error(f"OpenAI embedding failed: {e}")
        raise


# =============================================================================
# Utility Functions
# =============================================================================

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    import math

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def normalize_embedding(embedding: list[float]) -> list[float]:
    """Normalize embedding to unit vector."""
    import math

    norm = math.sqrt(sum(x * x for x in embedding))
    if norm == 0:
        return embedding

    return [x / norm for x in embedding]
