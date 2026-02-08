"""
Archon Document Chunking

Semantic chunking strategies for optimal retrieval.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

from src.core.types import Document, Chunk, DocumentType
from src.monitoring.logger import get_logger

logger = get_logger("rag.chunking")


# =============================================================================
# Chunking Configuration
# =============================================================================

@dataclass
class ChunkConfig:
    """Configuration for chunking."""
    chunk_size: int = 512  # Target chunk size in tokens (approx)
    chunk_overlap: int = 50  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum chunk size
    max_chunk_size: int = 1024  # Maximum chunk size
    preserve_sentences: bool = True  # Don't break mid-sentence
    preserve_paragraphs: bool = True  # Try to keep paragraphs together
    include_headers: bool = True  # Include section headers in chunks


# =============================================================================
# Chunking Strategies
# =============================================================================

def chunk_by_tokens(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[str]:
    """
    Simple chunking by approximate token count.

    Uses word count as proxy (avg 1.3 tokens per word).
    """
    words = text.split()
    words_per_chunk = int(chunk_size / 1.3)
    overlap_words = int(overlap / 1.3)

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start = end - overlap_words
        if start >= len(words) - overlap_words:
            break

    return chunks


def chunk_by_sentences(
    text: str,
    max_chunk_size: int = 512,
    min_chunk_size: int = 100,
) -> list[str]:
    """
    Chunk by sentences, respecting size limits.
    """
    # Split into sentences
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)

    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_size = len(sentence.split()) * 1.3  # Approx tokens

        if current_size + sentence_size > max_chunk_size and current_chunk:
            # Save current chunk
            chunk_text = " ".join(current_chunk)
            if len(chunk_text.split()) * 1.3 >= min_chunk_size:
                chunks.append(chunk_text)
            current_chunk = []
            current_size = 0

        current_chunk.append(sentence)
        current_size += sentence_size

    # Don't forget last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if len(chunk_text.split()) * 1.3 >= min_chunk_size:
            chunks.append(chunk_text)
        elif chunks:
            # Merge with previous if too small
            chunks[-1] = chunks[-1] + " " + chunk_text

    return chunks


def chunk_by_paragraphs(
    text: str,
    max_chunk_size: int = 1024,
    min_chunk_size: int = 100,
) -> list[str]:
    """
    Chunk by paragraphs, merging small ones.
    """
    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_size = len(para.split()) * 1.3

        if current_size + para_size > max_chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_size = 0

        current_chunk.append(para)
        current_size += para_size

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


# =============================================================================
# Semantic Chunker
# =============================================================================

class SemanticChunker:
    """
    Advanced semantic chunking.

    Uses document structure to create meaningful chunks.
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Chunk a document semantically.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        text = document.content

        # Detect and extract headers
        headers = self._extract_headers(text)

        # Choose chunking strategy based on document type
        if document.doc_type == DocumentType.MD:
            raw_chunks = self._chunk_markdown(text)
        elif document.doc_type == DocumentType.HTML:
            raw_chunks = self._chunk_html(text)
        else:
            raw_chunks = self._chunk_generic(text)

        # Create Chunk objects
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            # Find header for this chunk
            header = self._find_header_for_chunk(chunk_text, headers)

            chunk = Chunk(
                id=str(uuid4()),
                content=chunk_text,
                document_id=document.id,
                chunk_index=i,
                metadata={
                    "source": document.source,
                    "doc_type": document.doc_type.value,
                },
                header=header,
            )
            chunks.append(chunk)

        logger.info(
            f"Created {len(chunks)} chunks",
            metadata={
                "document_id": document.id,
                "avg_chunk_size": sum(len(c.content) for c in chunks) / max(len(chunks), 1),
            }
        )

        return chunks

    def _extract_headers(self, text: str) -> list[dict]:
        """Extract headers from text."""
        headers = []

        # Markdown headers
        md_pattern = r'^(#{1,6})\s+(.+)$'
        for match in re.finditer(md_pattern, text, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            headers.append({
                "level": level,
                "title": title,
                "position": match.start(),
            })

        return headers

    def _find_header_for_chunk(self, chunk_text: str, headers: list[dict]) -> Optional[str]:
        """Find the relevant header for a chunk."""
        # Simple heuristic: find header text in chunk
        for header in headers:
            if header["title"] in chunk_text:
                return header["title"]
        return None

    def _chunk_markdown(self, text: str) -> list[str]:
        """Chunk markdown document."""
        # Split by headers
        sections = re.split(r'\n(?=#{1,6}\s)', text)

        chunks = []
        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Check if section is too large
            if len(section.split()) * 1.3 > self.config.max_chunk_size:
                # Further split by paragraphs
                sub_chunks = chunk_by_paragraphs(
                    section,
                    max_chunk_size=self.config.chunk_size,
                    min_chunk_size=self.config.min_chunk_size,
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(section)

        return chunks

    def _chunk_html(self, text: str) -> list[str]:
        """Chunk HTML content (assumes already extracted text)."""
        return chunk_by_paragraphs(
            text,
            max_chunk_size=self.config.chunk_size,
            min_chunk_size=self.config.min_chunk_size,
        )

    def _chunk_generic(self, text: str) -> list[str]:
        """Chunk generic text."""
        if self.config.preserve_paragraphs:
            return chunk_by_paragraphs(
                text,
                max_chunk_size=self.config.chunk_size,
                min_chunk_size=self.config.min_chunk_size,
            )
        elif self.config.preserve_sentences:
            return chunk_by_sentences(
                text,
                max_chunk_size=self.config.chunk_size,
                min_chunk_size=self.config.min_chunk_size,
            )
        else:
            return chunk_by_tokens(
                text,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
            )


# =============================================================================
# Convenience Function
# =============================================================================

def chunk_document(
    document: Document,
    config: Optional[ChunkConfig] = None,
) -> list[Chunk]:
    """
    Chunk a document using semantic chunking.

    Args:
        document: Document to chunk
        config: Optional chunking configuration

    Returns:
        List of chunks
    """
    chunker = SemanticChunker(config)
    return chunker.chunk(document)
