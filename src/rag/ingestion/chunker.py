"""
Text chunking component for the Multimodal RAG System.

Splits page text into overlapping character-level chunks and attaches
DocumentMetadata to each chunk.

Requirements: 1.3, 1.4, 1.6
"""

import logging

from rag.logging_config import get_logger
from rag.models import Chunk, DocumentMetadata, PageContent

logger = get_logger(__name__)


class Chunker:
    """
    Splits page text into overlapping character-level chunks.

    Each chunk is annotated with ``DocumentMetadata`` containing the source
    filename, 1-indexed page number, and 0-indexed chunk position within
    the document.

    Args:
        chunk_size:    Maximum number of characters per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text: str) -> list[str]:
        """
        Split a single text string into overlapping character-level chunks.

        Returns an empty list if the text is empty or whitespace-only.
        """
        stripped = text.strip()
        if not stripped:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: list[str] = []
        start = 0

        while start < len(stripped):
            end = start + self.chunk_size
            chunk_text = stripped[start:end]
            chunks.append(chunk_text)
            if end >= len(stripped):
                break
            start += step

        return chunks

    def chunk(self, pages: list[PageContent], source_filename: str) -> list[Chunk]:
        """
        Split all pages into overlapping chunks with attached metadata.

        Args:
            pages:           List of ``PageContent`` objects from the PDF parser.
            source_filename: The original filename of the source document.
                             Stored verbatim in each chunk's metadata.

        Returns:
            A flat list of ``Chunk`` objects ordered by page then chunk index.
            Pages with no text produce no chunks (empty list contribution).
        """
        logger.info(
            "Chunking document: source=%s pages=%d chunk_size=%d chunk_overlap=%d",
            source_filename,
            len(pages),
            self.chunk_size,
            self.chunk_overlap,
        )

        all_chunks: list[Chunk] = []
        global_chunk_index = 0

        for page in pages:
            page_text_chunks = self._split_text(page.text)

            for local_index, chunk_text in enumerate(page_text_chunks):
                metadata = DocumentMetadata(
                    source=source_filename,
                    page=page.page_number,
                    chunk_index=global_chunk_index,
                )
                all_chunks.append(Chunk(text=chunk_text, metadata=metadata))
                global_chunk_index += 1

        logger.info(
            "Chunking complete: source=%s total_chunks=%d",
            source_filename,
            len(all_chunks),
        )

        return all_chunks
