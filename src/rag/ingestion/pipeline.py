"""
Ingestion pipeline for the Multimodal RAG System.

Orchestrates the full ingestion flow: PDF parsing → chunking → embedding → storage.

Requirements: 1.1, 1.6, 1.7, 11.4
"""

from __future__ import annotations

import time
from pathlib import Path

from rag.embedding.protocols import EmbeddingBackend
from rag.ingestion.chunker import Chunker
from rag.ingestion.pdf_parser import PDFParser
from rag.logging_config import get_logger
from rag.models import IngestionResult
from rag.storage.vector_store import VectorStore

logger = get_logger(__name__)


class IngestionPipeline:
    """
    Orchestrates the full document ingestion flow.

    Accepts a PDF file path and a collection name, then:
    1. Parses the PDF into per-page text using ``PDFParser``.
    2. Splits the pages into overlapping ``Chunk`` objects using ``Chunker``.
    3. Encodes the chunk texts into embedding vectors using ``EmbeddingBackend``.
    4. Persists the chunks and embeddings into ``VectorStore``.

    Args:
        parser:            A ``PDFParser`` instance for extracting page text.
        chunker:           A ``Chunker`` instance for splitting text into chunks.
        embedding_service: Any object satisfying the ``EmbeddingBackend`` protocol.
        vector_store:      A ``VectorStore`` instance for persisting embeddings.
    """

    def __init__(
        self,
        parser: PDFParser,
        chunker: Chunker,
        embedding_service: EmbeddingBackend,
        vector_store: VectorStore,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def ingest(self, file_path: str, collection_name: str) -> IngestionResult:
        """
        Ingest a PDF document into the vector store.

        Parses the PDF, chunks the text, embeds the chunks, and stores them
        in the named collection. Logs the document name, chunk count, and
        processing duration at INFO level.

        When the document produces zero chunks (e.g., a blank PDF), returns
        an ``IngestionResult`` with ``chunk_count=0`` and an informative
        status message — does NOT raise an exception.

        Args:
            file_path:       Absolute or relative path to the PDF file.
            collection_name: Name of the collection to store chunks in.

        Returns:
            An ``IngestionResult`` with ``status``, ``chunk_count``, and
            ``collection_name``.

        Raises:
            IngestionError: If the PDF cannot be parsed.
            EmbeddingError: If the embedding service fails.
        """
        filename = Path(file_path).name
        start_time = time.monotonic()

        logger.info(
            "Ingestion started: file=%s collection=%s",
            filename,
            collection_name,
        )

        # Step 1: Parse PDF into pages
        pages = self._parser.parse(file_path)

        # Step 2: Chunk pages into Chunk objects
        chunks = self._chunker.chunk(pages, source_filename=filename)

        chunk_count = len(chunks)

        if chunk_count == 0:
            duration = time.monotonic() - start_time
            logger.info(
                "Ingestion complete (no chunks): file=%s collection=%s "
                "chunk_count=0 duration_s=%.3f",
                filename,
                collection_name,
                duration,
            )
            return IngestionResult(
                status="No chunks produced",
                chunk_count=0,
                collection_name=collection_name,
            )

        # Step 3: Embed chunk texts
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self._embedding_service.encode(chunk_texts)

        # Step 4: Store chunks and embeddings
        self._vector_store.add(collection_name, chunks, embeddings)

        duration = time.monotonic() - start_time

        logger.info(
            "Ingestion complete: file=%s collection=%s chunk_count=%d duration_s=%.3f",
            filename,
            collection_name,
            chunk_count,
            duration,
        )

        return IngestionResult(
            status=f"Ingested {chunk_count} chunks from '{filename}'",
            chunk_count=chunk_count,
            collection_name=collection_name,
        )
