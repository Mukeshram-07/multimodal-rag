"""
Unit tests for IngestionPipeline.

Tests use MagicMock / create_autospec for all dependencies so that no real
PDF files, embedding models, or ChromaDB instances are required.

Requirements: 1.1, 1.6, 1.7
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, create_autospec

import pytest

from rag.ingestion.chunker import Chunker
from rag.ingestion.pdf_parser import PDFParser
from rag.ingestion.pipeline import IngestionPipeline
from rag.models import Chunk, DocumentMetadata, IngestionResult, PageContent
from rag.storage.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(text: str, source: str = "doc.pdf", page: int = 1, idx: int = 0) -> Chunk:
    return Chunk(
        text=text,
        metadata=DocumentMetadata(source=source, page=page, chunk_index=idx),
    )


def _make_pipeline(
    pages: list[PageContent] | None = None,
    chunks: list[Chunk] | None = None,
    embeddings: list[list[float]] | None = None,
) -> tuple[IngestionPipeline, MagicMock, MagicMock, MagicMock, MagicMock]:
    """
    Build an IngestionPipeline with fully mocked dependencies.

    Returns (pipeline, mock_parser, mock_chunker, mock_embedding, mock_store).
    """
    if pages is None:
        pages = [PageContent(page_number=1, text="Hello world")]
    if chunks is None:
        chunks = [_make_chunk("Hello world")]
    if embeddings is None:
        embeddings = [[0.1] * 384]

    mock_parser = create_autospec(PDFParser, instance=True)
    mock_parser.parse.return_value = pages

    mock_chunker = create_autospec(Chunker, instance=True)
    mock_chunker.chunk.return_value = chunks

    mock_embedding = MagicMock()
    mock_embedding.encode.return_value = embeddings

    mock_store = create_autospec(VectorStore, instance=True)

    pipeline = IngestionPipeline(
        parser=mock_parser,
        chunker=mock_chunker,
        embedding_service=mock_embedding,
        vector_store=mock_store,
    )
    return pipeline, mock_parser, mock_chunker, mock_embedding, mock_store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIngestionPipelineEmbeddingCall:
    """EmbeddingBackend.encode is called with the chunk texts."""

    def test_encode_called_with_chunk_texts(self) -> None:
        chunks = [
            _make_chunk("First chunk", idx=0),
            _make_chunk("Second chunk", idx=1),
        ]
        embeddings = [[0.1] * 384, [0.2] * 384]
        pipeline, _, _, mock_embedding, _ = _make_pipeline(
            chunks=chunks, embeddings=embeddings
        )

        pipeline.ingest("some/path/doc.pdf", "test_collection")

        mock_embedding.encode.assert_called_once_with(["First chunk", "Second chunk"])

    def test_encode_called_once_per_ingest(self) -> None:
        pipeline, _, _, mock_embedding, _ = _make_pipeline()

        pipeline.ingest("doc.pdf", "col")

        assert mock_embedding.encode.call_count == 1


class TestIngestionPipelineVectorStoreCall:
    """VectorStore.add is called with the correct collection name."""

    def test_add_called_with_correct_collection_name(self) -> None:
        collection_name = "my_collection"
        pipeline, _, _, _, mock_store = _make_pipeline()

        pipeline.ingest("doc.pdf", collection_name)

        # First positional arg to add() must be the collection name.
        args, _ = mock_store.add.call_args
        assert args[0] == collection_name

    def test_add_called_with_chunks_and_embeddings(self) -> None:
        chunks = [_make_chunk("text", idx=0)]
        embeddings = [[0.5] * 384]
        pipeline, _, _, _, mock_store = _make_pipeline(
            chunks=chunks, embeddings=embeddings
        )

        pipeline.ingest("doc.pdf", "col")

        args, _ = mock_store.add.call_args
        assert args[1] == chunks
        assert args[2] == embeddings


class TestIngestionPipelineZeroChunks:
    """Zero-chunk case returns IngestionResult with chunk_count=0 and does not raise."""

    def test_zero_chunks_returns_ingestion_result(self) -> None:
        pipeline, _, _, mock_embedding, mock_store = _make_pipeline(
            chunks=[], embeddings=[]
        )

        result = pipeline.ingest("blank.pdf", "col")

        assert isinstance(result, IngestionResult)
        assert result.chunk_count == 0

    def test_zero_chunks_does_not_raise(self) -> None:
        pipeline, _, _, _, _ = _make_pipeline(chunks=[], embeddings=[])

        # Must not raise any exception.
        pipeline.ingest("blank.pdf", "col")

    def test_zero_chunks_encode_not_called(self) -> None:
        pipeline, _, _, mock_embedding, _ = _make_pipeline(chunks=[], embeddings=[])

        pipeline.ingest("blank.pdf", "col")

        mock_embedding.encode.assert_not_called()

    def test_zero_chunks_store_add_not_called(self) -> None:
        pipeline, _, _, _, mock_store = _make_pipeline(chunks=[], embeddings=[])

        pipeline.ingest("blank.pdf", "col")

        mock_store.add.assert_not_called()

    def test_zero_chunks_collection_name_preserved(self) -> None:
        pipeline, _, _, _, _ = _make_pipeline(chunks=[], embeddings=[])

        result = pipeline.ingest("blank.pdf", "my_col")

        assert result.collection_name == "my_col"


class TestIngestionPipelineSuccessStatus:
    """IngestionResult.status is a non-empty string on success."""

    def test_status_is_non_empty_string_on_success(self) -> None:
        pipeline, _, _, _, _ = _make_pipeline()

        result = pipeline.ingest("doc.pdf", "col")

        assert isinstance(result.status, str)
        assert len(result.status) > 0

    def test_result_chunk_count_matches_chunks(self) -> None:
        chunks = [_make_chunk("a", idx=0), _make_chunk("b", idx=1)]
        embeddings = [[0.1] * 384, [0.2] * 384]
        pipeline, _, _, _, _ = _make_pipeline(chunks=chunks, embeddings=embeddings)

        result = pipeline.ingest("doc.pdf", "col")

        assert result.chunk_count == 2

    def test_result_collection_name_matches_input(self) -> None:
        pipeline, _, _, _, _ = _make_pipeline()

        result = pipeline.ingest("doc.pdf", "target_collection")

        assert result.collection_name == "target_collection"

    def test_result_is_ingestion_result_instance(self) -> None:
        pipeline, _, _, _, _ = _make_pipeline()

        result = pipeline.ingest("doc.pdf", "col")

        assert isinstance(result, IngestionResult)
