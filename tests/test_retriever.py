"""
Unit tests for Retriever.

Uses a real VectorStore (with tmp_chroma_dir) and a mock EmbeddingBackend
that returns fixed unit vectors — no real SentenceTransformer is loaded.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rag.models import Chunk, DocumentMetadata, RetrievalResult
from rag.retrieval.retriever import Retriever
from rag.storage.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 384


def _unit_embedding(dim: int = _DIM) -> list[float]:
    """Return a normalised unit vector of the given dimension."""
    val = 1.0 / math.sqrt(dim)
    return [val] * dim


def _make_mock_embedding_service(embedding: list[float] | None = None) -> MagicMock:
    """Return a mock EmbeddingBackend that always returns the given embedding."""
    if embedding is None:
        embedding = _unit_embedding()
    svc = MagicMock()
    svc.encode.return_value = [embedding]
    return svc


def _make_chunk(
    text: str,
    source: str = "doc.pdf",
    page: int = 1,
    chunk_index: int = 0,
) -> Chunk:
    return Chunk(
        text=text,
        metadata=DocumentMetadata(source=source, page=page, chunk_index=chunk_index),
    )


def _seed_collection(
    store: VectorStore,
    collection_name: str,
    chunks: list[Chunk],
    embedding: list[float] | None = None,
) -> None:
    """Upsert chunks into the store with identical dummy embeddings."""
    if embedding is None:
        embedding = _unit_embedding()
    embeddings = [embedding for _ in chunks]
    store.add(collection_name, chunks, embeddings)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_retrieve_returns_results_for_seeded_collection(tmp_chroma_dir: Path) -> None:
    """Retrieval against a seeded collection returns at least one result."""
    store = VectorStore(persist_directory=str(tmp_chroma_dir))
    chunks = [_make_chunk(f"chunk {i}", chunk_index=i) for i in range(3)]
    _seed_collection(store, "col", chunks)

    retriever = Retriever(
        embedding_service=_make_mock_embedding_service(),
        vector_store=store,
    )
    result = retriever.retrieve("test query", "col", top_k=5)

    assert isinstance(result, RetrievalResult)
    assert len(result.chunks) > 0


def test_top_k_limits_results(tmp_chroma_dir: Path) -> None:
    """top_k caps the number of returned results."""
    store = VectorStore(persist_directory=str(tmp_chroma_dir))
    chunks = [_make_chunk(f"chunk {i}", chunk_index=i) for i in range(10)]
    _seed_collection(store, "col", chunks)

    retriever = Retriever(
        embedding_service=_make_mock_embedding_service(),
        vector_store=store,
    )
    result = retriever.retrieve("query", "col", top_k=3)

    assert len(result.chunks) <= 3


def test_results_ordered_by_descending_score(tmp_chroma_dir: Path) -> None:
    """Results must be ordered by descending similarity score."""
    store = VectorStore(persist_directory=str(tmp_chroma_dir))
    chunks = [_make_chunk(f"chunk {i}", chunk_index=i) for i in range(5)]
    _seed_collection(store, "col", chunks)

    retriever = Retriever(
        embedding_service=_make_mock_embedding_service(),
        vector_store=store,
    )
    result = retriever.retrieve("query", "col", top_k=5)

    scores = [r.score for r in result.chunks]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"Score at index {i} ({scores[i]}) < score at index {i+1} ({scores[i+1]})"
        )


def test_filter_source_restricts_results(tmp_chroma_dir: Path) -> None:
    """filter_source restricts results to chunks from the specified source."""
    store = VectorStore(persist_directory=str(tmp_chroma_dir))
    chunks_a = [_make_chunk(f"doc a chunk {i}", source="a.pdf", chunk_index=i) for i in range(3)]
    chunks_b = [_make_chunk(f"doc b chunk {i}", source="b.pdf", chunk_index=i) for i in range(3)]
    _seed_collection(store, "col", chunks_a + chunks_b)

    retriever = Retriever(
        embedding_service=_make_mock_embedding_service(),
        vector_store=store,
    )
    result = retriever.retrieve("query", "col", top_k=10, filter_source="a.pdf")

    assert len(result.chunks) > 0
    for sr in result.chunks:
        assert sr.chunk.metadata.source == "a.pdf", (
            f"Expected source='a.pdf', got '{sr.chunk.metadata.source}'"
        )


def test_empty_collection_returns_empty_result(tmp_chroma_dir: Path) -> None:
    """Retrieving from an empty collection returns empty chunks and non-empty status."""
    store = VectorStore(persist_directory=str(tmp_chroma_dir))
    # Create the collection but add nothing to it — use add with empty list
    # (VectorStore.add is a no-op for empty input, so collection stays absent;
    # search handles missing collection gracefully)

    retriever = Retriever(
        embedding_service=_make_mock_embedding_service(),
        vector_store=store,
    )
    result = retriever.retrieve("query", "empty_col", top_k=5)

    assert result.chunks == []
    assert isinstance(result.status, str)
    assert len(result.status) > 0


def test_nonexistent_collection_returns_empty_result(tmp_chroma_dir: Path) -> None:
    """Retrieving from a non-existent collection returns empty chunks without raising."""
    store = VectorStore(persist_directory=str(tmp_chroma_dir))

    retriever = Retriever(
        embedding_service=_make_mock_embedding_service(),
        vector_store=store,
    )
    result = retriever.retrieve("query", "does_not_exist", top_k=5)

    assert result.chunks == []
    assert isinstance(result.status, str)
    assert len(result.status) > 0


def test_metadata_reconstruction(tmp_chroma_dir: Path) -> None:
    """Metadata (source, page, chunk_index) is correctly reconstructed from storage."""
    store = VectorStore(persist_directory=str(tmp_chroma_dir))
    chunk = _make_chunk(
        text="unique metadata test chunk",
        source="metadata_test.pdf",
        page=3,
        chunk_index=7,
    )
    _seed_collection(store, "col", [chunk])

    retriever = Retriever(
        embedding_service=_make_mock_embedding_service(),
        vector_store=store,
    )
    result = retriever.retrieve("query", "col", top_k=1)

    assert len(result.chunks) == 1
    meta = result.chunks[0].chunk.metadata
    assert meta.source == "metadata_test.pdf"
    assert meta.page == 3
    assert meta.chunk_index == 7


def test_retrieval_determinism(tmp_chroma_dir: Path) -> None:
    """Calling retrieve twice with the same query returns the same results."""
    store = VectorStore(persist_directory=str(tmp_chroma_dir))
    chunks = [_make_chunk(f"chunk {i}", chunk_index=i) for i in range(5)]
    _seed_collection(store, "col", chunks)

    retriever = Retriever(
        embedding_service=_make_mock_embedding_service(),
        vector_store=store,
    )

    result1 = retriever.retrieve("determinism query", "col", top_k=5)
    result2 = retriever.retrieve("determinism query", "col", top_k=5)

    assert len(result1.chunks) == len(result2.chunks)
    for r1, r2 in zip(result1.chunks, result2.chunks):
        assert r1.chunk.text == r2.chunk.text
        assert abs(r1.score - r2.score) < 1e-6


def test_retrieve_returns_retrieval_result_type(tmp_chroma_dir: Path) -> None:
    """Return type is always RetrievalResult regardless of collection state."""
    store = VectorStore(persist_directory=str(tmp_chroma_dir))

    retriever = Retriever(
        embedding_service=_make_mock_embedding_service(),
        vector_store=store,
    )
    result = retriever.retrieve("query", "nonexistent", top_k=5)

    assert isinstance(result, RetrievalResult)


def test_status_ok_when_results_found(tmp_chroma_dir: Path) -> None:
    """status is 'ok' when results are returned."""
    store = VectorStore(persist_directory=str(tmp_chroma_dir))
    chunks = [_make_chunk("some text", chunk_index=0)]
    _seed_collection(store, "col", chunks)

    retriever = Retriever(
        embedding_service=_make_mock_embedding_service(),
        vector_store=store,
    )
    result = retriever.retrieve("query", "col", top_k=5)

    assert result.status == "ok"


# ---------------------------------------------------------------------------
# Summary-mode detection tests
# ---------------------------------------------------------------------------


from rag.retrieval.retriever import _is_summary_query  # noqa: E402


class TestSummaryQueryDetection:
    """_is_summary_query correctly identifies summary-style queries."""

    def test_summarize_detected(self) -> None:
        assert _is_summary_query("summarize this document")

    def test_summarise_british_spelling(self) -> None:
        assert _is_summary_query("please summarise the paper")

    def test_summary_noun(self) -> None:
        assert _is_summary_query("give me a summary")

    def test_overview(self) -> None:
        assert _is_summary_query("give an overview of this article")

    def test_what_is_this_about(self) -> None:
        assert _is_summary_query("what is this document about")

    def test_main_topic(self) -> None:
        assert _is_summary_query("what is the main topic")

    def test_key_findings(self) -> None:
        assert _is_summary_query("what are the key findings")

    def test_describe_this_paper(self) -> None:
        assert _is_summary_query("describe this paper")

    def test_tell_me_about_this(self) -> None:
        assert _is_summary_query("tell me about this")

    def test_factual_query_not_detected(self) -> None:
        assert not _is_summary_query("what is the capital of France?")

    def test_specific_question_not_detected(self) -> None:
        assert not _is_summary_query("what year was the model trained?")

    def test_how_does_it_work_not_detected(self) -> None:
        assert not _is_summary_query("how does the attention mechanism work?")

    def test_empty_string_not_detected(self) -> None:
        assert not _is_summary_query("")

    def test_case_insensitive(self) -> None:
        assert _is_summary_query("SUMMARIZE THIS DOCUMENT")
        assert _is_summary_query("Give Me An Overview")


# ---------------------------------------------------------------------------
# Summary-mode retrieval tests
# ---------------------------------------------------------------------------


class TestSummaryModeRetrieval:
    """Retriever uses sequential chunk fetching for summary queries."""

    def test_summary_query_returns_results(self, tmp_chroma_dir: Path) -> None:
        store = VectorStore(persist_directory=str(tmp_chroma_dir))
        chunks = [_make_chunk(f"content of chunk {i}", chunk_index=i) for i in range(10)]
        _seed_collection(store, "col", chunks)

        retriever = Retriever(
            embedding_service=_make_mock_embedding_service(),
            vector_store=store,
        )
        result = retriever.retrieve("summarize this document", "col", top_k=5)

        assert len(result.chunks) > 0

    def test_summary_mode_returns_at_least_summary_chunk_count(self, tmp_chroma_dir: Path) -> None:
        """Summary mode fetches more chunks than the default top_k."""
        from rag.retrieval.retriever import _SUMMARY_CHUNK_COUNT
        store = VectorStore(persist_directory=str(tmp_chroma_dir))
        # Seed more chunks than _SUMMARY_CHUNK_COUNT
        n = _SUMMARY_CHUNK_COUNT + 5
        chunks = [_make_chunk(f"chunk {i}", chunk_index=i) for i in range(n)]
        _seed_collection(store, "col", chunks)

        retriever = Retriever(
            embedding_service=_make_mock_embedding_service(),
            vector_store=store,
        )
        result = retriever.retrieve("give me an overview", "col", top_k=5)

        # Should return at least _SUMMARY_CHUNK_COUNT (not just top_k=5)
        assert len(result.chunks) >= _SUMMARY_CHUNK_COUNT

    def test_summary_mode_chunks_ordered_by_chunk_index(self, tmp_chroma_dir: Path) -> None:
        """Summary mode returns chunks in document order (ascending chunk_index)."""
        store = VectorStore(persist_directory=str(tmp_chroma_dir))
        chunks = [_make_chunk(f"chunk {i}", chunk_index=i) for i in range(8)]
        _seed_collection(store, "col", chunks)

        retriever = Retriever(
            embedding_service=_make_mock_embedding_service(),
            vector_store=store,
        )
        result = retriever.retrieve("summarize this document", "col", top_k=8)

        indices = [sr.chunk.metadata.chunk_index for sr in result.chunks]
        assert indices == sorted(indices), f"Expected sorted indices, got {indices}"

    def test_summary_mode_status_contains_summary(self, tmp_chroma_dir: Path) -> None:
        """Summary mode status string indicates the retrieval mode used."""
        store = VectorStore(persist_directory=str(tmp_chroma_dir))
        chunks = [_make_chunk("text", chunk_index=i) for i in range(3)]
        _seed_collection(store, "col", chunks)

        retriever = Retriever(
            embedding_service=_make_mock_embedding_service(),
            vector_store=store,
        )
        result = retriever.retrieve("give me a summary", "col", top_k=5)

        assert "summary" in result.status.lower()

    def test_semantic_query_does_not_use_summary_mode(self, tmp_chroma_dir: Path) -> None:
        """A factual query uses semantic mode and returns status 'ok'."""
        store = VectorStore(persist_directory=str(tmp_chroma_dir))
        chunks = [_make_chunk(f"chunk {i}", chunk_index=i) for i in range(5)]
        _seed_collection(store, "col", chunks)

        retriever = Retriever(
            embedding_service=_make_mock_embedding_service(),
            vector_store=store,
        )
        result = retriever.retrieve("what is the capital of France?", "col", top_k=5)

        assert result.status == "ok"

    def test_summary_mode_respects_filter_source(self, tmp_chroma_dir: Path) -> None:
        """Summary mode honours filter_source."""
        store = VectorStore(persist_directory=str(tmp_chroma_dir))
        chunks_a = [_make_chunk(f"a {i}", source="a.pdf", chunk_index=i) for i in range(5)]
        chunks_b = [_make_chunk(f"b {i}", source="b.pdf", chunk_index=i) for i in range(5)]
        _seed_collection(store, "col", chunks_a + chunks_b)

        retriever = Retriever(
            embedding_service=_make_mock_embedding_service(),
            vector_store=store,
        )
        result = retriever.retrieve("summarize this document", "col", top_k=5, filter_source="a.pdf")

        assert all(sr.chunk.metadata.source == "a.pdf" for sr in result.chunks)

    def test_summary_mode_empty_collection_returns_empty(self, tmp_chroma_dir: Path) -> None:
        """Summary mode on a missing collection returns empty result gracefully."""
        store = VectorStore(persist_directory=str(tmp_chroma_dir))

        retriever = Retriever(
            embedding_service=_make_mock_embedding_service(),
            vector_store=store,
        )
        result = retriever.retrieve("summarize this document", "nonexistent", top_k=5)

        assert result.chunks == []
        assert len(result.status) > 0
