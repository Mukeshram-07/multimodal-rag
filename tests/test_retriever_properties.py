"""
Property-based tests for Retriever.

# Feature: multimodal-rag, Property 4: Retrieval result ordering
# Feature: multimodal-rag, Property 6: Empty retrieval graceful handling

Uses a mock EmbeddingBackend (no real SentenceTransformer) and a real
VectorStore with isolated per-example ChromaDB directories to avoid
Windows file-locking issues.

Validates: Requirements 4.2, 4.3, 4.4, 4.5
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rag.models import Chunk, DocumentMetadata, RetrievalResult
from rag.retrieval.retriever import Retriever
from rag.storage.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 8  # small dimension for speed; real model not used

_example_counter_p4 = itertools.count()
_example_counter_p6 = itertools.count()


def _unit_embedding(dim: int = _DIM) -> list[float]:
    val = 1.0 / math.sqrt(dim)
    return [val] * dim


def _make_mock_embedding_service() -> MagicMock:
    svc = MagicMock()
    svc.encode.return_value = [_unit_embedding()]
    return svc


def _make_chunk(text: str, idx: int, source: str = "seed.pdf") -> Chunk:
    return Chunk(
        text=text,
        metadata=DocumentMetadata(source=source, page=1, chunk_index=idx),
    )


def _seed_collection(store: VectorStore, collection_name: str, n: int = 10) -> None:
    """Seed a collection with n fixed chunks and identical unit embeddings."""
    chunks = [_make_chunk(f"seed chunk number {i}", idx=i) for i in range(n)]
    embeddings = [_unit_embedding() for _ in chunks]
    store.add(collection_name, chunks, embeddings)


# ---------------------------------------------------------------------------
# Property 4: Retrieval result ordering
# ---------------------------------------------------------------------------


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    query=st.text(min_size=1, max_size=100).filter(lambda s: s.strip()),
    top_k=st.integers(min_value=1, max_value=10),
)
def test_retrieval_result_ordering(
    tmp_chroma_dir: Path,
    query: str,
    top_k: int,
) -> None:
    """
    # Feature: multimodal-rag, Property 4: Retrieval result ordering
    Validates: Requirements 4.2, 4.3, 4.4

    For any query and top_k:
    - len(result.chunks) <= top_k
    - results are ordered by descending score
    - each result has non-empty text and valid metadata
    """
    example_dir = tmp_chroma_dir / f"p4_{next(_example_counter_p4)}"
    example_dir.mkdir(parents=True, exist_ok=True)

    store = VectorStore(persist_directory=str(example_dir))
    _seed_collection(store, "prop4_col", n=10)

    retriever = Retriever(
        embedding_service=_make_mock_embedding_service(),
        vector_store=store,
    )
    result = retriever.retrieve(query, "prop4_col", top_k=top_k)

    # Assert: result count bounded by top_k
    assert len(result.chunks) <= top_k, (
        f"Expected <= {top_k} results, got {len(result.chunks)}"
    )

    # Assert: descending score order
    scores = [r.score for r in result.chunks]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"Score at index {i} ({scores[i]}) < score at index {i+1} ({scores[i+1]})"
        )

    # Assert: each result has valid content and metadata
    for sr in result.chunks:
        assert len(sr.chunk.text) > 0, "Chunk text must be non-empty"
        assert len(sr.chunk.metadata.source) > 0, "Metadata source must be non-empty"
        assert sr.chunk.metadata.page >= 1, "Page must be >= 1"
        assert sr.chunk.metadata.chunk_index >= 0, "chunk_index must be >= 0"


# ---------------------------------------------------------------------------
# Property 6: Empty retrieval graceful handling
# ---------------------------------------------------------------------------


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    query=st.text(min_size=1, max_size=100).filter(lambda s: s.strip()),
)
def test_empty_retrieval_graceful_handling(
    tmp_chroma_dir: Path,
    query: str,
) -> None:
    """
    # Feature: multimodal-rag, Property 6: Empty retrieval graceful handling
    Validates: Requirements 4.5

    For any query against an empty or non-existent collection:
    - returns RetrievalResult with empty chunks list
    - status is a non-empty string
    - no exception is raised
    """
    example_dir = tmp_chroma_dir / f"p6_{next(_example_counter_p6)}"
    example_dir.mkdir(parents=True, exist_ok=True)

    store = VectorStore(persist_directory=str(example_dir))
    retriever = Retriever(
        embedding_service=_make_mock_embedding_service(),
        vector_store=store,
    )

    # Test against a non-existent collection
    result_missing = retriever.retrieve(query, "nonexistent_collection", top_k=5)
    assert isinstance(result_missing, RetrievalResult)
    assert result_missing.chunks == [], (
        f"Expected empty chunks for missing collection, got {result_missing.chunks}"
    )
    assert isinstance(result_missing.status, str) and len(result_missing.status) > 0

    # Test against an empty collection (add with empty list — store stays empty)
    store.add("empty_col", [], [])
    result_empty = retriever.retrieve(query, "empty_col", top_k=5)
    assert isinstance(result_empty, RetrievalResult)
    assert result_empty.chunks == [], (
        f"Expected empty chunks for empty collection, got {result_empty.chunks}"
    )
    assert isinstance(result_empty.status, str) and len(result_empty.status) > 0
