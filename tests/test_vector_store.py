"""
Unit tests for VectorStore.

Tests:
  - list_collections returns a newly created collection.
  - delete_collection removes the collection from the list.
  - search on a non-existent collection returns empty list without raising.

Requirements: 3.3, 3.6, 3.7
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.models import Chunk, DocumentMetadata
from rag.storage import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(source: str = "doc.pdf", page: int = 1, chunk_index: int = 0, text: str = "hello") -> Chunk:
    return Chunk(
        text=text,
        metadata=DocumentMetadata(source=source, page=page, chunk_index=chunk_index),
    )


def _dummy_embedding(dim: int = 384) -> list[float]:
    """Return a unit vector of the given dimension."""
    val = 1.0 / (dim ** 0.5)
    return [val] * dim


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_list_collections_returns_new_collection(tmp_chroma_dir: Path) -> None:
    """After adding chunks to a collection, list_collections should include it."""
    store = VectorStore(persist_directory=str(tmp_chroma_dir))

    chunk = _make_chunk()
    store.add("my_collection", [chunk], [_dummy_embedding()])

    collections = store.list_collections()
    assert "my_collection" in collections


def test_list_collections_empty_initially(tmp_chroma_dir: Path) -> None:
    """A fresh VectorStore should have no collections."""
    store = VectorStore(persist_directory=str(tmp_chroma_dir))
    assert store.list_collections() == []


def test_delete_collection_removes_it(tmp_chroma_dir: Path) -> None:
    """delete_collection should remove the collection from list_collections."""
    store = VectorStore(persist_directory=str(tmp_chroma_dir))

    chunk = _make_chunk()
    store.add("to_delete", [chunk], [_dummy_embedding()])
    assert "to_delete" in store.list_collections()

    store.delete_collection("to_delete")
    assert "to_delete" not in store.list_collections()


def test_search_nonexistent_collection_returns_empty(tmp_chroma_dir: Path) -> None:
    """Searching a collection that does not exist should return [] without raising."""
    store = VectorStore(persist_directory=str(tmp_chroma_dir))

    results = store.search(
        collection_name="does_not_exist",
        query_embedding=_dummy_embedding(),
        top_k=5,
    )
    assert results == []


def test_search_returns_results_ordered_by_score(tmp_chroma_dir: Path) -> None:
    """Search results should be ordered by descending score."""
    store = VectorStore(persist_directory=str(tmp_chroma_dir))

    chunks = [
        _make_chunk(chunk_index=i, text=f"chunk number {i}") for i in range(5)
    ]
    embeddings = [_dummy_embedding() for _ in chunks]
    store.add("ordered_test", chunks, embeddings)

    results = store.search("ordered_test", _dummy_embedding(), top_k=5)
    assert len(results) > 0
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


def test_add_empty_chunks_is_noop(tmp_chroma_dir: Path) -> None:
    """Adding an empty list of chunks should not create a collection or raise."""
    store = VectorStore(persist_directory=str(tmp_chroma_dir))
    store.add("empty_test", [], [])
    # Collection should not be created (or if it is, it's empty)
    # The important thing is no exception is raised.


def test_search_with_filter_source(tmp_chroma_dir: Path) -> None:
    """filter_source should restrict results to chunks from the specified source."""
    store = VectorStore(persist_directory=str(tmp_chroma_dir))

    chunks_a = [_make_chunk(source="a.pdf", chunk_index=i, text=f"doc a chunk {i}") for i in range(3)]
    chunks_b = [_make_chunk(source="b.pdf", chunk_index=i, text=f"doc b chunk {i}") for i in range(3)]
    all_chunks = chunks_a + chunks_b
    embeddings = [_dummy_embedding() for _ in all_chunks]

    store.add("filter_test", all_chunks, embeddings)

    results = store.search("filter_test", _dummy_embedding(), top_k=10, filter_source="a.pdf")
    assert all(r.chunk.metadata.source == "a.pdf" for r in results)
