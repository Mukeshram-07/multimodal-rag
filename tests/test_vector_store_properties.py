"""
Property-based tests for VectorStore.

# Feature: multimodal-rag, Property 3: Upsert idempotence

**Property 3: Upsert idempotence**
**Validates: Requirements 3.5**

For any document ingested into the VectorStore twice with the same filename
and collection name, the total number of stored chunks SHALL be the same after
the second ingestion as after the first — i.e., re-ingestion does not duplicate
entries.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rag.models import Chunk, DocumentMetadata
from rag.storage import VectorStore


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Source filenames: simple alphanumeric names with .pdf extension.
source_filename_strategy = st.from_regex(r"[a-z][a-z0-9]{1,8}\.pdf", fullmatch=True)

# Non-empty chunk text.
chunk_text_strategy = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())

# Counter used to give each Hypothesis example its own subdirectory so that
# ChromaDB file handles from one example do not interfere with the next.
_example_counter = itertools.count()


def _make_chunks_for_source(source: str, texts: list[str]) -> list[Chunk]:
    """Build a list of Chunk objects for a given source filename."""
    return [
        Chunk(
            text=text,
            metadata=DocumentMetadata(source=source, page=1, chunk_index=i),
        )
        for i, text in enumerate(texts)
    ]


def _dummy_embeddings(n: int, dim: int = 8) -> list[list[float]]:
    """Return n identical dummy embeddings of dimension dim."""
    val = 1.0 / (dim ** 0.5)
    return [[val] * dim for _ in range(n)]


# ---------------------------------------------------------------------------
# Property 3: Upsert idempotence
# ---------------------------------------------------------------------------


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    source=source_filename_strategy,
    texts=st.lists(chunk_text_strategy, min_size=1, max_size=10),
)
def test_upsert_idempotence(
    tmp_chroma_dir: Path,
    source: str,
    texts: list[str],
) -> None:
    """
    **Property 3: Upsert idempotence**
    **Validates: Requirements 3.5**

    Ingesting the same document twice into the same collection should not
    increase the number of stored items beyond the count after the first
    ingestion.

    Each Hypothesis example uses a unique subdirectory within ``tmp_chroma_dir``
    so that ChromaDB file handles from one example do not interfere with the
    next. The parent ``tmp_chroma_dir`` is cleaned up by pytest after the test
    session, avoiding Windows file-locking issues.
    """
    # Give each generated example its own isolated ChromaDB directory.
    example_dir = tmp_chroma_dir / f"example_{next(_example_counter)}"
    example_dir.mkdir(parents=True, exist_ok=True)

    store = VectorStore(persist_directory=str(example_dir))
    collection_name = "idempotence_test"

    chunks = _make_chunks_for_source(source, texts)
    embeddings = _dummy_embeddings(len(chunks))

    # First ingestion
    store.add(collection_name, chunks, embeddings)
    collection_after_first = store._client.get_collection(
        name=collection_name,
        embedding_function=None,
    )
    count_after_first = collection_after_first.count()

    # Second ingestion of the same data
    store.add(collection_name, chunks, embeddings)
    collection_after_second = store._client.get_collection(
        name=collection_name,
        embedding_function=None,
    )
    count_after_second = collection_after_second.count()

    assert count_after_second == count_after_first, (
        f"Expected {count_after_first} items after second ingestion, "
        f"got {count_after_second}. Upsert is not idempotent."
    )
