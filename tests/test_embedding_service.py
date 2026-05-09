"""
Unit tests for EmbeddingService.

Tests:
  - Embedding dimensionality is 384 for all-MiniLM-L6-v2.
  - Empty input list returns empty output list.

Requirements: 2.1, 2.2
"""

import pytest

from rag.embedding import EmbeddingService


@pytest.fixture(scope="module")
def embedding_service() -> EmbeddingService:
    """Shared EmbeddingService instance (model loaded once per module)."""
    return EmbeddingService(model_name="all-MiniLM-L6-v2")


def test_embedding_dimensionality(embedding_service: EmbeddingService) -> None:
    """Embeddings produced by all-MiniLM-L6-v2 should be 384-dimensional."""
    texts = ["Hello, world!", "The quick brown fox jumps over the lazy dog."]
    embeddings = embedding_service.encode(texts)

    assert len(embeddings) == len(texts)
    for vec in embeddings:
        assert len(vec) == 384, f"Expected 384 dimensions, got {len(vec)}"


def test_empty_input_returns_empty_output(embedding_service: EmbeddingService) -> None:
    """Encoding an empty list should return an empty list without raising."""
    result = embedding_service.encode([])
    assert result == []


def test_single_text_returns_one_embedding(embedding_service: EmbeddingService) -> None:
    """A single-element input should return exactly one embedding vector."""
    result = embedding_service.encode(["single sentence"])
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert len(result[0]) == 384


def test_embeddings_are_floats(embedding_service: EmbeddingService) -> None:
    """Each element of an embedding vector should be a float."""
    result = embedding_service.encode(["test sentence"])
    assert all(isinstance(v, float) for v in result[0])
