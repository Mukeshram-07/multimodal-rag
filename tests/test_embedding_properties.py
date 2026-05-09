"""
Property-based tests for EmbeddingService.

# Feature: multimodal-rag, Property 2: Embedding order preservation

**Property 2: Embedding order preservation**
**Validates: Requirements 2.1, 2.3, 2.5**

For any list of text strings passed to EmbeddingService.encode(), the returned
list of embeddings SHALL have the same length as the input list, and the
embedding at index i SHALL correspond to the text at index i (verified by
comparing batch encoding against individual encoding within floating-point
tolerance).
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rag.embedding import EmbeddingService

# ---------------------------------------------------------------------------
# Shared fixture — load the model once for the entire module to avoid
# repeated expensive initialisation during property testing.
# ---------------------------------------------------------------------------

_SERVICE: EmbeddingService | None = None


def _get_service() -> EmbeddingService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = EmbeddingService(model_name="all-MiniLM-L6-v2")
    return _SERVICE


# ---------------------------------------------------------------------------
# Hypothesis strategy: lists of 1–20 non-empty strings
# ---------------------------------------------------------------------------

non_empty_text = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())
text_list = st.lists(non_empty_text, min_size=1, max_size=20)


# ---------------------------------------------------------------------------
# Property 2: Embedding order preservation
# ---------------------------------------------------------------------------


@settings(max_examples=30, deadline=None)
@given(texts=text_list)
def test_embedding_order_preservation(texts: list[str]) -> None:
    """
    **Property 2: Embedding order preservation**
    **Validates: Requirements 2.1, 2.3, 2.5**

    Asserts:
      1. len(encode(texts)) == len(texts)
      2. Batch encoding produces the same vectors as individual encoding
         (within floating-point tolerance of 1e-5 per element).
    """
    service = _get_service()

    # --- Assertion 1: output length matches input length ---
    batch_embeddings = service.encode(texts)
    assert len(batch_embeddings) == len(texts), (
        f"Expected {len(texts)} embeddings, got {len(batch_embeddings)}"
    )

    # --- Assertion 2: batch == individual (within tolerance) ---
    for i, text in enumerate(texts):
        individual_embedding = service.encode([text])[0]
        batch_embedding = batch_embeddings[i]

        assert len(individual_embedding) == len(batch_embedding), (
            f"Dimension mismatch at index {i}: "
            f"individual={len(individual_embedding)}, batch={len(batch_embedding)}"
        )

        for j, (a, b) in enumerate(zip(individual_embedding, batch_embedding)):
            assert abs(a - b) < 1e-5, (
                f"Embedding mismatch at text index {i}, dimension {j}: "
                f"individual={a}, batch={b}, diff={abs(a - b)}"
            )
