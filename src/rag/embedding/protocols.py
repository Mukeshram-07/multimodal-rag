"""
Protocol definitions for the embedding layer.

Defines the EmbeddingBackend Protocol so that any object implementing
``encode(texts: list[str]) -> list[list[float]]`` can be used wherever
an embedding service is required — enabling easy substitution in tests
and future extensions.

Requirements: 2.1, 2.4
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingBackend(Protocol):
    """
    Protocol for any embedding backend.

    Any class that implements ``encode`` with the correct signature
    satisfies this protocol without explicit inheritance.
    """

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a list of text strings into dense embedding vectors.

        Args:
            texts: A list of text strings to embed.

        Returns:
            A list of embedding vectors, one per input text, in the same order.
        """
        ...
