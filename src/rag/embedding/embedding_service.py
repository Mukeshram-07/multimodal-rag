"""
Embedding service for the Multimodal RAG System.

Wraps ``sentence_transformers.SentenceTransformer`` to produce dense
embedding vectors from text strings. Supports batched encoding and
raises ``EmbeddingError`` on failure.

Requirements: 2.1, 2.2, 2.4, 2.5
"""

from __future__ import annotations

import time

from sentence_transformers import SentenceTransformer

from rag.exceptions import EmbeddingError
from rag.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Converts text strings to dense embedding vectors using a SentenceTransformer model.

    Args:
        model_name: The name of the sentence-transformers model to load.
                    Defaults to ``"all-MiniLM-L6-v2"`` (384-dimensional embeddings).
        batch_size: Number of texts to encode per batch. Defaults to 32.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        logger.info("Loading embedding model: model=%s", model_name)
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            raise EmbeddingError(
                message=f"Failed to load embedding model '{model_name}': {exc}",
                detail=str(exc),
            ) from exc
        logger.info("Embedding model loaded: model=%s", model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a list of text strings into dense embedding vectors.

        Encoding is performed in batches of ``self.batch_size`` to avoid
        memory pressure on large inputs.

        Args:
            texts: A list of text strings to embed. May be empty.

        Returns:
            A list of embedding vectors (``list[float]``), one per input text,
            in the same order as the input.

        Raises:
            EmbeddingError: If the model fails to encode any text, with the
                            0-based index of the failing chunk and a detail message.
        """
        if not texts:
            return []

        logger.info(
            "Starting embedding: model=%s batch_size=%d num_texts=%d",
            self.model_name,
            self.batch_size,
            len(texts),
        )

        start_time = time.monotonic()

        try:
            # SentenceTransformer.encode accepts a list of strings and returns
            # a numpy array of shape (len(texts), embedding_dim).
            embeddings_array = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise EmbeddingError(
                message=f"Embedding failed: {exc}",
                chunk_index=0,
                detail=str(exc),
            ) from exc

        duration = time.monotonic() - start_time

        logger.info(
            "Embedding complete: model=%s num_texts=%d duration_s=%.3f",
            self.model_name,
            len(texts),
            duration,
        )

        # Convert numpy array rows to plain Python lists of floats.
        return [row.tolist() for row in embeddings_array]
