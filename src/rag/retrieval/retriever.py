"""
Retriever component for the Multimodal RAG System.

Executes semantic search by embedding a query and searching the vector store.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 11.5
"""

from __future__ import annotations

import time

from rag.embedding.protocols import EmbeddingBackend
from rag.logging_config import get_logger
from rag.models import RetrievalResult
from rag.storage.vector_store import VectorStore

logger = get_logger(__name__)

# Maximum characters of the query to include in log messages.
_QUERY_LOG_MAX_CHARS = 100


class Retriever:
    """
    Executes semantic search against a vector store.

    Embeds the query using the provided ``EmbeddingBackend``, then calls
    ``VectorStore.search`` to retrieve the most similar chunks.

    Args:
        embedding_service: Any object satisfying the ``EmbeddingBackend`` protocol.
        vector_store:      A ``VectorStore`` instance to search.
    """

    def __init__(
        self,
        embedding_service: EmbeddingBackend,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
        filter_source: str | None = None,
    ) -> RetrievalResult:
        """
        Retrieve the most semantically similar chunks for a query.

        Embeds the query, searches the named collection, and returns the
        results ordered by descending similarity score. When the collection
        is empty or does not exist, returns a ``RetrievalResult`` with an
        empty ``chunks`` list and an informative status message — does NOT
        raise an exception.

        Args:
            query:           The natural language query string.
            collection_name: Name of the collection to search.
            top_k:           Maximum number of results to return. Defaults to 5.
            filter_source:   Optional filename filter to restrict results to a
                             specific source document.

        Returns:
            A ``RetrievalResult`` with ``chunks`` (ordered by descending score)
            and a ``status`` string.
        """
        truncated_query = query[:_QUERY_LOG_MAX_CHARS]
        start_time = time.monotonic()

        logger.info(
            "Retrieval started: query='%s' collection=%s top_k=%d",
            truncated_query,
            collection_name,
            top_k,
        )

        # Embed the query — encode returns a list of vectors; take the first.
        query_embedding: list[float] = self._embedding_service.encode([query])[0]

        # Search the vector store.
        search_results = self._vector_store.search(
            collection_name,
            query_embedding,
            top_k,
            filter_source,
        )

        duration = time.monotonic() - start_time

        if not search_results:
            logger.info(
                "Retrieval complete (no results): query='%s' collection=%s "
                "result_count=0 duration_s=%.3f",
                truncated_query,
                collection_name,
                duration,
            )
            return RetrievalResult(
                chunks=[],
                status=f"No results found for query in collection '{collection_name}'",
            )

        logger.info(
            "Retrieval complete: query='%s' collection=%s result_count=%d duration_s=%.3f",
            truncated_query,
            collection_name,
            len(search_results),
            duration,
        )

        return RetrievalResult(chunks=search_results, status="ok")
