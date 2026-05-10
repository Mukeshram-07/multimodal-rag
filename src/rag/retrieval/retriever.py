"""
Retriever component for the Multimodal RAG System.

Supports two retrieval modes:

* **semantic** (default) — embeds the query and returns the most similar
  chunks by cosine similarity.  Best for factual, specific questions.

* **summary** — fetches the first N chunks in document order (by
  chunk_index).  Triggered automatically for vague, broad queries like
  "summarise this document" where semantic similarity to a short query
  string would return scattered, unrepresentative chunks.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 11.5
"""

from __future__ import annotations

import re
import time

from rag.embedding.protocols import EmbeddingBackend
from rag.logging_config import get_logger
from rag.models import RetrievalResult
from rag.storage.vector_store import VectorStore

logger = get_logger(__name__)

# Maximum characters of the query to include in log messages.
_QUERY_LOG_MAX_CHARS = 100

# How many sequential chunks to fetch in summary mode.
# Larger than the default top_k so the LLM gets broad document coverage.
_SUMMARY_CHUNK_COUNT = 12

# ---------------------------------------------------------------------------
# Summary-query detection
# ---------------------------------------------------------------------------

# Patterns that indicate the user wants a broad overview rather than a
# specific fact.  Matched case-insensitively against the stripped query.
_SUMMARY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'\bsummar(?:ize|ise|y|ies)\b'),
    re.compile(r'\boverview\b'),
    re.compile(r'\bsynopsis\b'),
    re.compile(r'\babstract\b'),
    re.compile(r'\bwhat\s+is\s+this\s+(document|paper|article|about)\b'),
    re.compile(r'\bwhat\s+does\s+this\s+(document|paper|article)\s+(say|cover|discuss|contain)\b'),
    re.compile(r'\bdescribe\s+this\s+(document|paper|article)\b'),
    re.compile(r'\bexplain\s+this\s+(document|paper|article)\b'),
    re.compile(r'\bgive\s+(me\s+)?(an?\s+)?(overview|summary|synopsis|brief)\b'),
    re.compile(r'\bmain\s+(topic|idea|point|theme|subject|content)\b'),
    re.compile(r'\bkey\s+(point|finding|takeaway|insight|topic)s?\b'),
    re.compile(r'\bwhat\s+is\s+it\s+about\b'),
    re.compile(r'\btell\s+me\s+about\s+this\b'),
]


def _is_summary_query(query: str) -> bool:
    """Return True if the query is asking for a broad document overview."""
    q = query.strip().lower()
    return any(p.search(q) for p in _SUMMARY_PATTERNS)


class Retriever:
    """
    Executes retrieval against a vector store using the appropriate strategy.

    For factual queries, uses semantic similarity search.
    For summary/overview queries, fetches sequential chunks from the start
    of the document to give the LLM broad, ordered context.

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
        filter_source: str | None = None,
    ) -> RetrievalResult:
        """
        Retrieve chunks using the strategy best suited to the query.

        Automatically selects between semantic and summary retrieval modes
        based on the query text.  The caller can always override by passing
        ``top_k`` — summary mode uses ``max(top_k, _SUMMARY_CHUNK_COUNT)``
        to ensure broad coverage.

        Args:
            query:           The natural language query string.
            collection_name: Name of the collection to search.
            top_k:           Maximum number of results for semantic mode.
                             Summary mode uses at least ``_SUMMARY_CHUNK_COUNT``.
            filter_source:   Optional filename filter.

        Returns:
            A ``RetrievalResult`` with ``chunks`` and a ``status`` string.
        """
        if _is_summary_query(query):
            return self._retrieve_summary(query, collection_name, top_k, filter_source)
        return self._retrieve_semantic(query, collection_name, top_k, filter_source)

    # ------------------------------------------------------------------
    # Private: semantic mode
    # ------------------------------------------------------------------

    def _retrieve_semantic(
        self,
        query: str,
        collection_name: str,
        top_k: int,
        filter_source: str | None,
    ) -> RetrievalResult:
        truncated = query[:_QUERY_LOG_MAX_CHARS]
        start = time.monotonic()

        logger.info(
            "Retrieval [semantic]: query='%s' collection=%s top_k=%d",
            truncated,
            collection_name,
            top_k,
        )

        query_embedding = self._embedding_service.encode([query])[0]
        search_results = self._vector_store.search(
            collection_name, query_embedding, top_k, filter_source
        )

        duration = time.monotonic() - start

        if not search_results:
            logger.info(
                "Retrieval [semantic] complete (no results): query='%s' "
                "collection=%s duration_s=%.3f",
                truncated, collection_name, duration,
            )
            return RetrievalResult(
                chunks=[],
                status=f"No results found for query in collection '{collection_name}'",
            )

        self._log_chunks(search_results, mode="semantic")
        logger.info(
            "Retrieval [semantic] complete: query='%s' collection=%s "
            "result_count=%d duration_s=%.3f",
            truncated, collection_name, len(search_results), duration,
        )
        return RetrievalResult(chunks=search_results, status="ok")

    # ------------------------------------------------------------------
    # Private: summary mode
    # ------------------------------------------------------------------

    def _retrieve_summary(
        self,
        query: str,
        collection_name: str,
        top_k: int,
        filter_source: str | None,
    ) -> RetrievalResult:
        """
        Fetch sequential chunks from the start of the document.

        Summary queries need breadth, not precision.  Returning the first
        N chunks in document order gives the LLM the introduction, abstract,
        and early body sections — the parts most likely to contain an
        overview of the document's content.
        """
        truncated = query[:_QUERY_LOG_MAX_CHARS]
        # Use at least _SUMMARY_CHUNK_COUNT chunks; honour larger top_k if set.
        limit = max(top_k, _SUMMARY_CHUNK_COUNT)
        start = time.monotonic()

        logger.info(
            "Retrieval [summary]: query='%s' collection=%s limit=%d "
            "filter_source=%s — fetching sequential chunks",
            truncated, collection_name, limit, filter_source,
        )

        search_results = self._vector_store.get_sequential(
            collection_name, limit=limit, filter_source=filter_source
        )

        duration = time.monotonic() - start

        if not search_results:
            logger.info(
                "Retrieval [summary] complete (no results): query='%s' "
                "collection=%s duration_s=%.3f",
                truncated, collection_name, duration,
            )
            return RetrievalResult(
                chunks=[],
                status=f"No results found for query in collection '{collection_name}'",
            )

        self._log_chunks(search_results, mode="summary")
        logger.info(
            "Retrieval [summary] complete: query='%s' collection=%s "
            "result_count=%d duration_s=%.3f",
            truncated, collection_name, len(search_results), duration,
        )
        return RetrievalResult(
            chunks=search_results,
            status=f"summary-mode: {len(search_results)} sequential chunks",
        )

    # ------------------------------------------------------------------
    # Private: helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_chunks(search_results, *, mode: str) -> None:
        """Log the first few retrieved chunks for debugging."""
        for i, sr in enumerate(search_results[:3]):
            logger.debug(
                "Retrieved chunk[%d] [%s]: source=%s page=%d chunk_index=%d "
                "text_len=%d text_preview=%r",
                i, mode,
                sr.chunk.metadata.source,
                sr.chunk.metadata.page,
                sr.chunk.metadata.chunk_index,
                len(sr.chunk.text),
                sr.chunk.text[:120],
            )
