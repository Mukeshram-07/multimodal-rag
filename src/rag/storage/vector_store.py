"""
Vector store component for the Multimodal RAG System.

Persists and retrieves chunk embeddings using ChromaDB with local persistence.
Supports upsert semantics (deterministic IDs prevent duplicate entries on
re-ingestion), cosine-similarity search, collection management, and optional
source-level filtering.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7
"""

from __future__ import annotations

import chromadb
from chromadb.errors import InvalidCollectionException

from rag.logging_config import get_logger
from rag.models import Chunk, DocumentMetadata, SearchResult

logger = get_logger(__name__)


class VectorStore:
    """
    Persists and retrieves chunk embeddings using ChromaDB.

    Args:
        persist_directory: Path to the directory where ChromaDB stores its data.
    """

    def __init__(self, persist_directory: str) -> None:
        self.persist_directory = persist_directory
        logger.info("Initialising VectorStore: persist_directory=%s", persist_directory)
        self._client = chromadb.PersistentClient(path=persist_directory)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_id(chunk: Chunk) -> str:
        """Return a deterministic, stable ID for a chunk."""
        m = chunk.metadata
        return f"{m.source}_{m.page}_{m.chunk_index}"

    def _get_or_create_collection(self, collection_name: str) -> chromadb.Collection:
        """Return the named collection, creating it (with cosine space) if absent."""
        return self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        collection_name: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Upsert chunks and their embeddings into the named collection.

        Uses deterministic IDs (``{source}_{page}_{chunk_index}``) so that
        re-ingesting the same document does not create duplicate entries.

        Args:
            collection_name: Name of the target collection (created if absent).
            chunks:          List of ``Chunk`` objects to store.
            embeddings:      Pre-computed embedding vectors, one per chunk.
        """
        if not chunks:
            logger.info("add called with empty chunks list; nothing to store.")
            return

        collection = self._get_or_create_collection(collection_name)

        ids = [self._make_id(c) for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "source": c.metadata.source,
                "page": c.metadata.page,
                "chunk_index": c.metadata.chunk_index,
            }
            for c in chunks
        ]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(
            "Upserted chunks: collection=%s count=%d",
            collection_name,
            len(chunks),
        )

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int,
        filter_source: str | None = None,
    ) -> list[SearchResult]:
        """
        Search for the most similar chunks to ``query_embedding``.

        Args:
            collection_name: Name of the collection to search.
            query_embedding: Pre-computed query embedding vector.
            top_k:           Maximum number of results to return.
            filter_source:   If provided, restrict results to chunks whose
                             ``source`` metadata field equals this value.

        Returns:
            A list of ``SearchResult`` objects ordered by descending similarity
            score. Returns an empty list (without raising) if the collection
            does not exist or contains no items.
        """
        # Return empty list if collection does not exist.
        try:
            collection = self._client.get_collection(
                name=collection_name,
                embedding_function=None,
            )
        except (InvalidCollectionException, ValueError):
            logger.info(
                "search: collection '%s' does not exist; returning empty list.",
                collection_name,
            )
            return []

        # Guard against querying an empty collection (ChromaDB raises on n_results > count).
        item_count = collection.count()
        if item_count == 0:
            return []

        effective_k = min(top_k, item_count)

        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": effective_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_source is not None:
            query_kwargs["where"] = {"source": filter_source}

        results = collection.query(**query_kwargs)

        # Unpack the first (and only) query's results.
        ids_list = results["ids"][0]
        documents_list = results["documents"][0]
        metadatas_list = results["metadatas"][0]
        distances_list = results["distances"][0]

        search_results: list[SearchResult] = []
        for doc, meta, distance in zip(documents_list, metadatas_list, distances_list):
            # With cosine space: distance is cosine distance in [0, 2].
            # score = 1 - distance gives cosine similarity in [-1, 1].
            score = 1.0 - distance

            chunk = Chunk(
                text=doc,
                metadata=DocumentMetadata(
                    source=meta["source"],
                    page=int(meta["page"]),
                    chunk_index=int(meta["chunk_index"]),
                ),
            )
            search_results.append(SearchResult(chunk=chunk, score=score))

        # Ensure descending order by score (ChromaDB returns ascending distance).
        search_results.sort(key=lambda r: r.score, reverse=True)

        logger.info(
            "Search complete: collection=%s top_k=%d results=%d",
            collection_name,
            top_k,
            len(search_results),
        )
        return search_results

    def list_collections(self) -> list[str]:
        """
        Return the names of all collections in the store.

        Returns:
            A list of collection name strings.
        """
        collections = self._client.list_collections()
        return [col.name for col in collections]

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete the named collection and all its data.

        Args:
            collection_name: Name of the collection to delete.
        """
        self._client.delete_collection(name=collection_name)
        logger.info("Deleted collection: name=%s", collection_name)
