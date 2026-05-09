"""
Custom exception hierarchy for the Multimodal RAG System.

Each exception class maps to a specific component and HTTP status code:
  - IngestionError    → HTTP 422 (Unprocessable Entity)
  - EmbeddingError    → HTTP 500 (Internal Server Error)
  - RetrievalError    → HTTP 500 (Internal Server Error)
  - GenerationError   → HTTP 500 (Internal Server Error)
  - ConfigurationError → raised at startup, not mapped to HTTP

Requirements: 11.1, 11.2
"""


class RAGBaseError(Exception):
    """Base class for all RAG system exceptions."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class IngestionError(RAGBaseError):
    """
    Raised when a document cannot be parsed or the ingestion pipeline fails.

    Attributes:
        filename: The name of the file that caused the error (if known).
        reason:   A human-readable description of the failure.
    """

    def __init__(self, message: str, filename: str | None = None, reason: str | None = None) -> None:
        super().__init__(message)
        self.filename = filename
        self.reason = reason


class EmbeddingError(RAGBaseError):
    """
    Raised when the embedding service fails to encode one or more chunks.

    Attributes:
        chunk_index: The 0-based index of the chunk that caused the failure (if known).
        detail:      Additional context about the failure.
    """

    def __init__(self, message: str, chunk_index: int | None = None, detail: str | None = None) -> None:
        super().__init__(message)
        self.chunk_index = chunk_index
        self.detail = detail


class RetrievalError(RAGBaseError):
    """
    Raised when the vector store or retriever encounters an unexpected error.

    Attributes:
        collection_name: The collection that was being queried (if known).
    """

    def __init__(self, message: str, collection_name: str | None = None) -> None:
        super().__init__(message)
        self.collection_name = collection_name


class GenerationError(RAGBaseError):
    """
    Raised when the LLM backend fails to produce a response.

    Attributes:
        detail: Additional context such as the HTTP status code or timeout info.
    """

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.detail = detail


class ConfigurationError(RAGBaseError):
    """
    Raised at startup when a required configuration value is missing or invalid.

    Attributes:
        key: The name of the missing or invalid configuration key.
    """

    def __init__(self, message: str, key: str | None = None) -> None:
        super().__init__(message)
        self.key = key
