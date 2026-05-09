"""
Embedding module for the Multimodal RAG System.

Exports:
  - EmbeddingService: Wraps SentenceTransformer for batched text encoding.
  - EmbeddingBackend: Protocol for any embedding backend.
"""

from rag.embedding.embedding_service import EmbeddingService
from rag.embedding.protocols import EmbeddingBackend

__all__ = ["EmbeddingService", "EmbeddingBackend"]
