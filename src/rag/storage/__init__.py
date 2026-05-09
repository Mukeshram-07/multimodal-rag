"""
Storage module for the Multimodal RAG System.

Exports:
  - VectorStore: ChromaDB-backed vector store for chunk embeddings.
"""

from rag.storage.vector_store import VectorStore

__all__ = ["VectorStore"]
