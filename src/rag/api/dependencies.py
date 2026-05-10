"""
FastAPI dependency providers for the Multimodal RAG System.

All heavy components (EmbeddingService, VectorStore, etc.) are initialised
once at module import time and reused across requests via module-level
singletons. This avoids reloading the embedding model on every request.

Requirements: 10.5
"""

from __future__ import annotations

from functools import lru_cache

from rag.config import Settings, get_settings
from rag.embedding.embedding_service import EmbeddingService
from rag.generation.llm_backends import OpenAICompatibleBackend
from rag.generation.response_generator import ResponseGenerator
from rag.ingestion.chunker import Chunker
from rag.ingestion.pdf_parser import PDFParser
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.retriever import Retriever
from rag.storage.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Singleton accessors — each is initialised once per process
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(model_name=settings.embedding_model)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    settings = get_settings()
    return VectorStore(persist_directory=settings.chroma_persist_dir)


@lru_cache(maxsize=1)
def get_ingestion_pipeline() -> IngestionPipeline:
    settings = get_settings()
    return IngestionPipeline(
        parser=PDFParser(),
        chunker=Chunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )


@lru_cache(maxsize=1)
def get_response_generator() -> ResponseGenerator:
    settings = get_settings()
    llm = OpenAICompatibleBackend(
        api_base=settings.llm_api_base,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        provider=settings.llm_provider,
    )
    return ResponseGenerator(llm_backend=llm)
