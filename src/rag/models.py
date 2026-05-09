"""
Pydantic v2 data models for the Multimodal RAG System.

These models are the shared data contracts used across all components:
ingestion, embedding, storage, retrieval, generation, and the API layer.

Requirements: 10.1
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------


class DocumentMetadata(BaseModel):
    """Metadata attached to every Chunk produced by the ingestion pipeline."""

    source: str = Field(..., description="Original filename of the source document")
    page: int = Field(..., ge=1, description="1-indexed page number within the document")
    chunk_index: int = Field(..., ge=0, description="0-indexed position of this chunk within the document")


class Chunk(BaseModel):
    """A discrete segment of text extracted from a document, with its metadata."""

    text: str = Field(..., description="The raw text content of this chunk")
    metadata: DocumentMetadata = Field(..., description="Source provenance for this chunk")


class PageContent(BaseModel):
    """Raw text extracted from a single page of a document."""

    page_number: int = Field(..., ge=1, description="1-indexed page number")
    text: str = Field(..., description="Full text content of the page")


# ---------------------------------------------------------------------------
# Retrieval models
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    """A single result returned by the vector store search."""

    chunk: Chunk = Field(..., description="The retrieved chunk")
    score: float = Field(..., description="Cosine similarity score; higher means more similar")


# ---------------------------------------------------------------------------
# Generation models
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A reference to the source document location used in a generated answer."""

    source: str = Field(..., description="Original filename of the source document")
    page: int = Field(..., ge=1, description="1-indexed page number")
    chunk_index: int = Field(..., ge=0, description="0-indexed chunk position")
    score: float = Field(..., description="Similarity score of the retrieved chunk")


class GeneratedResponse(BaseModel):
    """The output of the response generator: an answer with supporting citations."""

    answer: str = Field(..., description="The generated natural language answer")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Structured citations referencing the source chunks used",
    )


# ---------------------------------------------------------------------------
# Pipeline result models
# ---------------------------------------------------------------------------


class IngestionResult(BaseModel):
    """Result returned by the ingestion pipeline after processing a document."""

    status: str = Field(..., description="Human-readable status message")
    chunk_count: int = Field(..., ge=0, description="Number of chunks stored")
    collection_name: str = Field(..., description="Name of the collection the chunks were stored in")


class RetrievalResult(BaseModel):
    """Result returned by the retriever for a given query."""

    chunks: list[SearchResult] = Field(
        default_factory=list,
        description="Ranked list of retrieved chunks",
    )
    status: str = Field(..., description="Human-readable status message")


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for the POST /query endpoint."""

    query: str = Field(..., min_length=1, description="The natural language query string")
    collection_name: str = Field(default="default", description="Name of the collection to query")
    top_k: int = Field(default=5, ge=1, description="Maximum number of chunks to retrieve")
    filter_source: str | None = Field(
        default=None,
        description="Optional filename filter to restrict retrieval to a specific document",
    )


class IngestResponse(BaseModel):
    """Response body for the POST /ingest endpoint."""

    status: str = Field(..., description="Human-readable ingestion status")
    chunk_count: int = Field(..., ge=0, description="Number of chunks stored")
    collection_name: str = Field(..., description="Name of the collection the document was stored in")


class HealthResponse(BaseModel):
    """Response body for the GET /health endpoint."""

    status: str = Field(..., description="Operational status, e.g. 'ok'")
    version: str = Field(..., description="Application version string")
