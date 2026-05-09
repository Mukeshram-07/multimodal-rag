"""
HTTP API client for the Multimodal RAG System frontend.

All backend communication goes through this module. The Streamlit app
never imports backend services directly — it only calls these functions.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

# Default base URL — can be overridden via environment variable.
_DEFAULT_BASE_URL = "http://localhost:8000"

# Timeout for all API calls (seconds).
_TIMEOUT = 60.0


@dataclass
class IngestResult:
    status: str
    chunk_count: int
    collection_name: str


@dataclass
class Citation:
    source: str
    page: int
    chunk_index: int
    score: float


@dataclass
class QueryResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class APIError:
    """Represents a failed API call."""
    status_code: int
    message: str


def _base_url() -> str:
    import os
    return os.environ.get("RAG_API_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    collection_name: str,
) -> IngestResult | APIError:
    """POST /ingest — upload a PDF and ingest it into the vector store."""
    try:
        response = httpx.post(
            f"{_base_url()}/ingest",
            files={"file": (filename, file_bytes, "application/pdf")},
            data={"collection_name": collection_name},
            timeout=_TIMEOUT,
        )
        if response.status_code == 200:
            data = response.json()
            return IngestResult(
                status=data["status"],
                chunk_count=data["chunk_count"],
                collection_name=data["collection_name"],
            )
        data = response.json()
        return APIError(
            status_code=response.status_code,
            message=data.get("detail", response.text),
        )
    except httpx.ConnectError:
        return APIError(status_code=0, message="Cannot connect to the API server. Is it running?")
    except httpx.TimeoutException:
        return APIError(status_code=0, message="Request timed out. The server may be overloaded.")
    except Exception as exc:
        return APIError(status_code=0, message=f"Unexpected error: {exc}")


def query_documents(
    query: str,
    collection_name: str,
    top_k: int = 5,
    filter_source: str | None = None,
) -> QueryResult | APIError:
    """POST /query — retrieve and generate a grounded answer."""
    payload: dict[str, Any] = {
        "query": query,
        "collection_name": collection_name,
        "top_k": top_k,
    }
    if filter_source:
        payload["filter_source"] = filter_source

    try:
        response = httpx.post(
            f"{_base_url()}/query",
            json=payload,
            timeout=_TIMEOUT,
        )
        if response.status_code == 200:
            data = response.json()
            citations = [
                Citation(
                    source=c["source"],
                    page=c["page"],
                    chunk_index=c["chunk_index"],
                    score=c["score"],
                )
                for c in data.get("citations", [])
            ]
            return QueryResult(answer=data["answer"], citations=citations)
        data = response.json()
        return APIError(
            status_code=response.status_code,
            message=data.get("detail", response.text),
        )
    except httpx.ConnectError:
        return APIError(status_code=0, message="Cannot connect to the API server. Is it running?")
    except httpx.TimeoutException:
        return APIError(status_code=0, message="Request timed out.")
    except Exception as exc:
        return APIError(status_code=0, message=f"Unexpected error: {exc}")


def list_collections() -> list[str] | APIError:
    """GET /collections — return all collection names."""
    try:
        response = httpx.get(f"{_base_url()}/collections", timeout=_TIMEOUT)
        if response.status_code == 200:
            return response.json().get("collections", [])
        return APIError(status_code=response.status_code, message=response.text)
    except httpx.ConnectError:
        return APIError(status_code=0, message="Cannot connect to the API server.")
    except Exception as exc:
        return APIError(status_code=0, message=f"Unexpected error: {exc}")


def delete_collection(collection_name: str) -> bool | APIError:
    """DELETE /collections/{name} — delete a collection."""
    try:
        response = httpx.delete(
            f"{_base_url()}/collections/{collection_name}",
            timeout=_TIMEOUT,
        )
        if response.status_code == 200:
            return True
        data = response.json()
        return APIError(
            status_code=response.status_code,
            message=data.get("detail", response.text),
        )
    except httpx.ConnectError:
        return APIError(status_code=0, message="Cannot connect to the API server.")
    except Exception as exc:
        return APIError(status_code=0, message=f"Unexpected error: {exc}")
