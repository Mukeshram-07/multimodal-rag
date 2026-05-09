"""
Tests for the frontend API client.

All HTTP calls are intercepted with unittest.mock.patch so no real server
is required. Tests verify that the client correctly parses API responses,
handles errors, and maps data to the right dataclass fields.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.frontend.api_client import (
    APIError,
    Citation,
    IngestResult,
    QueryResult,
    delete_collection,
    ingest_pdf,
    list_collections,
    query_documents,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(status_code: int, json_data: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = str(json_data)
    return resp


# ---------------------------------------------------------------------------
# ingest_pdf
# ---------------------------------------------------------------------------


class TestIngestPdf:
    def test_successful_ingest_returns_ingest_result(self) -> None:
        mock_resp = _mock_response(200, {
            "status": "Ingested 5 chunks from 'doc.pdf'",
            "chunk_count": 5,
            "collection_name": "default",
        })
        with patch("httpx.post", return_value=mock_resp):
            result = ingest_pdf(b"%PDF-1.4", "doc.pdf", "default")
        assert isinstance(result, IngestResult)
        assert result.chunk_count == 5
        assert result.collection_name == "default"

    def test_ingest_error_returns_api_error(self) -> None:
        mock_resp = _mock_response(422, {"detail": "Not a valid PDF"})
        with patch("httpx.post", return_value=mock_resp):
            result = ingest_pdf(b"not a pdf", "bad.txt", "default")
        assert isinstance(result, APIError)
        assert result.status_code == 422
        assert "Not a valid PDF" in result.message

    def test_ingest_connection_error_returns_api_error(self) -> None:
        import httpx as _httpx
        with patch("httpx.post", side_effect=_httpx.ConnectError("refused")):
            result = ingest_pdf(b"%PDF", "doc.pdf", "default")
        assert isinstance(result, APIError)
        assert result.status_code == 0
        assert "connect" in result.message.lower() or "api" in result.message.lower()

    def test_ingest_timeout_returns_api_error(self) -> None:
        import httpx as _httpx
        with patch("httpx.post", side_effect=_httpx.TimeoutException("timeout")):
            result = ingest_pdf(b"%PDF", "doc.pdf", "default")
        assert isinstance(result, APIError)
        assert result.status_code == 0

    def test_ingest_status_field_preserved(self) -> None:
        mock_resp = _mock_response(200, {
            "status": "ok",
            "chunk_count": 2,
            "collection_name": "research",
        })
        with patch("httpx.post", return_value=mock_resp):
            result = ingest_pdf(b"%PDF", "doc.pdf", "research")
        assert isinstance(result, IngestResult)
        assert result.status == "ok"


# ---------------------------------------------------------------------------
# query_documents
# ---------------------------------------------------------------------------


class TestQueryDocuments:
    def test_successful_query_returns_query_result(self) -> None:
        mock_resp = _mock_response(200, {
            "answer": "The answer is 42.",
            "citations": [
                {"source": "doc.pdf", "page": 1, "chunk_index": 0, "score": 0.95}
            ],
        })
        with patch("httpx.post", return_value=mock_resp):
            result = query_documents("What is the answer?", "default")
        assert isinstance(result, QueryResult)
        assert result.answer == "The answer is 42."
        assert len(result.citations) == 1

    def test_citation_fields_mapped_correctly(self) -> None:
        mock_resp = _mock_response(200, {
            "answer": "Some answer.",
            "citations": [
                {"source": "report.pdf", "page": 3, "chunk_index": 7, "score": 0.88}
            ],
        })
        with patch("httpx.post", return_value=mock_resp):
            result = query_documents("query", "default")
        assert isinstance(result, QueryResult)
        c = result.citations[0]
        assert isinstance(c, Citation)
        assert c.source == "report.pdf"
        assert c.page == 3
        assert c.chunk_index == 7
        assert abs(c.score - 0.88) < 1e-6

    def test_empty_citations_returns_empty_list(self) -> None:
        mock_resp = _mock_response(200, {
            "answer": "No relevant information found.",
            "citations": [],
        })
        with patch("httpx.post", return_value=mock_resp):
            result = query_documents("unknown topic", "default")
        assert isinstance(result, QueryResult)
        assert result.citations == []

    def test_query_error_returns_api_error(self) -> None:
        mock_resp = _mock_response(500, {"detail": "Embedding service failed"})
        with patch("httpx.post", return_value=mock_resp):
            result = query_documents("query", "default")
        assert isinstance(result, APIError)
        assert result.status_code == 500

    def test_query_connection_error_returns_api_error(self) -> None:
        import httpx as _httpx
        with patch("httpx.post", side_effect=_httpx.ConnectError("refused")):
            result = query_documents("query", "default")
        assert isinstance(result, APIError)
        assert result.status_code == 0

    def test_multiple_citations_preserved_in_order(self) -> None:
        mock_resp = _mock_response(200, {
            "answer": "Answer.",
            "citations": [
                {"source": "a.pdf", "page": 1, "chunk_index": 0, "score": 0.9},
                {"source": "b.pdf", "page": 2, "chunk_index": 1, "score": 0.8},
                {"source": "c.pdf", "page": 3, "chunk_index": 2, "score": 0.7},
            ],
        })
        with patch("httpx.post", return_value=mock_resp):
            result = query_documents("query", "default")
        assert isinstance(result, QueryResult)
        assert [c.source for c in result.citations] == ["a.pdf", "b.pdf", "c.pdf"]


# ---------------------------------------------------------------------------
# list_collections
# ---------------------------------------------------------------------------


class TestListCollections:
    def test_returns_list_of_strings(self) -> None:
        mock_resp = _mock_response(200, {"collections": ["default", "research"]})
        with patch("httpx.get", return_value=mock_resp):
            result = list_collections()
        assert result == ["default", "research"]

    def test_empty_collections_returns_empty_list(self) -> None:
        mock_resp = _mock_response(200, {"collections": []})
        with patch("httpx.get", return_value=mock_resp):
            result = list_collections()
        assert result == []

    def test_connection_error_returns_api_error(self) -> None:
        import httpx as _httpx
        with patch("httpx.get", side_effect=_httpx.ConnectError("refused")):
            result = list_collections()
        assert isinstance(result, APIError)
        assert result.status_code == 0


# ---------------------------------------------------------------------------
# delete_collection
# ---------------------------------------------------------------------------


class TestDeleteCollection:
    def test_successful_delete_returns_true(self) -> None:
        mock_resp = _mock_response(200, {"status": "deleted", "collection_name": "old"})
        with patch("httpx.delete", return_value=mock_resp):
            result = delete_collection("old")
        assert result is True

    def test_not_found_returns_api_error(self) -> None:
        mock_resp = _mock_response(404, {"detail": "Collection 'x' not found"})
        with patch("httpx.delete", return_value=mock_resp):
            result = delete_collection("x")
        assert isinstance(result, APIError)
        assert result.status_code == 404

    def test_connection_error_returns_api_error(self) -> None:
        import httpx as _httpx
        with patch("httpx.delete", side_effect=_httpx.ConnectError("refused")):
            result = delete_collection("col")
        assert isinstance(result, APIError)
        assert result.status_code == 0
