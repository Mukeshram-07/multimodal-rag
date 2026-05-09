"""
FastAPI integration tests using TestClient.

All heavy dependencies (EmbeddingService, VectorStore, IngestionPipeline,
Retriever, ResponseGenerator) are replaced with lightweight mocks via
FastAPI's dependency_overrides mechanism — no real model loading, no real
ChromaDB, no real LLM calls.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rag.api.dependencies import (
    get_ingestion_pipeline,
    get_response_generator,
    get_retriever,
    get_vector_store,
)
from rag.api.main import app
from rag.exceptions import IngestionError
from rag.models import (
    Chunk,
    Citation,
    DocumentMetadata,
    GeneratedResponse,
    IngestionResult,
    RetrievalResult,
    SearchResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(text: str = "chunk text", source: str = "doc.pdf", page: int = 1, idx: int = 0) -> Chunk:
    return Chunk(text=text, metadata=DocumentMetadata(source=source, page=page, chunk_index=idx))


def _make_search_result(text: str = "chunk text", score: float = 0.9) -> SearchResult:
    return SearchResult(chunk=_make_chunk(text), score=score)


def _make_citation(source: str = "doc.pdf", page: int = 1, idx: int = 0, score: float = 0.9) -> Citation:
    return Citation(source=source, page=page, chunk_index=idx, score=score)


def _minimal_pdf_bytes() -> bytes:
    """Return the bytes of a minimal valid PDF (same builder as conftest)."""
    objects: list[bytes] = []
    objects.append(b"")  # catalog placeholder
    objects.append(b"")  # page tree placeholder

    page_text = "Test PDF content for API testing."
    safe_text = page_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream_content = f"BT\n/F1 12 Tf\n50 750 Td\n({safe_text}) Tj\nET\n".encode("latin-1")

    content_obj_num = 3
    content_obj = (
        f"{content_obj_num} 0 obj\n<< /Length {len(stream_content)} >>\nstream\n"
    ).encode("ascii") + stream_content + b"\nendstream\nendobj\n"
    objects.append(content_obj)

    page_obj_num = 4
    page_obj = (
        f"{page_obj_num} 0 obj\n"
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        f"/Contents {content_obj_num} 0 R "
        f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>\n"
        f"endobj\n"
    ).encode("ascii")
    objects.append(page_obj)

    objects[0] = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    objects[1] = f"2 0 obj\n<< /Type /Pages /Kids [{page_obj_num} 0 R] /Count 1 >>\nendobj\n".encode("ascii")

    header = b"%PDF-1.4\n"
    body = bytearray(header)
    offsets = []
    for obj in objects:
        offsets.append(len(body))
        body.extend(obj)

    xref_offset = len(body)
    xref = bytearray(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    xref.extend(b"0000000000 65535 f \n")
    for off in offsets:
        xref.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")

    return bytes(body) + bytes(xref) + trailer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.ingest.return_value = IngestionResult(
        status="Ingested 3 chunks from 'test.pdf'",
        chunk_count=3,
        collection_name="default",
    )
    return pipeline


@pytest.fixture
def mock_retriever() -> MagicMock:
    retriever = MagicMock()
    retriever.retrieve.return_value = RetrievalResult(
        chunks=[_make_search_result("relevant chunk", score=0.95)],
        status="ok",
    )
    return retriever


@pytest.fixture
def mock_generator() -> MagicMock:
    generator = MagicMock()
    generator.generate.return_value = GeneratedResponse(
        answer="The answer is in the document.",
        citations=[_make_citation()],
    )
    return generator


@pytest.fixture
def mock_store() -> MagicMock:
    store = MagicMock()
    store.list_collections.return_value = ["default", "research"]
    return store


@pytest.fixture
def client(mock_pipeline, mock_retriever, mock_generator, mock_store) -> TestClient:
    """TestClient with all heavy dependencies overridden."""
    app.dependency_overrides[get_ingestion_pipeline] = lambda: mock_pipeline
    app.dependency_overrides[get_retriever] = lambda: mock_retriever
    app.dependency_overrides[get_response_generator] = lambda: mock_generator
    app.dependency_overrides[get_vector_store] = lambda: mock_store
    yield TestClient(app)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_returns_version(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "version" in data
        assert len(data["version"]) > 0


# ---------------------------------------------------------------------------
# Ingest endpoint
# ---------------------------------------------------------------------------


class TestIngestEndpoint:
    def test_ingest_returns_200_with_valid_pdf(self, client: TestClient) -> None:
        pdf_bytes = _minimal_pdf_bytes()
        response = client.post(
            "/ingest",
            files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            data={"collection_name": "default"},
        )
        assert response.status_code == 200

    def test_ingest_returns_chunk_count(self, client: TestClient) -> None:
        pdf_bytes = _minimal_pdf_bytes()
        data = client.post(
            "/ingest",
            files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            data={"collection_name": "default"},
        ).json()
        assert "chunk_count" in data
        assert data["chunk_count"] == 3

    def test_ingest_returns_collection_name(self, client: TestClient) -> None:
        pdf_bytes = _minimal_pdf_bytes()
        data = client.post(
            "/ingest",
            files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            data={"collection_name": "my_col"},
        ).json()
        # The mock always returns "default" — just verify the field exists
        assert "collection_name" in data

    def test_ingest_returns_status_string(self, client: TestClient) -> None:
        pdf_bytes = _minimal_pdf_bytes()
        data = client.post(
            "/ingest",
            files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            data={"collection_name": "default"},
        ).json()
        assert isinstance(data["status"], str)
        assert len(data["status"]) > 0

    def test_ingest_returns_422_for_ingestion_error(
        self, mock_pipeline: MagicMock, mock_retriever, mock_generator, mock_store
    ) -> None:
        mock_pipeline.ingest.side_effect = IngestionError(
            "Not a PDF", filename="bad.txt", reason="not a pdf"
        )
        app.dependency_overrides[get_ingestion_pipeline] = lambda: mock_pipeline
        app.dependency_overrides[get_retriever] = lambda: mock_retriever
        app.dependency_overrides[get_response_generator] = lambda: mock_generator
        app.dependency_overrides[get_vector_store] = lambda: mock_store
        c = TestClient(app, raise_server_exceptions=False)
        response = c.post(
            "/ingest",
            files={"file": ("bad.txt", io.BytesIO(b"not a pdf"), "text/plain")},
            data={"collection_name": "default"},
        )
        app.dependency_overrides.clear()
        assert response.status_code == 422

    def test_ingest_error_response_has_error_field(
        self, mock_pipeline: MagicMock, mock_retriever, mock_generator, mock_store
    ) -> None:
        mock_pipeline.ingest.side_effect = IngestionError("Bad file")
        app.dependency_overrides[get_ingestion_pipeline] = lambda: mock_pipeline
        app.dependency_overrides[get_retriever] = lambda: mock_retriever
        app.dependency_overrides[get_response_generator] = lambda: mock_generator
        app.dependency_overrides[get_vector_store] = lambda: mock_store
        c = TestClient(app, raise_server_exceptions=False)
        data = c.post(
            "/ingest",
            files={"file": ("bad.txt", io.BytesIO(b"x"), "text/plain")},
            data={"collection_name": "default"},
        ).json()
        app.dependency_overrides.clear()
        assert "error" in data
        assert "detail" in data


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    def test_query_returns_200(self, client: TestClient) -> None:
        response = client.post(
            "/query",
            json={"query": "What is the main topic?", "collection_name": "default"},
        )
        assert response.status_code == 200

    def test_query_returns_answer(self, client: TestClient) -> None:
        data = client.post(
            "/query",
            json={"query": "What is the main topic?", "collection_name": "default"},
        ).json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_query_returns_citations(self, client: TestClient) -> None:
        data = client.post(
            "/query",
            json={"query": "What is the main topic?", "collection_name": "default"},
        ).json()
        assert "citations" in data
        assert isinstance(data["citations"], list)

    def test_query_empty_retrieval_returns_200(
        self, mock_pipeline, mock_retriever: MagicMock, mock_generator: MagicMock, mock_store
    ) -> None:
        mock_retriever.retrieve.return_value = RetrievalResult(chunks=[], status="no results")
        mock_generator.generate.return_value = GeneratedResponse(
            answer="I could not find relevant information.", citations=[]
        )
        app.dependency_overrides[get_ingestion_pipeline] = lambda: mock_pipeline
        app.dependency_overrides[get_retriever] = lambda: mock_retriever
        app.dependency_overrides[get_response_generator] = lambda: mock_generator
        app.dependency_overrides[get_vector_store] = lambda: mock_store
        c = TestClient(app)
        response = c.post(
            "/query",
            json={"query": "unknown topic", "collection_name": "empty"},
        )
        app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        assert data["citations"] == []

    def test_query_validation_fails_for_empty_query(self, client: TestClient) -> None:
        response = client.post(
            "/query",
            json={"query": "", "collection_name": "default"},
        )
        assert response.status_code == 422

    def test_query_passes_top_k_to_retriever(
        self, mock_pipeline, mock_retriever: MagicMock, mock_generator, mock_store
    ) -> None:
        app.dependency_overrides[get_ingestion_pipeline] = lambda: mock_pipeline
        app.dependency_overrides[get_retriever] = lambda: mock_retriever
        app.dependency_overrides[get_response_generator] = lambda: mock_generator
        app.dependency_overrides[get_vector_store] = lambda: mock_store
        c = TestClient(app)
        c.post(
            "/query",
            json={"query": "test", "collection_name": "default", "top_k": 7},
        )
        app.dependency_overrides.clear()
        call_kwargs = mock_retriever.retrieve.call_args
        assert call_kwargs.kwargs.get("top_k") == 7 or (
            len(call_kwargs.args) >= 3 and call_kwargs.args[2] == 7
        )


# ---------------------------------------------------------------------------
# Collections endpoints
# ---------------------------------------------------------------------------


class TestCollectionsEndpoints:
    def test_list_collections_returns_200(self, client: TestClient) -> None:
        response = client.get("/collections")
        assert response.status_code == 200

    def test_list_collections_returns_list(self, client: TestClient) -> None:
        data = client.get("/collections").json()
        assert "collections" in data
        assert isinstance(data["collections"], list)

    def test_list_collections_contains_expected_names(self, client: TestClient) -> None:
        data = client.get("/collections").json()
        assert "default" in data["collections"]
        assert "research" in data["collections"]

    def test_delete_collection_returns_200(
        self, mock_pipeline, mock_retriever, mock_generator, mock_store: MagicMock
    ) -> None:
        mock_store.list_collections.return_value = ["to_delete"]
        app.dependency_overrides[get_ingestion_pipeline] = lambda: mock_pipeline
        app.dependency_overrides[get_retriever] = lambda: mock_retriever
        app.dependency_overrides[get_response_generator] = lambda: mock_generator
        app.dependency_overrides[get_vector_store] = lambda: mock_store
        c = TestClient(app)
        response = c.delete("/collections/to_delete")
        app.dependency_overrides.clear()
        assert response.status_code == 200

    def test_delete_collection_calls_store(
        self, mock_pipeline, mock_retriever, mock_generator, mock_store: MagicMock
    ) -> None:
        mock_store.list_collections.return_value = ["my_col"]
        app.dependency_overrides[get_ingestion_pipeline] = lambda: mock_pipeline
        app.dependency_overrides[get_retriever] = lambda: mock_retriever
        app.dependency_overrides[get_response_generator] = lambda: mock_generator
        app.dependency_overrides[get_vector_store] = lambda: mock_store
        c = TestClient(app)
        c.delete("/collections/my_col")
        app.dependency_overrides.clear()
        mock_store.delete_collection.assert_called_once_with("my_col")

    def test_delete_nonexistent_collection_returns_404(self, client: TestClient) -> None:
        response = client.delete("/collections/does_not_exist")
        assert response.status_code == 404
