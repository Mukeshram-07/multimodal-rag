"""
Shared pytest fixtures for the Multimodal RAG System test suite.

Provides:
  - tmp_chroma_dir:   A temporary directory for ChromaDB persistence (cleaned up after each test).
  - mock_llm_backend: A mock LLMBackend that returns a fixed answer string.
  - small_pdf_path:   Path to a small, real PDF fixture file used across tests.

Requirements: 10.5 (dependency injection / testability)
"""

from __future__ import annotations

import io
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Protocol
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Temporary ChromaDB directory fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_chroma_dir(tmp_path: Path) -> Path:
    """
    Return a temporary directory path for ChromaDB persistence.

    The directory is automatically cleaned up after each test by pytest's
    built-in ``tmp_path`` fixture.
    """
    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chroma_dir


# ---------------------------------------------------------------------------
# Mock LLM backend fixture
# ---------------------------------------------------------------------------


class LLMBackendProtocol(Protocol):
    """Minimal protocol matching the LLMBackend interface."""

    def complete(self, prompt: str) -> str:
        ...


@pytest.fixture
def mock_llm_backend() -> MagicMock:
    """
    Return a mock object that satisfies the LLMBackend protocol.

    The mock's ``complete`` method returns a fixed answer string by default.
    Tests can override the return value via ``mock_llm_backend.complete.return_value``.
    """
    backend = MagicMock(spec=LLMBackendProtocol)
    backend.complete.return_value = (
        "Based on the provided context, the answer is found in the document. "
        "The relevant information appears on the referenced pages."
    )
    return backend


# ---------------------------------------------------------------------------
# Small PDF fixture
# ---------------------------------------------------------------------------


def _build_minimal_pdf(text_pages: list[str]) -> bytes:
    """
    Build a minimal but valid PDF in memory containing the given text pages.

    This avoids a dependency on external PDF files in the test suite while
    still producing a file that PyMuPDF can parse correctly.
    """
    # We'll build a simple PDF with one content stream per page.
    # PDF structure: header, objects, xref table, trailer.

    objects: list[bytes] = []
    offsets: list[int] = []

    def add_object(content: bytes) -> int:
        """Append an object and return its 1-based object number."""
        obj_num = len(objects) + 1
        objects.append(content)
        return obj_num

    # Object 1: Catalog (added last after we know page tree object number)
    # Object 2: Page tree (added after pages)
    # Objects 3+: Page objects and content streams

    page_obj_nums: list[int] = []
    content_obj_nums: list[int] = []

    # Reserve slots for catalog (1) and page tree (2)
    objects.append(b"")  # placeholder for catalog
    objects.append(b"")  # placeholder for page tree

    for page_text in text_pages:
        # Encode text as a PDF content stream using a basic font
        # Escape parentheses in the text
        safe_text = page_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_content = (
            f"BT\n/F1 12 Tf\n50 750 Td\n({safe_text}) Tj\nET\n"
        ).encode("latin-1")

        content_obj_num = len(objects) + 1
        content_obj = (
            f"{content_obj_num} 0 obj\n"
            f"<< /Length {len(stream_content)} >>\n"
            f"stream\n"
        ).encode("ascii") + stream_content + b"\nendstream\nendobj\n"
        objects.append(content_obj)
        content_obj_nums.append(content_obj_num)

        page_obj_num = len(objects) + 1
        page_obj = (
            f"{page_obj_num} 0 obj\n"
            f"<< /Type /Page\n"
            f"   /Parent 2 0 R\n"
            f"   /MediaBox [0 0 612 792]\n"
            f"   /Contents {content_obj_num} 0 R\n"
            f"   /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>\n"
            f">>\n"
            f"endobj\n"
        ).encode("ascii")
        objects.append(page_obj)
        page_obj_nums.append(page_obj_num)

    # Fill in catalog (object 1)
    catalog = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    objects[0] = catalog

    # Fill in page tree (object 2)
    kids = " ".join(f"{n} 0 R" for n in page_obj_nums)
    page_tree = (
        f"2 0 obj\n"
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_obj_nums)} >>\n"
        f"endobj\n"
    ).encode("ascii")
    objects[1] = page_tree

    # Assemble the PDF body
    header = b"%PDF-1.4\n"
    body = bytearray(header)

    offsets = []
    for obj in objects:
        offsets.append(len(body))
        body.extend(obj)

    # Cross-reference table
    xref_offset = len(body)
    xref = bytearray(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    xref.extend(b"0000000000 65535 f \n")
    for off in offsets:
        xref.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    trailer = (
        f"trailer\n"
        f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n"
        f"{xref_offset}\n"
        f"%%EOF\n"
    ).encode("ascii")

    return bytes(body) + bytes(xref) + trailer


@pytest.fixture(scope="session")
def small_pdf_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Return the path to a small, valid PDF fixture file.

    The PDF contains two pages with known text content, making it suitable
    for testing the PDF parser, chunker, and ingestion pipeline.

    Scoped to the session so the file is created only once.
    """
    pdf_dir = tmp_path_factory.mktemp("pdf_fixtures")
    pdf_path = pdf_dir / "sample.pdf"

    pages = [
        "This is the first page of the sample PDF document. "
        "It contains some text that can be extracted and chunked for testing purposes. "
        "The RAG system should be able to parse this content correctly.",
        "This is the second page of the sample PDF document. "
        "It contains additional text to verify multi-page extraction. "
        "Citations should reference the correct page numbers after retrieval.",
    ]

    pdf_bytes = _build_minimal_pdf(pages)
    pdf_path.write_bytes(pdf_bytes)
    return pdf_path
