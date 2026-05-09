"""
Unit tests for PDFParser.

Tests:
  - Parsing a known small PDF fixture: verify page count and text content.
  - Parsing a non-PDF file: verify IngestionError is raised with a descriptive message.

Requirements: 1.2, 1.5
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.exceptions import IngestionError
from rag.ingestion.pdf_parser import PDFParser
from rag.models import PageContent


class TestPDFParserWithValidPDF:
    """Tests for PDFParser against the known small PDF fixture."""

    def test_returns_correct_page_count(self, small_pdf_path: Path) -> None:
        """The fixture PDF has exactly 2 pages."""
        parser = PDFParser()
        pages = parser.parse(str(small_pdf_path))
        assert len(pages) == 2

    def test_returns_list_of_page_content(self, small_pdf_path: Path) -> None:
        """Every item in the result must be a PageContent instance."""
        parser = PDFParser()
        pages = parser.parse(str(small_pdf_path))
        for page in pages:
            assert isinstance(page, PageContent)

    def test_page_numbers_are_one_indexed(self, small_pdf_path: Path) -> None:
        """Page numbers must start at 1 and be sequential."""
        parser = PDFParser()
        pages = parser.parse(str(small_pdf_path))
        for expected_num, page in enumerate(pages, start=1):
            assert page.page_number == expected_num

    def test_first_page_contains_expected_text(self, small_pdf_path: Path) -> None:
        """The first page should contain text from the fixture's first page."""
        parser = PDFParser()
        pages = parser.parse(str(small_pdf_path))
        # The fixture's first page contains "first page" in its text
        assert "first page" in pages[0].text.lower()

    def test_second_page_contains_expected_text(self, small_pdf_path: Path) -> None:
        """The second page should contain text from the fixture's second page."""
        parser = PDFParser()
        pages = parser.parse(str(small_pdf_path))
        # The fixture's second page contains "second page" in its text
        assert "second page" in pages[1].text.lower()

    def test_page_text_is_string(self, small_pdf_path: Path) -> None:
        """Each page's text attribute must be a string."""
        parser = PDFParser()
        pages = parser.parse(str(small_pdf_path))
        for page in pages:
            assert isinstance(page.text, str)


class TestPDFParserWithInvalidInput:
    """Tests for PDFParser error handling."""

    def test_non_pdf_file_raises_ingestion_error(self, tmp_path: Path) -> None:
        """Passing a plain text file should raise IngestionError."""
        non_pdf = tmp_path / "not_a_pdf.txt"
        non_pdf.write_text("This is not a PDF file.", encoding="utf-8")

        parser = PDFParser()
        with pytest.raises(IngestionError):
            parser.parse(str(non_pdf))

    def test_ingestion_error_contains_filename(self, tmp_path: Path) -> None:
        """The IngestionError should include the filename in its attributes."""
        non_pdf = tmp_path / "bad_file.txt"
        non_pdf.write_bytes(b"not a pdf")

        parser = PDFParser()
        with pytest.raises(IngestionError) as exc_info:
            parser.parse(str(non_pdf))

        error = exc_info.value
        assert error.filename == "bad_file.txt"

    def test_ingestion_error_has_descriptive_message(self, tmp_path: Path) -> None:
        """The IngestionError message should be non-empty and descriptive."""
        non_pdf = tmp_path / "garbage.bin"
        non_pdf.write_bytes(b"\x00\x01\x02\x03" * 100)

        parser = PDFParser()
        with pytest.raises(IngestionError) as exc_info:
            parser.parse(str(non_pdf))

        error = exc_info.value
        assert len(str(error)) > 0
        assert error.reason is not None

    def test_nonexistent_file_raises_ingestion_error(self, tmp_path: Path) -> None:
        """Passing a path that does not exist should raise IngestionError."""
        missing = tmp_path / "does_not_exist.pdf"

        parser = PDFParser()
        with pytest.raises(IngestionError):
            parser.parse(str(missing))
