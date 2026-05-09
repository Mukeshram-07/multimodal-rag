"""
PDF parsing component for the Multimodal RAG System.

Uses PyMuPDF (fitz) to extract per-page text from PDF files.

Requirements: 1.2, 1.5
"""

import logging
from pathlib import Path

import fitz  # PyMuPDF

from rag.exceptions import IngestionError
from rag.logging_config import get_logger
from rag.models import PageContent

logger = get_logger(__name__)


class PDFParser:
    """
    Extracts text from PDF files on a per-page basis using PyMuPDF.

    Each page is returned as a ``PageContent`` with a 1-indexed page number
    and the full text extracted from that page.
    """

    def parse(self, file_path: str) -> list[PageContent]:
        """
        Open a PDF file and extract text from every page.

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            A list of ``PageContent`` objects, one per page, with 1-indexed
            page numbers.

        Raises:
            IngestionError: If the file cannot be opened or is not a valid PDF.
        """
        path = Path(file_path)
        filename = path.name

        logger.info("Starting PDF parsing: file=%s", filename)

        try:
            doc = fitz.open(file_path)
        except Exception as exc:
            reason = f"Could not open file: {exc}"
            logger.error("PDF parse failure: file=%s reason=%s", filename, reason)
            raise IngestionError(
                message=f"Failed to parse PDF '{filename}': {reason}",
                filename=filename,
                reason=reason,
            ) from exc

        # Verify the opened document is actually a PDF
        if not doc.is_pdf:
            doc.close()
            reason = "File is not a valid PDF document"
            logger.error("PDF parse failure: file=%s reason=%s", filename, reason)
            raise IngestionError(
                message=f"Failed to parse PDF '{filename}': {reason}",
                filename=filename,
                reason=reason,
            )

        try:
            pages: list[PageContent] = []
            page_count = len(doc)

            for page_index in range(page_count):
                page = doc[page_index]
                text = page.get_text()
                pages.append(
                    PageContent(
                        page_number=page_index + 1,  # 1-indexed
                        text=text,
                    )
                )

            logger.info(
                "PDF parsing complete: file=%s pages=%d",
                filename,
                page_count,
            )
            return pages

        except IngestionError:
            raise
        except Exception as exc:
            reason = f"Error extracting text: {exc}"
            logger.error("PDF text extraction failure: file=%s reason=%s", filename, reason)
            raise IngestionError(
                message=f"Failed to extract text from PDF '{filename}': {reason}",
                filename=filename,
                reason=reason,
            ) from exc
        finally:
            doc.close()
