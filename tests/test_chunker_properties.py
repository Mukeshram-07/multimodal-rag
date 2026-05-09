"""
Property-based tests for Chunker — chunk metadata integrity.

# Feature: multimodal-rag, Property 1: Chunk metadata integrity

Property 1: Chunk metadata integrity
  For any PDF document ingested into the system, every Chunk produced by the
  Ingestion_Pipeline SHALL have a DocumentMetadata whose `source` matches the
  input filename, `page` is within the valid page range of the document, and
  `chunk_index` is a non-negative integer unique within that page.

Validates: Requirements 1.3, 1.4
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from rag.ingestion.chunker import Chunker
from rag.models import PageContent


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Filenames: printable ASCII, non-empty, no path separators
filename_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="-_.",
    ),
    min_size=1,
    max_size=64,
).map(lambda s: s + ".pdf")

# Page text: any unicode text, 0–5000 characters
page_text_strategy = st.text(min_size=0, max_size=5000)

# Page count: 1–50
page_count_strategy = st.integers(min_value=1, max_value=50)


def build_pages(page_count: int, texts: list[str]) -> list[PageContent]:
    """Build a list of PageContent with 1-indexed page numbers."""
    return [
        PageContent(page_number=i + 1, text=texts[i])
        for i in range(page_count)
    ]


# ---------------------------------------------------------------------------
# Property 1: Chunk metadata integrity
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    filename=filename_strategy,
    page_count=page_count_strategy,
    texts=st.lists(page_text_strategy, min_size=1, max_size=50),
)
def test_chunk_metadata_integrity(
    filename: str,
    page_count: int,
    texts: list[str],
) -> None:
    """
    **Validates: Requirements 1.3, 1.4**

    Property 1: Chunk metadata integrity

    For any combination of filename, page count, and page texts:
    - Every chunk's metadata.source equals the input filename.
    - Every chunk's metadata.page is within [1, page_count].
    - Chunk indices within a page are unique and non-negative.
    """
    # Use only as many texts as we have pages (or pad with empty strings)
    actual_page_count = min(page_count, len(texts))
    if actual_page_count == 0:
        actual_page_count = 1
        texts = [""]

    pages = build_pages(actual_page_count, texts[:actual_page_count])

    chunker = Chunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk(pages, source_filename=filename)

    # Group chunk indices by page for uniqueness check
    page_to_chunk_indices: dict[int, list[int]] = {}

    for chunk in chunks:
        meta = chunk.metadata

        # Assert: source matches input filename
        assert meta.source == filename, (
            f"Expected source={filename!r}, got {meta.source!r}"
        )

        # Assert: page is within [1, actual_page_count]
        assert 1 <= meta.page <= actual_page_count, (
            f"Page {meta.page} is outside valid range [1, {actual_page_count}]"
        )

        # Assert: chunk_index is non-negative
        assert meta.chunk_index >= 0, (
            f"chunk_index {meta.chunk_index} is negative"
        )

        page_to_chunk_indices.setdefault(meta.page, []).append(meta.chunk_index)

    # Assert: chunk indices within each page are unique
    for page_num, indices in page_to_chunk_indices.items():
        assert len(indices) == len(set(indices)), (
            f"Duplicate chunk indices on page {page_num}: {indices}"
        )
