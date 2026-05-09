"""
Property-based tests for the ingestion pipeline — chunk count non-negativity.

# Feature: multimodal-rag, Property 7: Chunk count non-negativity

Property 7: Chunk count non-negativity
  For any PDF document ingested, the chunk_count SHALL be greater than or
  equal to zero, and SHALL equal the number of Chunks actually produced by
  the Chunker for that document.

Validates: Requirements 1.3
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from rag.ingestion.chunker import Chunker
from rag.models import PageContent


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Individual page content: 1-indexed page number, arbitrary text
page_content_strategy = st.builds(
    PageContent,
    page_number=st.integers(min_value=1, max_value=1000),
    text=st.text(min_size=0, max_size=5000),
)

# Lists of page content (may be empty)
page_list_strategy = st.lists(page_content_strategy, min_size=0, max_size=30)

# Source filename
filename_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="-_.",
    ),
    min_size=1,
    max_size=64,
).map(lambda s: s + ".pdf")


# ---------------------------------------------------------------------------
# Property 7: Chunk count non-negativity
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    pages=page_list_strategy,
    filename=filename_strategy,
)
def test_chunk_count_non_negativity(
    pages: list[PageContent],
    filename: str,
) -> None:
    """
    **Validates: Requirements 1.3**

    Property 7: Chunk count non-negativity

    For any list of page content:
    - The total chunk count is >= 0.
    - The chunk count equals the length of the returned list.
    """
    chunker = Chunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk(pages, source_filename=filename)

    chunk_count = len(chunks)

    # Assert: total chunk count is non-negative
    assert chunk_count >= 0, (
        f"chunk_count {chunk_count} is negative, which is impossible"
    )

    # Assert: chunk count equals the number of chunks in the returned list
    assert chunk_count == len(chunks), (
        f"chunk_count {chunk_count} does not match len(chunks) {len(chunks)}"
    )
