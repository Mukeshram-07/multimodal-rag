"""
Property-based tests for ResponseGenerator.

# Feature: multimodal-rag, Property 5: Citation completeness

Property 5: Citation completeness
  For any GeneratedResponse produced from a non-empty list of SearchResults,
  every Citation SHALL reference a source, page, and chunk_index that exists
  in the input SearchResult list.

Validates: Requirements 5.3, 5.4
"""

from __future__ import annotations

from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from rag.generation.response_generator import ResponseGenerator
from rag.models import Chunk, DocumentMetadata, SearchResult


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

source_strategy = st.from_regex(r"[a-z][a-z0-9]{1,15}\.pdf", fullmatch=True)
page_strategy = st.integers(min_value=1, max_value=500)
chunk_index_strategy = st.integers(min_value=0, max_value=999)
score_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
text_strategy = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())


def _search_result_strategy() -> st.SearchStrategy[SearchResult]:
    return st.builds(
        SearchResult,
        chunk=st.builds(
            Chunk,
            text=text_strategy,
            metadata=st.builds(
                DocumentMetadata,
                source=source_strategy,
                page=page_strategy,
                chunk_index=chunk_index_strategy,
            ),
        ),
        score=score_strategy,
    )


# ---------------------------------------------------------------------------
# Property 5: Citation completeness
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(search_results=st.lists(_search_result_strategy(), min_size=1, max_size=10))
def test_citation_completeness(search_results: list[SearchResult]) -> None:
    """
    # Feature: multimodal-rag, Property 5: Citation completeness
    Validates: Requirements 5.3, 5.4

    For any non-empty list of SearchResults:
    - Every Citation references a (source, page, chunk_index) that exists
      in the input SearchResult list.
    - answer is a non-empty string.
    - citation count equals the number of input search results.
    """
    llm = MagicMock()
    llm.complete.return_value = "This is the generated answer."

    gen = ResponseGenerator(llm_backend=llm)
    response = gen.generate("test query", search_results)

    # Build a set of valid (source, page, chunk_index) tuples from input
    valid_keys = {
        (r.chunk.metadata.source, r.chunk.metadata.page, r.chunk.metadata.chunk_index)
        for r in search_results
    }

    # Assert: answer is non-empty
    assert isinstance(response.answer, str) and len(response.answer) > 0, (
        "answer must be a non-empty string"
    )

    # Assert: citation count equals retrieved chunk count
    assert len(response.citations) == len(search_results), (
        f"Expected {len(search_results)} citations, got {len(response.citations)}"
    )

    # Assert: every citation references a real retrieved chunk
    for citation in response.citations:
        key = (citation.source, citation.page, citation.chunk_index)
        assert key in valid_keys, (
            f"Citation {key} does not correspond to any input SearchResult. "
            f"Valid keys: {valid_keys}"
        )
