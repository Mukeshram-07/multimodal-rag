"""
Unit tests for ResponseGenerator.

All tests use a mock LLMBackend — no real Ollama instance is required.

Requirements: 5.1, 5.5
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag.generation.response_generator import ResponseGenerator, _build_prompt, _NO_CONTEXT_ANSWER
from rag.models import Chunk, Citation, DocumentMetadata, GeneratedResponse, SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_search_result(
    text: str = "chunk text",
    source: str = "doc.pdf",
    page: int = 1,
    chunk_index: int = 0,
    score: float = 0.9,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            text=text,
            metadata=DocumentMetadata(source=source, page=page, chunk_index=chunk_index),
        ),
        score=score,
    )


def _make_mock_llm(answer: str = "The answer is 42.") -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = answer
    return llm


# ---------------------------------------------------------------------------
# Empty retrieval handling
# ---------------------------------------------------------------------------


class TestEmptyRetrievalHandling:
    """When search_results is empty, no LLM call is made and a fixed answer is returned."""

    def test_empty_results_returns_no_context_answer(self) -> None:
        gen = ResponseGenerator(llm_backend=_make_mock_llm())
        result = gen.generate("what is X?", [])
        assert _NO_CONTEXT_ANSWER in result.answer

    def test_empty_results_returns_empty_citations(self) -> None:
        gen = ResponseGenerator(llm_backend=_make_mock_llm())
        result = gen.generate("what is X?", [])
        assert result.citations == []

    def test_empty_results_llm_not_called(self) -> None:
        llm = _make_mock_llm()
        gen = ResponseGenerator(llm_backend=llm)
        gen.generate("what is X?", [])
        llm.complete.assert_not_called()

    def test_empty_results_returns_generated_response_type(self) -> None:
        gen = ResponseGenerator(llm_backend=_make_mock_llm())
        result = gen.generate("query", [])
        assert isinstance(result, GeneratedResponse)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    """The prompt includes the query text and chunk texts."""

    def test_prompt_contains_query(self) -> None:
        query = "What is the capital of France?"
        results = [_make_search_result("Paris is the capital.")]
        prompt = _build_prompt(query, results)
        assert query in prompt

    def test_prompt_contains_chunk_text(self) -> None:
        chunk_text = "The mitochondria is the powerhouse of the cell."
        results = [_make_search_result(chunk_text)]
        prompt = _build_prompt(query="biology question", search_results=results)
        assert chunk_text in prompt

    def test_prompt_contains_source_metadata(self) -> None:
        results = [_make_search_result(source="biology.pdf", page=3)]
        prompt = _build_prompt("question", results)
        assert "biology.pdf" in prompt
        assert "3" in prompt

    def test_prompt_contains_all_chunks(self) -> None:
        results = [
            _make_search_result(f"chunk {i}", chunk_index=i) for i in range(3)
        ]
        prompt = _build_prompt("question", results)
        for i in range(3):
            assert f"chunk {i}" in prompt

    def test_prompt_contains_grounding_instruction(self) -> None:
        results = [_make_search_result()]
        prompt = _build_prompt("question", results)
        # Must contain some form of grounding instruction
        assert "ONLY" in prompt or "only" in prompt

    def test_prompt_contains_hallucination_prevention(self) -> None:
        results = [_make_search_result()]
        prompt = _build_prompt("question", results)
        lower = prompt.lower()
        assert "hallucinate" in lower or "do not" in lower or "don't" in lower

    def test_llm_called_with_prompt_containing_query(self) -> None:
        llm = _make_mock_llm()
        gen = ResponseGenerator(llm_backend=llm)
        query = "unique query string 12345"
        gen.generate(query, [_make_search_result()])
        call_args = llm.complete.call_args[0][0]
        assert query in call_args


# ---------------------------------------------------------------------------
# Citation reconstruction
# ---------------------------------------------------------------------------


class TestCitationReconstruction:
    """Citations are built from retrieval metadata, not from LLM output."""

    def test_citations_match_search_result_count(self) -> None:
        results = [_make_search_result(chunk_index=i) for i in range(3)]
        gen = ResponseGenerator(llm_backend=_make_mock_llm())
        response = gen.generate("query", results)
        assert len(response.citations) == 3

    def test_citation_source_matches_metadata(self) -> None:
        results = [_make_search_result(source="report.pdf")]
        gen = ResponseGenerator(llm_backend=_make_mock_llm())
        response = gen.generate("query", results)
        assert response.citations[0].source == "report.pdf"

    def test_citation_page_matches_metadata(self) -> None:
        results = [_make_search_result(page=7)]
        gen = ResponseGenerator(llm_backend=_make_mock_llm())
        response = gen.generate("query", results)
        assert response.citations[0].page == 7

    def test_citation_chunk_index_matches_metadata(self) -> None:
        results = [_make_search_result(chunk_index=42)]
        gen = ResponseGenerator(llm_backend=_make_mock_llm())
        response = gen.generate("query", results)
        assert response.citations[0].chunk_index == 42

    def test_citation_score_matches_search_result(self) -> None:
        results = [_make_search_result(score=0.87)]
        gen = ResponseGenerator(llm_backend=_make_mock_llm())
        response = gen.generate("query", results)
        assert abs(response.citations[0].score - 0.87) < 1e-6

    def test_citation_ordering_matches_retrieval_ordering(self) -> None:
        """Citations preserve the order of the input search results."""
        results = [
            _make_search_result(source=f"doc{i}.pdf", chunk_index=i, score=1.0 - i * 0.1)
            for i in range(4)
        ]
        gen = ResponseGenerator(llm_backend=_make_mock_llm())
        response = gen.generate("query", results)
        for i, citation in enumerate(response.citations):
            assert citation.source == f"doc{i}.pdf"
            assert citation.chunk_index == i

    def test_citations_are_citation_instances(self) -> None:
        results = [_make_search_result()]
        gen = ResponseGenerator(llm_backend=_make_mock_llm())
        response = gen.generate("query", results)
        for c in response.citations:
            assert isinstance(c, Citation)


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------


class TestResponseStructure:
    """GeneratedResponse has the correct shape."""

    def test_response_is_generated_response_type(self) -> None:
        gen = ResponseGenerator(llm_backend=_make_mock_llm())
        result = gen.generate("query", [_make_search_result()])
        assert isinstance(result, GeneratedResponse)

    def test_answer_is_non_empty_string(self) -> None:
        gen = ResponseGenerator(llm_backend=_make_mock_llm("Some answer."))
        result = gen.generate("query", [_make_search_result()])
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_answer_matches_llm_output(self) -> None:
        expected = "The answer is forty-two."
        gen = ResponseGenerator(llm_backend=_make_mock_llm(expected))
        result = gen.generate("query", [_make_search_result()])
        assert result.answer == expected


# ---------------------------------------------------------------------------
# Hallucination prevention
# ---------------------------------------------------------------------------


class TestHallucinationPrevention:
    """Citations come only from retrieved chunks, never from LLM output."""

    def test_citations_not_derived_from_llm_output(self) -> None:
        """Even if the LLM mentions a fake source, citations come from metadata."""
        llm = _make_mock_llm(
            "The answer is in fake_source.pdf page 99 chunk 999."
        )
        real_result = _make_search_result(source="real.pdf", page=1, chunk_index=0)
        gen = ResponseGenerator(llm_backend=llm)
        response = gen.generate("query", [real_result])

        # Only one citation, from the real retrieved chunk
        assert len(response.citations) == 1
        assert response.citations[0].source == "real.pdf"
        assert response.citations[0].page == 1
        assert response.citations[0].chunk_index == 0

    def test_citation_count_equals_retrieved_chunk_count(self) -> None:
        """Number of citations always equals number of retrieved chunks."""
        for n in [1, 3, 5]:
            results = [_make_search_result(chunk_index=i) for i in range(n)]
            gen = ResponseGenerator(llm_backend=_make_mock_llm())
            response = gen.generate("query", results)
            assert len(response.citations) == n, (
                f"Expected {n} citations for {n} chunks, got {len(response.citations)}"
            )

    def test_metadata_preserved_regardless_of_llm_answer(self) -> None:
        """Metadata in citations is always from retrieval, not influenced by LLM."""
        llm = _make_mock_llm("I don't know anything about this topic.")
        result = _make_search_result(source="truth.pdf", page=5, chunk_index=3)
        gen = ResponseGenerator(llm_backend=llm)
        response = gen.generate("query", [result])

        assert response.citations[0].source == "truth.pdf"
        assert response.citations[0].page == 5
        assert response.citations[0].chunk_index == 3
