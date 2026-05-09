"""
Response generator for the Multimodal RAG System.

Builds a grounded prompt from retrieved chunks, calls the LLM backend,
and returns a structured response with citations reconstructed from
retrieval metadata — never from model output.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

from __future__ import annotations

import time

from rag.exceptions import GenerationError
from rag.generation.protocols import LLMBackend
from rag.logging_config import get_logger
from rag.models import Citation, GeneratedResponse, SearchResult

logger = get_logger(__name__)

# Answer returned when no chunks are available.
_NO_CONTEXT_ANSWER = (
    "I could not find relevant information in the provided documents to answer your question."
)


def _build_prompt(query: str, search_results: list[SearchResult]) -> str:
    """
    Build a grounded prompt that instructs the LLM to answer only from context.

    Each chunk is labelled with its 1-based index so the model can reference
    them naturally. The system instructions explicitly forbid hallucination.
    """
    context_blocks: list[str] = []
    for i, result in enumerate(search_results, start=1):
        meta = result.chunk.metadata
        block = (
            f"[{i}] Source: {meta.source} | Page: {meta.page} | "
            f"Chunk: {meta.chunk_index}\n"
            f"{result.chunk.text}"
        )
        context_blocks.append(block)

    context_section = "\n\n".join(context_blocks)

    prompt = (
        "You are a precise, citation-aware assistant. "
        "Answer the user's question using ONLY the context passages provided below. "
        "Do NOT use any knowledge outside of these passages. "
        "If the answer cannot be found in the context, say so clearly — "
        "do NOT guess or hallucinate.\n\n"
        "=== CONTEXT ===\n"
        f"{context_section}\n\n"
        "=== QUESTION ===\n"
        f"{query}\n\n"
        "=== ANSWER ===\n"
        "Provide a concise, accurate answer based solely on the context above."
    )
    return prompt


class ResponseGenerator:
    """
    Generates citation-aware answers from retrieved chunks.

    Citations are reconstructed directly from retrieval metadata — they are
    never parsed from the LLM's output text. This guarantees that every
    citation references a real, retrieved chunk.

    Args:
        llm_backend: Any object satisfying the ``LLMBackend`` protocol.
    """

    def __init__(self, llm_backend: LLMBackend) -> None:
        self._llm = llm_backend

    def generate(
        self,
        query: str,
        search_results: list[SearchResult],
    ) -> GeneratedResponse:
        """
        Generate a grounded answer with structured citations.

        When ``search_results`` is empty, returns a fixed "no information found"
        answer with an empty citations list — no LLM call is made.

        Args:
            query:          The user's natural language question.
            search_results: Ranked list of retrieved chunks from the vector store.

        Returns:
            A ``GeneratedResponse`` with ``answer`` text and ``citations`` list.

        Raises:
            GenerationError: If the LLM backend fails.
        """
        if not search_results:
            logger.info(
                "Generation skipped: no search results for query='%s'",
                query[:100],
            )
            return GeneratedResponse(answer=_NO_CONTEXT_ANSWER, citations=[])

        prompt = _build_prompt(query, search_results)
        start_time = time.monotonic()

        logger.info(
            "Generation started: query='%s' chunk_count=%d prompt_chars=%d",
            query[:100],
            len(search_results),
            len(prompt),
        )

        answer = self._llm.complete(prompt)

        duration = time.monotonic() - start_time

        # Build citations from retrieval metadata — NOT from model output.
        # This is the hallucination prevention guarantee: citations are always
        # grounded in what was actually retrieved.
        citations: list[Citation] = [
            Citation(
                source=result.chunk.metadata.source,
                page=result.chunk.metadata.page,
                chunk_index=result.chunk.metadata.chunk_index,
                score=result.score,
            )
            for result in search_results
        ]

        logger.info(
            "Generation complete: query='%s' chunk_count=%d citation_count=%d "
            "answer_chars=%d duration_s=%.3f",
            query[:100],
            len(search_results),
            len(citations),
            len(answer),
            duration,
        )

        return GeneratedResponse(answer=answer, citations=citations)
