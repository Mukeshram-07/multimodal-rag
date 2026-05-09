"""
POST /query endpoint — retrieval + generation route.

Requirements: 6.2
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from rag.api.dependencies import get_response_generator, get_retriever
from rag.generation.response_generator import ResponseGenerator
from rag.logging_config import get_logger
from rag.models import GeneratedResponse, QueryRequest
from rag.retrieval.retriever import Retriever

router = APIRouter()
logger = get_logger(__name__)


@router.post("/query", response_model=GeneratedResponse)
def query_documents(
    request: QueryRequest,
    retriever: Retriever = Depends(get_retriever),
    generator: ResponseGenerator = Depends(get_response_generator),
) -> GeneratedResponse:
    """
    Retrieve relevant chunks and generate a grounded answer.

    Embeds the query, searches the collection, and passes the results to
    the response generator. Returns the answer with structured citations.
    """
    start = time.monotonic()
    logger.info(
        "POST /query: query='%s' collection=%s top_k=%d",
        request.query[:100],
        request.collection_name,
        request.top_k,
    )

    retrieval_result = retriever.retrieve(
        query=request.query,
        collection_name=request.collection_name,
        top_k=request.top_k,
        filter_source=request.filter_source,
    )

    response = generator.generate(
        query=request.query,
        search_results=retrieval_result.chunks,
    )

    duration = time.monotonic() - start
    logger.info(
        "POST /query complete: citation_count=%d duration_s=%.3f",
        len(response.citations),
        duration,
    )

    return response
