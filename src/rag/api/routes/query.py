"""
POST /query endpoint — retrieval + generation route (authenticated).
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from rag.api.dependencies import get_response_generator, get_retriever
from rag.auth.database import get_db
from rag.auth.deps import get_current_user
from rag.auth.models import User
from rag.auth.service import get_user_collection
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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> GeneratedResponse:
    """
    Retrieve relevant chunks and generate a grounded answer.

    Only searches the authenticated user's own collection.
    Returns 404 if the collection doesn't exist for this user.
    """
    start = time.monotonic()

    # Resolve the user-scoped ChromaDB collection name.
    user_col = get_user_collection(db, current_user.id, request.collection_name)
    if not user_col:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{request.collection_name}' not found",
        )
    chroma_collection = user_col.chroma_name

    logger.info(
        "POST /query: user=%s query='%s' collection=%s top_k=%d",
        current_user.id, request.query[:100], request.collection_name, request.top_k,
    )

    retrieval_result = retriever.retrieve(
        query=request.query,
        collection_name=chroma_collection,
        top_k=request.top_k,
        filter_source=request.filter_source,
    )

    response = generator.generate(
        query=request.query,
        search_results=retrieval_result.chunks,
    )

    duration = time.monotonic() - start
    logger.info(
        "POST /query complete: user=%s citation_count=%d duration_s=%.3f",
        current_user.id, len(response.citations), duration,
    )

    return response
