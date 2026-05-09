"""
GET /collections and DELETE /collections/{name} endpoints.

Requirements: 6.3, 6.4
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from rag.api.dependencies import get_vector_store
from rag.logging_config import get_logger
from rag.storage.vector_store import VectorStore

router = APIRouter()
logger = get_logger(__name__)


class CollectionsResponse(BaseModel):
    collections: list[str]


class DeleteResponse(BaseModel):
    status: str
    collection_name: str


@router.get("/collections", response_model=CollectionsResponse)
def list_collections(
    store: VectorStore = Depends(get_vector_store),
) -> CollectionsResponse:
    """Return the names of all collections in the vector store."""
    names = store.list_collections()
    logger.info("GET /collections: count=%d", len(names))
    return CollectionsResponse(collections=names)


@router.delete("/collections/{collection_name}", response_model=DeleteResponse)
def delete_collection(
    collection_name: str,
    store: VectorStore = Depends(get_vector_store),
) -> DeleteResponse:
    """Delete the named collection and all its stored chunks."""
    existing = store.list_collections()
    if collection_name not in existing:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' not found",
        )
    store.delete_collection(collection_name)
    logger.info("DELETE /collections/%s: deleted", collection_name)
    return DeleteResponse(status="deleted", collection_name=collection_name)
