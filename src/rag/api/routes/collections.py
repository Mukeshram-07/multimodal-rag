"""
GET /collections and DELETE /collections/{name} endpoints (authenticated).

Collections are scoped per-user: each user only sees and can delete
their own collections.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from rag.api.dependencies import get_vector_store
from rag.auth.database import get_db
from rag.auth.deps import get_current_user
from rag.auth.models import User
from rag.auth.service import delete_user_collection, list_user_collections
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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CollectionsResponse:
    """Return the names of all collections owned by the current user."""
    cols = list_user_collections(db, current_user.id)
    names = [c.name for c in cols]
    logger.info("GET /collections: user=%s count=%d", current_user.id, len(names))
    return CollectionsResponse(collections=names)


@router.delete("/collections/{collection_name}", response_model=DeleteResponse)
def delete_collection(
    collection_name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    store: VectorStore = Depends(get_vector_store),
) -> DeleteResponse:
    """Delete one of the current user's collections from both the DB and ChromaDB."""
    from rag.auth.service import get_user_collection

    user_col = get_user_collection(db, current_user.id, collection_name)
    if not user_col:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_name}' not found",
        )

    # Delete from ChromaDB (ignore if already gone)
    try:
        store.delete_collection(user_col.chroma_name)
    except Exception:
        pass

    # Delete from the ownership table
    delete_user_collection(db, current_user.id, collection_name)

    logger.info(
        "DELETE /collections/%s: user=%s chroma=%s",
        collection_name, current_user.id, user_col.chroma_name,
    )
    return DeleteResponse(status="deleted", collection_name=collection_name)
