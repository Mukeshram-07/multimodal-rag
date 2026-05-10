"""
POST /ingest endpoint — PDF ingestion route (authenticated).

Each user's collection is namespaced in ChromaDB so users cannot
access each other's data even if they choose the same collection name.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from rag.api.dependencies import get_ingestion_pipeline
from rag.auth.database import get_db
from rag.auth.deps import get_current_user
from rag.auth.models import User
from rag.auth.service import get_or_create_collection
from rag.ingestion.pipeline import IngestionPipeline
from rag.logging_config import get_logger
from rag.models import IngestResponse

router = APIRouter()
logger = get_logger(__name__)


@router.post("/ingest", response_model=IngestResponse)
def ingest_document(
    file: UploadFile = File(..., description="PDF file to ingest"),
    collection_name: str = Form(default="default", description="Target collection name"),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> IngestResponse:
    """
    Ingest a PDF document into the authenticated user's collection.

    The collection is namespaced per-user in ChromaDB so two users can
    both have a collection called "default" without conflict.
    """
    original_filename = file.filename or "upload.pdf"
    suffix = Path(original_filename).suffix or ".pdf"

    # Resolve (or create) the user-scoped ChromaDB collection name.
    user_col = get_or_create_collection(db, current_user.id, collection_name)
    chroma_collection = user_col.chroma_name

    start = time.monotonic()
    logger.info(
        "POST /ingest: user=%s filename=%s collection=%s chroma=%s",
        current_user.id, original_filename, collection_name, chroma_collection,
    )

    content = file.file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = pipeline.ingest(
            tmp_path,
            chroma_collection,
            source_filename=original_filename,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    duration = time.monotonic() - start
    logger.info(
        "POST /ingest complete: user=%s filename=%s chunk_count=%d duration_s=%.3f",
        current_user.id, original_filename, result.chunk_count, duration,
    )

    return IngestResponse(
        status=result.status,
        chunk_count=result.chunk_count,
        collection_name=collection_name,  # return the user-facing name, not the chroma name
    )
