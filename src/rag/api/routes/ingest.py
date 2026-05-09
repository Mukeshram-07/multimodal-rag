"""
POST /ingest endpoint — PDF ingestion route.

Requirements: 6.1, 1.1
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile

from rag.api.dependencies import get_ingestion_pipeline
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
) -> IngestResponse:
    """
    Ingest a PDF document into the vector store.

    Saves the uploaded file to a temporary path, runs the full ingestion
    pipeline (parse → chunk → embed → store), and returns the result.
    """
    start = time.monotonic()
    logger.info(
        "POST /ingest: filename=%s collection=%s",
        file.filename,
        collection_name,
    )

    content = file.file.read()

    with tempfile.NamedTemporaryFile(
        suffix=Path(file.filename or "upload.pdf").suffix,
        delete=False,
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = pipeline.ingest(tmp_path, collection_name)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    duration = time.monotonic() - start
    logger.info(
        "POST /ingest complete: filename=%s chunk_count=%d duration_s=%.3f",
        file.filename,
        result.chunk_count,
        duration,
    )

    return IngestResponse(
        status=result.status,
        chunk_count=result.chunk_count,
        collection_name=result.collection_name,
    )
