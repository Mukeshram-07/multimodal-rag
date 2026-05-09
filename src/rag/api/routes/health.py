"""
GET /health endpoint.

Requirements: 6.5
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from rag.api.dependencies import get_vector_store
from rag.config import get_settings
from rag.logging_config import get_logger
from rag.models import HealthResponse
from rag.storage.vector_store import VectorStore

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return operational status and application version."""
    settings = get_settings()
    logger.info("GET /health: status=ok version=%s", settings.app_version)
    return HealthResponse(status="ok", version=settings.app_version)
