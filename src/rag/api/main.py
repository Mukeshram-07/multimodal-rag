"""
FastAPI application entry point for the Multimodal RAG System.

Wires together all routes, registers global exception handlers, and adds
request logging middleware.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag.api.routes import auth, collections, health, ingest, query
from rag.exceptions import (
    EmbeddingError,
    GenerationError,
    IngestionError,
    RetrievalError,
)
from rag.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables on startup (idempotent)."""
    from rag.auth.database import create_tables
    create_tables()
    logger.info("Database tables ready.")
    yield


def _get_allowed_origins() -> list[str]:
    """Parse the comma-separated ALLOWED_ORIGINS setting."""
    from rag.config import get_settings
    raw = get_settings().allowed_origins
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(
    title="Multimodal RAG System",
    description="PDF ingestion, semantic retrieval, and citation-aware answer generation.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins() or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(auth.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(collections.router)
app.include_router(health.router)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration = time.monotonic() - start
    logger.info(
        "HTTP %s %s → %d (%.3fs)",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(IngestionError)
async def ingestion_error_handler(request: Request, exc: IngestionError) -> JSONResponse:
    logger.warning("IngestionError: %s", exc.message)
    return JSONResponse(status_code=422, content={"error": "IngestionError", "detail": exc.message})


@app.exception_handler(EmbeddingError)
async def embedding_error_handler(request: Request, exc: EmbeddingError) -> JSONResponse:
    logger.error("EmbeddingError: %s", exc.message)
    return JSONResponse(status_code=500, content={"error": "EmbeddingError", "detail": exc.message})


@app.exception_handler(RetrievalError)
async def retrieval_error_handler(request: Request, exc: RetrievalError) -> JSONResponse:
    logger.error("RetrievalError: %s", exc.message)
    return JSONResponse(status_code=500, content={"error": "RetrievalError", "detail": exc.message})


@app.exception_handler(GenerationError)
async def generation_error_handler(request: Request, exc: GenerationError) -> JSONResponse:
    logger.error("GenerationError: %s", exc.message)
    return JSONResponse(status_code=500, content={"error": "GenerationError", "detail": exc.message})
