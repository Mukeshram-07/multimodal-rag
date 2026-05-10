"""
SQLAlchemy database setup for the Multimodal RAG System.

The engine and session factory are created lazily on first use so that
tests can override the DATABASE_URL environment variable before any
database connection is attempted.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


# Module-level singletons — populated on first call to _get_engine().
_engine = None
_SessionLocal = None


def _get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        from rag.config import get_settings
        settings = get_settings()
        url = settings.database_url
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine


def _get_session_factory():
    _get_engine()  # ensure initialised
    return _SessionLocal


def get_db():
    """FastAPI dependency that yields a database session."""
    factory = _get_session_factory()
    db: Session = factory()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """Create all tables if they don't exist. Called at application startup."""
    engine = _get_engine()
    Base.metadata.create_all(bind=engine)


def reset_engine() -> None:
    """
    Reset the cached engine and session factory.
    Used in tests to force re-initialisation with a different DATABASE_URL.
    """
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
