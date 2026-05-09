"""
Structured logging configuration for the Multimodal RAG System.

Sets up a consistent log format across all modules. The log level is
controlled by the `log_level` field in Settings (default: INFO).

Requirements: 11.1
"""

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.config import Settings


def configure_logging(settings: "Settings | None" = None, log_level: str | None = None) -> None:
    """
    Configure the root logger with a structured format.

    Call this once at application startup (e.g., in the FastAPI lifespan or
    the Streamlit app entry point).

    Args:
        settings:  A Settings instance. If provided, its `log_level` is used.
        log_level: An explicit log level string (e.g. "DEBUG"). Takes precedence
                   over `settings.log_level` when both are supplied.
    """
    effective_level_str: str = "INFO"

    if log_level is not None:
        effective_level_str = log_level
    elif settings is not None:
        effective_level_str = settings.log_level

    numeric_level = getattr(logging, effective_level_str.upper(), logging.INFO)

    # Build a structured formatter that includes timestamp, level, logger name,
    # and message — easy to parse with log aggregation tools.
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Remove any existing handlers to avoid duplicate log lines when called
    # multiple times (e.g., during testing).
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    # Quieten noisy third-party loggers that are not relevant to the application.
    for noisy_logger in ("httpx", "httpcore", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger for use within a module.

    Usage::

        from rag.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Processing document: %s", filename)
    """
    return logging.getLogger(name)
