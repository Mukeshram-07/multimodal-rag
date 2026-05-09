"""
Configuration management for the Multimodal RAG System.

All settings are read from environment variables with sensible defaults.
A .env file is supported for local development.

Requirements: 9.1, 9.2, 9.3, 9.4
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or a .env file.

    All fields have defaults so the system can run out-of-the-box for local
    development. Override any value via environment variable or .env file.
    """

    # Embedding configuration
    embedding_model: str = "all-MiniLM-L6-v2"

    # Chunking configuration
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval configuration
    top_k: int = 5

    # ChromaDB configuration
    chroma_persist_dir: str = "./chroma_db"

    # LLM backend configuration (OpenAI-compatible, defaults to Ollama)
    llm_api_base: str = "http://localhost:11434/v1"
    llm_model: str = "llama3"
    llm_api_key: str = "ollama"

    # Observability
    log_level: str = "INFO"

    # Application metadata
    app_version: str = "0.1.0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Allow extra fields to be ignored rather than raising validation errors
        extra="ignore",
    )


def get_settings() -> Settings:
    """Return a Settings instance. Suitable for use as a FastAPI dependency."""
    return Settings()
