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

    # LLM backend configuration
    # Provider: "groq" (default) or "ollama" (local)
    llm_provider: str = "groq"
    llm_api_base: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_api_key: str = ""  # Required for Groq; set LLM_API_KEY in .env

    # Observability
    log_level: str = "INFO"

    # Application metadata
    app_version: str = "0.1.0"

    # Authentication
    database_url: str = "sqlite:///./rag_auth.db"
    jwt_secret_key: str = "change-me-in-production-use-a-long-random-string"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24  # 24 hours

    # Email (Resend API)
    resend_api_key: str = ""          # set RESEND_API_KEY in .env / docker-compose
    email_from: str = "noreply@example.com"  # set EMAIL_FROM in .env

    # OTP settings
    otp_expire_minutes: int = 5
    otp_max_requests_per_window: int = 5
    otp_rate_limit_window_minutes: int = 15

    # Google OAuth
    google_client_id: str = ""  # set GOOGLE_CLIENT_ID in .env

    # CORS / deployment
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    # Comma-separated list; add your Vercel URL in production

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Allow extra fields to be ignored rather than raising validation errors
        extra="ignore",
    )


def get_settings() -> Settings:
    """Return a Settings instance. Suitable for use as a FastAPI dependency."""
    return Settings()
