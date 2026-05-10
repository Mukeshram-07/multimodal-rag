"""
SQLAlchemy ORM models for authentication and user data.

Tables:
  users            — registered accounts (passwordless via OTP)
  user_collections — per-user ChromaDB collection ownership
  otp_codes        — one-time passwords for email verification
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag.auth.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    # nullable — OTP-only users have no password
    hashed_password: Mapped[str | None] = mapped_column(String(255), nullable=True)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    # Google OAuth fields
    google_id: Mapped[str | None] = mapped_column(String(255), nullable=True, unique=True, index=True)
    avatar_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    auth_provider: Mapped[str] = mapped_column(String(20), nullable=False, default="otp")
    # Supported values: "otp", "google"

    collections: Mapped[list["UserCollection"]] = relationship(
        "UserCollection", back_populates="owner", cascade="all, delete-orphan"
    )
    otp_codes: Mapped[list["OtpCode"]] = relationship(
        "OtpCode", back_populates="user", cascade="all, delete-orphan"
    )


class UserCollection(Base):
    """Maps a user to a ChromaDB collection they own."""

    __tablename__ = "user_collections"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    chroma_name: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    owner: Mapped["User"] = relationship("User", back_populates="collections")


class OtpCode(Base):
    """
    One-time password record.

    The plaintext OTP is never stored — only its Argon2 hash.
    Each record is single-use and expires after otp_expire_minutes.
    """

    __tablename__ = "otp_codes"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    otp_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    user: Mapped["User"] = relationship("User", back_populates="otp_codes")
