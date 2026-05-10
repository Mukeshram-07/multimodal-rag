"""
OTP generation, hashing, verification, and rate-limiting.

Security properties:
  - OTPs are 6-digit codes generated with Python's secrets module
  - Only the Argon2 hash is stored — plaintext is never persisted
  - Each OTP expires after otp_expire_minutes (default 5)
  - Each OTP is single-use (used=True after successful verification)
  - Rate limiting: max 5 OTP requests per 15-minute window per email
  - Max 5 failed verification attempts per OTP record
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

from passlib.context import CryptContext
from sqlalchemy.orm import Session

from rag.auth.models import OtpCode, User
from rag.config import get_settings
from rag.logging_config import get_logger

logger = get_logger(__name__)

_pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# Maximum failed verification attempts before an OTP is invalidated.
_MAX_ATTEMPTS = 5


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# OTP generation
# ---------------------------------------------------------------------------


def generate_otp() -> str:
    """Return a cryptographically secure 6-digit OTP string."""
    return f"{secrets.randbelow(1_000_000):06d}"


def hash_otp(otp: str) -> str:
    return _pwd_context.hash(otp)


def verify_otp_hash(otp: str, hashed: str) -> bool:
    return _pwd_context.verify(otp, hashed)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def check_rate_limit(db: Session, user: User) -> bool:
    """
    Return True if the user is within the allowed OTP request rate.
    Return False if they have exceeded the limit (caller should return 429).
    """
    settings = get_settings()
    window_start = _utcnow() - timedelta(minutes=settings.otp_rate_limit_window_minutes)
    recent_count = (
        db.query(OtpCode)
        .filter(
            OtpCode.user_id == user.id,
            OtpCode.created_at >= window_start,
        )
        .count()
    )
    return recent_count < settings.otp_max_requests_per_window


# ---------------------------------------------------------------------------
# OTP CRUD
# ---------------------------------------------------------------------------


def create_otp_record(db: Session, user: User, otp: str) -> OtpCode:
    """Hash the OTP and persist a new OtpCode record."""
    settings = get_settings()
    expires_at = _utcnow() + timedelta(minutes=settings.otp_expire_minutes)
    record = OtpCode(
        user_id=user.id,
        otp_hash=hash_otp(otp),
        expires_at=expires_at,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.info("OTP created: user_id=%s expires_at=%s", user.id, expires_at)
    return record


def get_latest_unused_otp(db: Session, user: User) -> OtpCode | None:
    """Return the most recent unused, unexpired OTP for a user."""
    now = _utcnow()
    return (
        db.query(OtpCode)
        .filter(
            OtpCode.user_id == user.id,
            OtpCode.used == False,  # noqa: E712
            OtpCode.expires_at > now,
        )
        .order_by(OtpCode.created_at.desc())
        .first()
    )


def verify_and_consume_otp(db: Session, user: User, otp: str) -> bool:
    """
    Verify the OTP against the latest unused record.

    Returns True and marks the record as used on success.
    Returns False on invalid OTP, expiry, or too many attempts.
    """
    record = get_latest_unused_otp(db, user)
    if not record:
        logger.warning("OTP verify: no valid record for user_id=%s", user.id)
        return False

    # Increment attempt counter before verifying to prevent timing attacks.
    record.attempt_count += 1
    db.commit()

    if record.attempt_count > _MAX_ATTEMPTS:
        logger.warning(
            "OTP verify: too many attempts for user_id=%s record_id=%s",
            user.id, record.id,
        )
        return False

    if not verify_otp_hash(otp, record.otp_hash):
        logger.warning("OTP verify: invalid OTP for user_id=%s", user.id)
        return False

    # Mark as used — prevents replay attacks.
    record.used = True
    db.commit()
    logger.info("OTP verified: user_id=%s record_id=%s", user.id, record.id)
    return True
