"""
Authentication service: JWT creation/verification and user CRUD.

Password-based auth helpers are kept for backward compatibility with
existing tests but are no longer used by the primary OTP flow.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from rag.auth.models import User, UserCollection
from rag.config import get_settings
from rag.logging_config import get_logger

logger = get_logger(__name__)

_pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


# ---------------------------------------------------------------------------
# Password helpers (kept for backward compat / optional password auth)
# ---------------------------------------------------------------------------


def hash_password(plain: str) -> str:
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def create_access_token(user_id: str) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> str | None:
    """Return the user_id from a valid token, or None if invalid/expired."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        return payload.get("sub")
    except JWTError:
        return None


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email.lower()).first()


def get_user_by_id(db: Session, user_id: str) -> User | None:
    return db.query(User).filter(User.id == user_id).first()


def get_or_create_user(db: Session, email: str, display_name: str = "") -> User:
    """Return existing user or create a new passwordless one."""
    user = get_user_by_email(db, email)
    if user:
        return user
    user = User(
        email=email.lower(),
        hashed_password=None,
        display_name=display_name or email.split("@")[0],
        auth_provider="otp",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("Created user (OTP): id=%s email=%s", user.id, user.email)
    return user


def get_or_create_google_user(
    db: Session,
    google_id: str,
    email: str,
    display_name: str = "",
    avatar_url: str | None = None,
) -> User:
    """
    Find or create a user from a verified Google identity.

    If a user with the same email already exists (e.g. from OTP login),
    link the Google ID to that account and update the avatar.
    """
    # Try by google_id first (fastest path for returning Google users)
    user = db.query(User).filter(User.google_id == google_id).first()
    if user:
        # Refresh avatar in case it changed
        if avatar_url and user.avatar_url != avatar_url:
            user.avatar_url = avatar_url
            db.commit()
            db.refresh(user)
        return user

    # Try by email (account may exist from OTP login)
    user = get_user_by_email(db, email)
    if user:
        # Link Google to existing account
        user.google_id = google_id
        user.auth_provider = "google"
        if avatar_url:
            user.avatar_url = avatar_url
        if not user.display_name and display_name:
            user.display_name = display_name
        db.commit()
        db.refresh(user)
        logger.info("Linked Google to existing user: id=%s email=%s", user.id, user.email)
        return user

    # New user — create from Google identity
    user = User(
        email=email.lower(),
        hashed_password=None,
        display_name=display_name or email.split("@")[0],
        google_id=google_id,
        avatar_url=avatar_url,
        auth_provider="google",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("Created user (Google): id=%s email=%s", user.id, user.email)
    return user


def create_user(db: Session, email: str, password: str, display_name: str = "") -> User:
    """Create a user with a hashed password (legacy / optional)."""
    user = User(
        email=email.lower(),
        hashed_password=hash_password(password),
        display_name=display_name or email.split("@")[0],
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("Created user: id=%s email=%s", user.id, user.email)
    return user


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    user = get_user_by_email(db, email)
    if not user or not user.hashed_password:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def update_display_name(db: Session, user: User, display_name: str) -> User:
    user.display_name = display_name
    db.commit()
    db.refresh(user)
    return user


# ---------------------------------------------------------------------------
# Collection ownership
# ---------------------------------------------------------------------------


def _chroma_name(user_id: str, collection_name: str) -> str:
    safe = collection_name.replace(" ", "_").replace("/", "_")
    return f"u{user_id[:8]}__{safe}"


def get_or_create_collection(db: Session, user_id: str, collection_name: str) -> UserCollection:
    chroma = _chroma_name(user_id, collection_name)
    col = (
        db.query(UserCollection)
        .filter(UserCollection.user_id == user_id, UserCollection.name == collection_name)
        .first()
    )
    if col:
        return col
    col = UserCollection(user_id=user_id, name=collection_name, chroma_name=chroma)
    db.add(col)
    db.commit()
    db.refresh(col)
    return col


def list_user_collections(db: Session, user_id: str) -> list[UserCollection]:
    return (
        db.query(UserCollection)
        .filter(UserCollection.user_id == user_id)
        .order_by(UserCollection.created_at)
        .all()
    )


def get_user_collection(db: Session, user_id: str, collection_name: str) -> UserCollection | None:
    return (
        db.query(UserCollection)
        .filter(UserCollection.user_id == user_id, UserCollection.name == collection_name)
        .first()
    )


def delete_user_collection(db: Session, user_id: str, collection_name: str) -> bool:
    col = get_user_collection(db, user_id, collection_name)
    if not col:
        return False
    db.delete(col)
    db.commit()
    return True
