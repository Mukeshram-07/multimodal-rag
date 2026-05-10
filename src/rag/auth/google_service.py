"""
Google OAuth server-side token verification.

The backend NEVER trusts frontend-provided user data.
It verifies the Google ID token signature, issuer, audience, and expiry
using Google's public keys via google-auth.
"""

from __future__ import annotations

from dataclasses import dataclass

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from rag.config import get_settings
from rag.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class GoogleUserInfo:
    google_id: str
    email: str
    display_name: str
    avatar_url: str | None


def verify_google_token(credential: str) -> GoogleUserInfo | None:
    """
    Verify a Google ID token and return the user's identity.

    Returns None if the token is invalid, expired, or has the wrong audience.
    Never raises — callers should treat None as an auth failure.
    """
    settings = get_settings()
    client_id = settings.google_client_id

    if not client_id:
        logger.error("GOOGLE_CLIENT_ID is not configured — cannot verify Google tokens")
        return None

    try:
        idinfo = id_token.verify_oauth2_token(
            credential,
            google_requests.Request(),
            client_id,
        )
    except Exception as exc:
        logger.warning("Google token verification failed: %s", exc)
        return None

    # Verify issuer
    if idinfo.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
        logger.warning("Google token has unexpected issuer: %s", idinfo.get("iss"))
        return None

    email = idinfo.get("email", "").lower()
    if not email:
        logger.warning("Google token missing email claim")
        return None

    return GoogleUserInfo(
        google_id=idinfo["sub"],
        email=email,
        display_name=idinfo.get("name", email.split("@")[0]),
        avatar_url=idinfo.get("picture"),
    )
