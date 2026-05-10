"""
FastAPI dependencies for authentication.

Provides ``get_current_user`` which extracts and validates the Bearer JWT
from the Authorization header and returns the authenticated User ORM object.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from rag.auth.database import get_db
from rag.auth.models import User
from rag.auth.service import decode_access_token, get_user_by_id

_bearer = HTTPBearer(auto_error=True)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    """
    Extract the JWT from the Authorization header, verify it, and return
    the corresponding User.  Raises HTTP 401 if the token is missing,
    invalid, or expired.
    """
    token = credentials.credentials
    user_id = decode_access_token(token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = get_user_by_id(db, user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
