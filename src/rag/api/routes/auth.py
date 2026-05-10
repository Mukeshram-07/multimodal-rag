"""
Authentication endpoints.

OTP flow (primary):
  POST /auth/request-otp  — send a 6-digit OTP to the user's email
  POST /auth/verify-otp   — verify OTP and receive a JWT

Google OAuth:
  POST /auth/google       — verify Google ID token and receive a JWT

Legacy password flow (kept for backward compat / tests):
  POST /auth/signup
  POST /auth/login

Profile:
  GET  /auth/me
  PUT  /auth/me
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from rag.auth.database import get_db
from rag.auth.deps import get_current_user
from rag.auth.email_service import send_otp_email
from rag.auth.google_service import verify_google_token
from rag.auth.models import User
from rag.auth.otp_service import (
    check_rate_limit,
    create_otp_record,
    generate_otp,
    verify_and_consume_otp,
)
from rag.auth.schemas import (
    GoogleLoginRequest,
    LoginRequest,
    OtpTokenResponse,
    RequestOtpRequest,
    RequestOtpResponse,
    SignupRequest,
    TokenResponse,
    UpdateProfileRequest,
    UserResponse,
    VerifyOtpRequest,
)
from rag.auth.service import (
    authenticate_user,
    create_access_token,
    create_user,
    get_or_create_google_user,
    get_or_create_user,
    get_user_by_email,
    update_display_name,
)
from rag.logging_config import get_logger

router = APIRouter(prefix="/auth", tags=["auth"])
logger = get_logger(__name__)


def _user_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        is_active=user.is_active,
        created_at=user.created_at.isoformat() if user.created_at else "",
        auth_provider=user.auth_provider,
        avatar_url=user.avatar_url,
    )


# ---------------------------------------------------------------------------
# OTP flow
# ---------------------------------------------------------------------------


@router.post("/request-otp", response_model=RequestOtpResponse)
def request_otp(body: RequestOtpRequest, db: Session = Depends(get_db)) -> RequestOtpResponse:
    user = get_or_create_user(db, body.email)

    if not check_rate_limit(db, user):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many OTP requests. Please wait before requesting another code.",
        )

    otp = generate_otp()
    create_otp_record(db, user, otp)

    sent = send_otp_email(str(body.email), otp)
    if not sent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to send verification email. Please try again.",
        )

    logger.info("OTP requested: user_id=%s email=%s", user.id, user.email)
    return RequestOtpResponse(message="Verification code sent to your email.", email=str(body.email))


@router.post("/verify-otp", response_model=OtpTokenResponse)
def verify_otp(body: VerifyOtpRequest, db: Session = Depends(get_db)) -> OtpTokenResponse:
    user = get_user_by_email(db, str(body.email))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or expired verification code.")

    if not verify_and_consume_otp(db, user, body.otp):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or expired verification code.")

    token = create_access_token(user.id)
    logger.info("OTP verified, JWT issued: user_id=%s", user.id)
    return OtpTokenResponse(access_token=token, user=_user_response(user))


# ---------------------------------------------------------------------------
# Google OAuth
# ---------------------------------------------------------------------------


@router.post("/google", response_model=OtpTokenResponse)
def google_login(body: GoogleLoginRequest, db: Session = Depends(get_db)) -> OtpTokenResponse:
    """
    Verify a Google ID token (issued by the frontend via Google Sign-In)
    and return a JWT access token.

    The backend validates the token server-side — the frontend email/name
    data is never trusted directly.
    """
    user_info = verify_google_token(body.credential)
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google credential. Please try again.",
        )

    user = get_or_create_google_user(
        db,
        google_id=user_info.google_id,
        email=user_info.email,
        display_name=user_info.display_name,
        avatar_url=user_info.avatar_url,
    )

    token = create_access_token(user.id)
    logger.info("Google login: user_id=%s email=%s", user.id, user.email)
    return OtpTokenResponse(access_token=token, user=_user_response(user))


# ---------------------------------------------------------------------------
# Legacy password flow (backward compat)
# ---------------------------------------------------------------------------


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def signup(body: SignupRequest, db: Session = Depends(get_db)) -> TokenResponse:
    if get_user_by_email(db, body.email):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="An account with this email already exists")
    user = create_user(db, email=body.email, password=body.password, display_name=body.display_name)
    token = create_access_token(user.id)
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = authenticate_user(db, body.email, body.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect email or password",
                            headers={"WWW-Authenticate": "Bearer"})
    token = create_access_token(user.id)
    return TokenResponse(access_token=token)


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    return _user_response(current_user)


@router.put("/me", response_model=UserResponse)
def update_me(
    body: UpdateProfileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserResponse:
    user = update_display_name(db, current_user, body.display_name)
    return _user_response(user)
