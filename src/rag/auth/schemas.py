"""Pydantic schemas for auth request/response bodies."""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field, field_validator


# ── OTP flow ──────────────────────────────────────────────────────────────────

class RequestOtpRequest(BaseModel):
    email: EmailStr


class RequestOtpResponse(BaseModel):
    message: str
    email: str


class VerifyOtpRequest(BaseModel):
    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6, pattern=r"^\d{6}$")


class OtpTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserResponse"


# ── Legacy password flow (kept for backward compat) ───────────────────────────

class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    display_name: str = Field(default="", max_length=100)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── Google OAuth ──────────────────────────────────────────────────────────────

class GoogleLoginRequest(BaseModel):
    credential: str  # Google ID token from the frontend


# ── Profile ───────────────────────────────────────────────────────────────────

class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str
    is_active: bool
    created_at: str = ""
    auth_provider: str = "otp"
    avatar_url: str | None = None

    model_config = {"from_attributes": True}


class UpdateProfileRequest(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=100)
