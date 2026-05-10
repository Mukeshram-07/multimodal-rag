"""
OTP authentication tests.

Tests cover:
  - OTP request (creates user, sends email)
  - OTP verification (issues JWT)
  - Expired OTP rejection
  - Invalid OTP rejection
  - Replay attack prevention (used OTP)
  - Rate limiting (429 after too many requests)
  - JWT issuance and protected route access
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

os.environ["DATABASE_URL"] = "sqlite://"

from rag.api.main import app  # noqa: E402
from rag.auth.database import Base, get_db  # noqa: E402
from rag.auth.models import OtpCode  # noqa: E402
from rag.auth.otp_service import create_otp_record, generate_otp, hash_otp  # noqa: E402
from rag.auth.service import create_access_token, decode_access_token, get_or_create_user  # noqa: E402


# ---------------------------------------------------------------------------
# Test DB fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def setup_db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def _override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override
    yield
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def _auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _request_otp(client: TestClient, email: str = "bob@example.com") -> dict:
    with patch("rag.api.routes.auth.send_otp_email", return_value=True):
        return client.post("/auth/request-otp", json={"email": email})


def _get_db_session():
    """Get a direct DB session for test setup."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    # Re-use the override registered in setup_db
    override = app.dependency_overrides.get(get_db)
    if override:
        gen = override()
        return next(gen)
    return None


# ---------------------------------------------------------------------------
# OTP request tests
# ---------------------------------------------------------------------------


class TestRequestOtp:
    def test_request_otp_returns_200(self, client: TestClient) -> None:
        resp = _request_otp(client)
        assert resp.status_code == 200

    def test_request_otp_returns_message(self, client: TestClient) -> None:
        data = _request_otp(client).json()
        assert "message" in data
        assert "email" in data

    def test_request_otp_creates_user_if_not_exists(self, client: TestClient) -> None:
        _request_otp(client, "newuser@example.com")
        # Verify user was created by requesting OTP again (rate limit not hit yet)
        resp = _request_otp(client, "newuser@example.com")
        assert resp.status_code == 200

    def test_request_otp_invalid_email_returns_422(self, client: TestClient) -> None:
        resp = client.post("/auth/request-otp", json={"email": "not-an-email"})
        assert resp.status_code == 422

    def test_request_otp_calls_email_service(self, client: TestClient) -> None:
        with patch("rag.api.routes.auth.send_otp_email", return_value=True) as mock_send:
            client.post("/auth/request-otp", json={"email": "test@example.com"})
            mock_send.assert_called_once()

    def test_request_otp_email_failure_returns_503(self, client: TestClient) -> None:
        with patch("rag.api.routes.auth.send_otp_email", return_value=False):
            resp = client.post("/auth/request-otp", json={"email": "fail@example.com"})
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# OTP verification tests
# ---------------------------------------------------------------------------


class TestVerifyOtp:
    def _setup_otp(self, client: TestClient, email: str = "bob@example.com") -> str:
        """Request OTP and return the plaintext OTP by intercepting the call."""
        captured = {}

        def capture_send(to_email, otp):
            captured["otp"] = otp
            return True

        with patch("rag.api.routes.auth.send_otp_email", side_effect=capture_send):
            client.post("/auth/request-otp", json={"email": email})

        return captured["otp"]

    def test_verify_valid_otp_returns_200(self, client: TestClient) -> None:
        otp = self._setup_otp(client)
        resp = client.post("/auth/verify-otp", json={"email": "bob@example.com", "otp": otp})
        assert resp.status_code == 200

    def test_verify_valid_otp_returns_access_token(self, client: TestClient) -> None:
        otp = self._setup_otp(client)
        data = client.post("/auth/verify-otp", json={"email": "bob@example.com", "otp": otp}).json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_verify_valid_otp_returns_user(self, client: TestClient) -> None:
        otp = self._setup_otp(client)
        data = client.post("/auth/verify-otp", json={"email": "bob@example.com", "otp": otp}).json()
        assert "user" in data
        assert data["user"]["email"] == "bob@example.com"

    def test_verify_token_is_valid_jwt(self, client: TestClient) -> None:
        otp = self._setup_otp(client)
        data = client.post("/auth/verify-otp", json={"email": "bob@example.com", "otp": otp}).json()
        user_id = decode_access_token(data["access_token"])
        assert user_id is not None

    def test_verify_invalid_otp_returns_401(self, client: TestClient) -> None:
        self._setup_otp(client)
        resp = client.post("/auth/verify-otp", json={"email": "bob@example.com", "otp": "000000"})
        assert resp.status_code == 401

    def test_verify_wrong_format_returns_422(self, client: TestClient) -> None:
        resp = client.post("/auth/verify-otp", json={"email": "bob@example.com", "otp": "abc"})
        assert resp.status_code == 422

    def test_verify_unknown_email_returns_401(self, client: TestClient) -> None:
        resp = client.post("/auth/verify-otp", json={"email": "ghost@example.com", "otp": "123456"})
        assert resp.status_code == 401

    def test_replay_attack_prevented(self, client: TestClient) -> None:
        """Using the same OTP twice must fail on the second attempt."""
        otp = self._setup_otp(client)
        client.post("/auth/verify-otp", json={"email": "bob@example.com", "otp": otp})
        resp = client.post("/auth/verify-otp", json={"email": "bob@example.com", "otp": otp})
        assert resp.status_code == 401

    def test_expired_otp_returns_401(self, client: TestClient) -> None:
        """An OTP with expires_at in the past must be rejected."""
        # Create user and an already-expired OTP record directly.
        override = app.dependency_overrides.get(get_db)
        db = next(override())
        user = get_or_create_user(db, "expired@example.com")
        otp = "999999"
        record = OtpCode(
            user_id=user.id,
            otp_hash=hash_otp(otp),
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        )
        db.add(record)
        db.commit()
        db.close()

        resp = client.post("/auth/verify-otp", json={"email": "expired@example.com", "otp": otp})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def test_rate_limit_after_max_requests(self, client: TestClient) -> None:
        """After 5 OTP requests in the window, the 6th must return 429."""
        email = "ratelimit@example.com"
        with patch("rag.api.routes.auth.send_otp_email", return_value=True):
            for _ in range(5):
                client.post("/auth/request-otp", json={"email": email})
            resp = client.post("/auth/request-otp", json={"email": email})
        assert resp.status_code == 429


# ---------------------------------------------------------------------------
# JWT + protected route tests
# ---------------------------------------------------------------------------


class TestOtpJwtProtectedRoutes:
    def _get_token(self, client: TestClient, email: str = "jwt@example.com") -> str:
        captured = {}

        def capture(to_email, otp):
            captured["otp"] = otp
            return True

        with patch("rag.api.routes.auth.send_otp_email", side_effect=capture):
            client.post("/auth/request-otp", json={"email": email})

        data = client.post("/auth/verify-otp", json={"email": email, "otp": captured["otp"]}).json()
        return data["access_token"]

    def test_jwt_allows_access_to_collections(self, client: TestClient) -> None:
        token = self._get_token(client)
        resp = client.get("/collections", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200

    def test_jwt_allows_access_to_me(self, client: TestClient) -> None:
        token = self._get_token(client)
        resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["email"] == "jwt@example.com"
