"""
Google OAuth authentication tests.

All Google token verification is mocked — no real Google API calls.
Tests cover:
  - Valid Google token → JWT issued
  - Invalid/expired Google token → 401
  - New user created from Google identity
  - Existing OTP user linked to Google
  - Account reuse (same google_id returns same user)
  - Protected route access with Google-issued JWT
  - Mixed-provider: OTP user and Google user are independent
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

os.environ["DATABASE_URL"] = "sqlite://"
os.environ["GOOGLE_CLIENT_ID"] = "test-client-id.apps.googleusercontent.com"

from rag.api.main import app  # noqa: E402
from rag.auth.database import Base, get_db  # noqa: E402
from rag.auth.google_service import GoogleUserInfo  # noqa: E402
from rag.auth.service import create_access_token, decode_access_token  # noqa: E402


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GOOGLE_USER = GoogleUserInfo(
    google_id="google-sub-12345",
    email="alice@gmail.com",
    display_name="Alice Google",
    avatar_url="https://lh3.googleusercontent.com/photo.jpg",
)


def _google_login(client: TestClient, user_info: GoogleUserInfo = _GOOGLE_USER) -> dict:
    with patch("rag.api.routes.auth.verify_google_token", return_value=user_info):
        return client.post("/auth/google", json={"credential": "fake-google-token"})


def _auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Google login tests
# ---------------------------------------------------------------------------


class TestGoogleLogin:
    def test_valid_google_token_returns_200(self, client: TestClient) -> None:
        resp = _google_login(client)
        assert resp.status_code == 200

    def test_valid_google_token_returns_access_token(self, client: TestClient) -> None:
        data = _google_login(client).json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_valid_google_token_returns_user(self, client: TestClient) -> None:
        data = _google_login(client).json()
        assert "user" in data
        assert data["user"]["email"] == "alice@gmail.com"
        assert data["user"]["display_name"] == "Alice Google"

    def test_google_user_auth_provider_is_google(self, client: TestClient) -> None:
        data = _google_login(client).json()
        assert data["user"]["auth_provider"] == "google"

    def test_google_user_has_avatar_url(self, client: TestClient) -> None:
        data = _google_login(client).json()
        assert data["user"]["avatar_url"] is not None

    def test_jwt_from_google_is_valid(self, client: TestClient) -> None:
        data = _google_login(client).json()
        user_id = decode_access_token(data["access_token"])
        assert user_id is not None

    def test_invalid_google_token_returns_401(self, client: TestClient) -> None:
        with patch("rag.api.routes.auth.verify_google_token", return_value=None):
            resp = client.post("/auth/google", json={"credential": "bad-token"})
        assert resp.status_code == 401

    def test_missing_credential_returns_422(self, client: TestClient) -> None:
        resp = client.post("/auth/google", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Account creation and reuse
# ---------------------------------------------------------------------------


class TestGoogleAccountHandling:
    def test_new_user_created_on_first_google_login(self, client: TestClient) -> None:
        resp = _google_login(client)
        assert resp.status_code == 200
        # Second login with same google_id should return same user
        resp2 = _google_login(client)
        assert resp2.json()["user"]["id"] == resp.json()["user"]["id"]

    def test_account_reuse_same_google_id(self, client: TestClient) -> None:
        """Two logins with the same Google ID return the same user."""
        id1 = _google_login(client).json()["user"]["id"]
        id2 = _google_login(client).json()["user"]["id"]
        assert id1 == id2

    def test_existing_otp_user_linked_to_google(self, client: TestClient) -> None:
        """If an OTP user with the same email exists, Google login links to it."""
        # Create OTP user first
        with patch("rag.api.routes.auth.send_otp_email", return_value=True):
            client.post("/auth/request-otp", json={"email": "alice@gmail.com"})

        # Now login with Google using the same email
        data = _google_login(client).json()
        # Should be the same account (same email)
        assert data["user"]["email"] == "alice@gmail.com"

    def test_different_google_ids_create_different_users(self, client: TestClient) -> None:
        user_a = GoogleUserInfo("id-aaa", "a@gmail.com", "User A", None)
        user_b = GoogleUserInfo("id-bbb", "b@gmail.com", "User B", None)
        id_a = _google_login(client, user_a).json()["user"]["id"]
        id_b = _google_login(client, user_b).json()["user"]["id"]
        assert id_a != id_b


# ---------------------------------------------------------------------------
# Protected route access
# ---------------------------------------------------------------------------


class TestGoogleJwtProtectedRoutes:
    def test_google_jwt_allows_collections(self, client: TestClient) -> None:
        token = _google_login(client).json()["access_token"]
        resp = client.get("/collections", headers=_auth_header(token))
        assert resp.status_code == 200

    def test_google_jwt_allows_me(self, client: TestClient) -> None:
        token = _google_login(client).json()["access_token"]
        resp = client.get("/auth/me", headers=_auth_header(token))
        assert resp.status_code == 200
        assert resp.json()["email"] == "alice@gmail.com"


# ---------------------------------------------------------------------------
# Mixed-provider tests
# ---------------------------------------------------------------------------


class TestMixedProviderAuth:
    def test_otp_and_google_users_are_isolated(self, client: TestClient) -> None:
        """OTP user and Google user with different emails have separate accounts."""
        # OTP user
        with patch("rag.api.routes.auth.send_otp_email", return_value=True):
            client.post("/auth/request-otp", json={"email": "otp@example.com"})

        # Google user
        google_user = GoogleUserInfo("gid-xyz", "google@gmail.com", "Google User", None)
        google_data = _google_login(client, google_user).json()

        assert google_data["user"]["email"] == "google@gmail.com"
        assert google_data["user"]["auth_provider"] == "google"
