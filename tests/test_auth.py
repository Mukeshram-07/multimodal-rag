"""
Backend authentication tests.

Uses an in-memory SQLite database so no external PostgreSQL is needed.
Tests cover signup, login, JWT verification, protected routes, and
duplicate-email prevention.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

# Set the test database URL BEFORE importing anything that touches the engine.
os.environ["DATABASE_URL"] = "sqlite://"

from rag.api.main import app  # noqa: E402
from rag.auth.database import Base, get_db  # noqa: E402
from rag.auth.service import create_access_token, decode_access_token  # noqa: E402


# ---------------------------------------------------------------------------
# In-memory SQLite test database
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def setup_db():
    """
    Before each test: create a shared in-memory SQLite database, create all
    tables on it, and override the FastAPI get_db dependency to use it.

    Uses StaticPool so all connections share the same in-memory database
    (SQLite creates a new DB per connection by default with sqlite://).
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # all connections share the same in-memory DB
    )
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def _override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db
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

_SIGNUP_PAYLOAD = {
    "email": "alice@example.com",
    "password": "securepass123",
    "display_name": "Alice",
}


def _signup(client: TestClient, payload: dict | None = None) -> dict:
    return client.post("/auth/signup", json=payload or _SIGNUP_PAYLOAD).json()


def _login(client: TestClient, email: str = "alice@example.com", password: str = "securepass123") -> dict:
    return client.post("/auth/login", json={"email": email, "password": password}).json()


def _auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Signup tests
# ---------------------------------------------------------------------------


class TestSignup:
    def test_signup_returns_201(self, client: TestClient) -> None:
        resp = client.post("/auth/signup", json=_SIGNUP_PAYLOAD)
        assert resp.status_code == 201

    def test_signup_returns_access_token(self, client: TestClient) -> None:
        data = _signup(client)
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_signup_token_is_valid_jwt(self, client: TestClient) -> None:
        data = _signup(client)
        user_id = decode_access_token(data["access_token"])
        assert user_id is not None

    def test_duplicate_email_returns_409(self, client: TestClient) -> None:
        _signup(client)
        resp = client.post("/auth/signup", json=_SIGNUP_PAYLOAD)
        assert resp.status_code == 409

    def test_short_password_returns_422(self, client: TestClient) -> None:
        payload = {**_SIGNUP_PAYLOAD, "password": "short"}
        resp = client.post("/auth/signup", json=payload)
        assert resp.status_code == 422

    def test_invalid_email_returns_422(self, client: TestClient) -> None:
        payload = {**_SIGNUP_PAYLOAD, "email": "not-an-email"}
        resp = client.post("/auth/signup", json=payload)
        assert resp.status_code == 422

    def test_missing_email_returns_422(self, client: TestClient) -> None:
        resp = client.post("/auth/signup", json={"password": "securepass123"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Login tests
# ---------------------------------------------------------------------------


class TestLogin:
    def test_login_returns_200(self, client: TestClient) -> None:
        _signup(client)
        resp = client.post("/auth/login", json={"email": "alice@example.com", "password": "securepass123"})
        assert resp.status_code == 200

    def test_login_returns_access_token(self, client: TestClient) -> None:
        _signup(client)
        data = _login(client)
        assert "access_token" in data

    def test_wrong_password_returns_401(self, client: TestClient) -> None:
        _signup(client)
        resp = client.post("/auth/login", json={"email": "alice@example.com", "password": "wrongpass"})
        assert resp.status_code == 401

    def test_unknown_email_returns_401(self, client: TestClient) -> None:
        resp = client.post("/auth/login", json={"email": "nobody@example.com", "password": "pass"})
        assert resp.status_code == 401

    def test_email_case_insensitive(self, client: TestClient) -> None:
        _signup(client)
        resp = client.post("/auth/login", json={"email": "ALICE@EXAMPLE.COM", "password": "securepass123"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /auth/me tests
# ---------------------------------------------------------------------------


class TestMe:
    def test_me_returns_user_profile(self, client: TestClient) -> None:
        token = _signup(client)["access_token"]
        resp = client.get("/auth/me", headers=_auth_header(token))
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "alice@example.com"
        assert data["display_name"] == "Alice"

    def test_me_without_token_returns_403(self, client: TestClient) -> None:
        resp = client.get("/auth/me")
        assert resp.status_code in (401, 403)

    def test_me_with_invalid_token_returns_403(self, client: TestClient) -> None:
        resp = client.get("/auth/me", headers={"Authorization": "Bearer invalid.token.here"})
        assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# JWT verification tests
# ---------------------------------------------------------------------------


class TestJWT:
    def test_decode_valid_token_returns_user_id(self, client: TestClient) -> None:
        token = _signup(client)["access_token"]
        user_id = decode_access_token(token)
        assert user_id is not None
        assert len(user_id) > 0

    def test_decode_garbage_returns_none(self) -> None:
        assert decode_access_token("garbage") is None

    def test_decode_tampered_token_returns_none(self, client: TestClient) -> None:
        token = _signup(client)["access_token"]
        tampered = token[:-5] + "XXXXX"
        assert decode_access_token(tampered) is None

    def test_create_and_decode_roundtrip(self) -> None:
        token = create_access_token("test-user-id-123")
        assert decode_access_token(token) == "test-user-id-123"


# ---------------------------------------------------------------------------
# Protected route tests
# ---------------------------------------------------------------------------


class TestProtectedRoutes:
    def test_collections_requires_auth(self, client: TestClient) -> None:
        resp = client.get("/collections")
        assert resp.status_code in (401, 403)

    def test_ingest_requires_auth(self, client: TestClient) -> None:
        resp = client.post("/ingest", files={"file": ("f.pdf", b"%PDF", "application/pdf")})
        assert resp.status_code in (401, 403)

    def test_query_requires_auth(self, client: TestClient) -> None:
        resp = client.post("/query", json={"query": "test", "collection_name": "default"})
        assert resp.status_code in (401, 403)

    def test_health_is_public(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_authenticated_collections_returns_empty_list(self, client: TestClient) -> None:
        token = _signup(client)["access_token"]
        resp = client.get("/collections", headers=_auth_header(token))
        assert resp.status_code == 200
        assert resp.json()["collections"] == []
