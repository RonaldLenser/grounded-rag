"""Fixtures for integration tests (everything in this folder needs the database)."""

import asyncio
import os
import socket
from collections.abc import Iterator
from urllib.parse import urlparse

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from grounded.api.deps import get_embedder
from grounded.config import Settings
from grounded.main import create_app
from tests.fakes import FakeEmbedder

ADMIN_KEY = "test-admin-key-0123456789"


def _database_reachable(database_url: str) -> bool:
    parsed = urlparse(database_url)
    try:
        with socket.create_connection((parsed.hostname or "localhost", parsed.port or 5432), 1):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session", autouse=True)
def require_database() -> None:
    """Runs once, before any other fixture in this folder (session scope goes first).

    Skips the folder locally when Postgres isn't running; fails it in CI
    (REQUIRE_DB=1) so a broken database can never pass as "all green".
    """
    url = os.environ["DATABASE_URL"]
    if _database_reachable(url):
        return
    parsed = urlparse(url)
    message = f"PostgreSQL not reachable at {parsed.hostname}:{parsed.port}"
    if os.environ.get("REQUIRE_DB") == "1":
        pytest.fail(message)
    pytest.skip(f"{message} (start it with: docker compose up -d db)")


@pytest.fixture(scope="session", autouse=True)
def migrated(require_database: None) -> None:
    """Rebuild the schema from zero once per test run: downgrade to nothing, upgrade to latest.

    This doubles as a round-trip test: both directions of every migration run.
    """
    config = Config("alembic.ini")
    command.downgrade(config, "base")
    command.upgrade(config, "head")


@pytest.fixture(autouse=True)
def clean_tables(migrated: None) -> None:
    """Every test starts with empty tables, so tests can't affect each other."""

    async def truncate() -> None:
        engine = create_async_engine(os.environ["DATABASE_URL"])
        async with engine.begin() as conn:
            await conn.execute(text("TRUNCATE collections, documents, chunks, query_logs CASCADE"))
        await engine.dispose()

    asyncio.run(truncate())


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def api(fake_embedder: FakeEmbedder) -> Iterator[TestClient]:
    """A client for the full app: real database, fake embeddings, admin key configured."""
    settings = Settings(
        app_env="test",
        database_url=os.environ["DATABASE_URL"],
        admin_api_key=SecretStr(ADMIN_KEY),
        max_upload_mb=1,
    )
    app = create_app(settings)
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    with TestClient(app, headers={"X-API-Key": ADMIN_KEY}) as client:
        yield client
