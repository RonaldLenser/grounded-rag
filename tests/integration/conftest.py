"""Fixtures for integration tests (everything in this folder needs the database)."""

import os
import socket
from urllib.parse import urlparse

import pytest


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
