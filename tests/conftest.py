"""Shared test fixtures.

Unit tests never need a database. Integration tests (marked `integration`)
need PostgreSQL + pgvector at DATABASE_URL:
  - Locally: `docker compose up -d db`, then `uv run pytest`.
    If the database isn't running, integration tests are skipped.
  - In CI: REQUIRE_DB=1 is set, so a missing database FAILS the run
    instead of silently skipping it.
"""

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from grounded.config import Settings, get_settings
from grounded.main import create_app

LOCAL_TEST_DATABASE_URL = "postgresql+asyncpg://grounded:grounded@localhost:5433/grounded"

# Alembic's env.py reads settings from the environment, so make sure it's set.
os.environ.setdefault("DATABASE_URL", LOCAL_TEST_DATABASE_URL)


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Settings are cached per process; reset between tests so env changes apply."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    return Settings(app_env="test", database_url=os.environ["DATABASE_URL"])


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    # `with` runs the app's lifespan (startup + shutdown), like a real server.
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Tag every test under tests/integration/ so `pytest -m "not integration"` works."""
    for item in items:
        if "integration" in item.path.parts:
            item.add_marker(pytest.mark.integration)
