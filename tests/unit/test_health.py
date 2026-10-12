from fastapi.testclient import TestClient

from grounded import __version__
from grounded.config import Settings
from grounded.main import create_app


def test_health_returns_ok_and_version(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_unknown_route_returns_404(client: TestClient) -> None:
    response = client.get("/does-not-exist")

    assert response.status_code == 404


def test_ready_returns_503_when_database_is_unreachable() -> None:
    # Port 1 on localhost: nothing listens there, so the connection is refused fast.
    settings = Settings(
        app_env="test",
        database_url="postgresql+asyncpg://user:pass@127.0.0.1:1/nope",
    )
    with TestClient(create_app(settings)) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "database": "unreachable"}


def test_health_stays_ok_when_database_is_unreachable() -> None:
    """Liveness must not depend on the database (see health.py docstring)."""
    settings = Settings(
        app_env="test",
        database_url="postgresql+asyncpg://user:pass@127.0.0.1:1/nope",
    )
    with TestClient(create_app(settings)) as client:
        response = client.get("/health")

    assert response.status_code == 200
