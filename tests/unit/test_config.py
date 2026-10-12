import pytest
from pydantic import ValidationError

from grounded.config import Settings

VALID_URL = "postgresql+asyncpg://u:p@localhost:5432/db"


def test_valid_settings_load() -> None:
    settings = Settings(database_url=VALID_URL, app_env="prod")

    assert settings.database_url == VALID_URL
    assert settings.json_logs is True


def test_local_env_uses_human_readable_logs() -> None:
    assert Settings(database_url=VALID_URL, app_env="local").json_logs is False


def test_sync_driver_url_is_rejected() -> None:
    with pytest.raises(ValidationError, match="postgresql\\+asyncpg://"):
        Settings(database_url="postgresql://u:p@localhost:5432/db")


def test_missing_database_url_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValidationError, match="database_url"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_invalid_log_level_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(database_url=VALID_URL, log_level="LOUD")  # type: ignore[arg-type]


def test_pool_size_bounds_are_enforced() -> None:
    with pytest.raises(ValidationError):
        Settings(database_url=VALID_URL, db_pool_size=0)
