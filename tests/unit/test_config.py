import pytest
from pydantic import ValidationError

from grounded.config import Settings

VALID_URL = "postgresql+asyncpg://u:p@localhost:5432/db"


def test_valid_settings_load() -> None:
    settings = Settings(
        database_url=VALID_URL,
        app_env="prod",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        anthropic_api_key="sk-ant-test",  # type: ignore[arg-type]
        admin_api_key="a-long-enough-admin-key",  # type: ignore[arg-type]
    )

    assert settings.database_url == VALID_URL
    assert settings.json_logs is True


def test_local_env_uses_human_readable_logs() -> None:
    assert Settings(database_url=VALID_URL, app_env="local").json_logs is False


def test_neon_style_url_is_normalized_for_asyncpg() -> None:
    raw = "postgresql://u:p@ep-cool-1.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

    settings = Settings(database_url=raw)

    assert settings.database_url == (
        "postgresql+asyncpg://u:p@ep-cool-1.us-east-1.aws.neon.tech/neondb?ssl=require"
    )


def test_postgres_scheme_alias_is_accepted() -> None:
    assert Settings(database_url="postgres://u:p@h/db").database_url == (
        "postgresql+asyncpg://u:p@h/db"
    )


def test_non_postgres_url_is_rejected() -> None:
    with pytest.raises(ValidationError, match="PostgreSQL URL"):
        Settings(database_url="mysql://u:p@localhost/db")


def test_pooler_endpoint_is_rejected_with_a_clear_message() -> None:
    with pytest.raises(ValidationError, match="direct connection string"):
        Settings(database_url="postgresql://u:p@ep-x-pooler.us-east-1.aws.neon.tech/db")


def test_secrets_with_trailing_newlines_are_stripped() -> None:
    settings = Settings(
        database_url=VALID_URL + "\r\n",
        openai_api_key="sk-abc\r\n",  # type: ignore[arg-type]
    )

    assert settings.database_url == VALID_URL
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-abc"


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


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP_TOKENS"):
        Settings(database_url=VALID_URL, chunk_size_tokens=100, chunk_overlap_tokens=100)


def test_embedding_dim_must_match_the_database_column() -> None:
    with pytest.raises(ValidationError, match="requires a migration"):
        Settings(database_url=VALID_URL, embedding_dim=3072)


def test_prod_requires_api_keys() -> None:
    with pytest.raises(ValidationError, match="OPENAI_API_KEY, ANTHROPIC_API_KEY, ADMIN_API_KEY"):
        Settings(database_url=VALID_URL, app_env="prod", _env_file=None)  # type: ignore[call-arg]


def test_short_admin_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least 16"):
        Settings(database_url=VALID_URL, admin_api_key="short")  # type: ignore[arg-type]


def test_secrets_are_hidden_in_repr() -> None:
    settings = Settings(
        database_url=VALID_URL,
        openai_api_key="sk-supersecretvalue",  # type: ignore[arg-type]
    )

    assert "sk-supersecretvalue" not in repr(settings)


def test_empty_keys_are_treated_as_unset() -> None:
    settings = Settings(
        database_url=VALID_URL,
        openai_api_key="",  # type: ignore[arg-type]
        admin_api_key="",  # type: ignore[arg-type]
    )

    assert settings.openai_api_key is None
    assert settings.admin_api_key is None


def test_validation_errors_never_echo_credentials() -> None:
    """A rejected DATABASE_URL must not leak its password into the error message."""
    with pytest.raises(ValidationError) as excinfo:
        Settings(database_url='"postgresql://user:hunter2-secret@host/db"')

    assert "hunter2-secret" not in str(excinfo.value)
