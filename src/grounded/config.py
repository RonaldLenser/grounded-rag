"""Application settings, loaded once from environment variables (and .env locally).

Every setting is typed and validated at startup. A missing or malformed value
crashes the app immediately with a clear error, instead of failing later in
the middle of a request. Settings are added here only when code uses them.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env holds vars for later milestones; don't reject them
    )

    app_env: Literal["local", "test", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Required: no default, so prod can never silently fall back to localhost.
    database_url: str
    db_pool_size: int = Field(default=5, ge=1, le=50)
    db_echo: bool = False  # log every SQL statement (debugging only)

    @field_validator("database_url")
    @classmethod
    def require_async_driver(cls, value: str) -> str:
        if not value.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "DATABASE_URL must start with 'postgresql+asyncpg://' (async driver required)"
            )
        return value

    @property
    def json_logs(self) -> bool:
        """Machine-readable logs everywhere except a developer's terminal."""
        return self.app_env != "local"


@lru_cache
def get_settings() -> Settings:
    """Build settings once per process and reuse them."""
    return Settings()  # type: ignore[call-arg]  # values come from the environment
