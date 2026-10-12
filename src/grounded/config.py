"""Application settings, loaded once from environment variables (and .env locally).

Every setting is typed and validated at startup. A missing or malformed value
crashes the app immediately with a clear error, instead of failing later in
the middle of a request. Settings are added here only when code uses them.
"""

from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from grounded.db.models import EMBEDDING_DIM


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env holds vars for later milestones; don't reject them
        # Secrets piped into a secret manager often gain a trailing newline,
        # which silently breaks API keys. Strip surrounding whitespace everywhere.
        str_strip_whitespace=True,
        # Validation errors must never echo the rejected value: DATABASE_URL and
        # API keys contain credentials, and errors end up in terminals and logs.
        hide_input_in_errors=True,
    )

    # ----- App -----
    app_env: Literal["local", "test", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ----- Database -----
    # Required: no default, so prod can never silently fall back to localhost.
    database_url: str
    db_pool_size: int = Field(default=5, ge=1, le=50)
    db_echo: bool = False  # log every SQL statement (debugging only)

    # ----- Security -----
    # Protects write endpoints (create collection, upload, delete).
    # Unset = writes are disabled (safe default for a public demo).
    admin_api_key: SecretStr | None = None

    # ----- Embeddings -----
    # SecretStr hides the value in logs, reprs, and error messages.
    openai_api_key: SecretStr | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = EMBEDDING_DIM
    embedding_batch_size: int = Field(default=100, ge=1, le=2048)
    embedding_timeout_seconds: float = Field(default=30.0, gt=0)
    embedding_max_retries: int = Field(default=3, ge=0, le=10)

    # ----- Generation (answers) -----
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-sonnet-5-5"
    # Claim-by-claim JSON (each with quotes) is wordier than prose: list answers
    # reached ~950 tokens in the M7 eval and one hit 1024 (cut off -> 502).
    # Billing is per token generated, so a higher cap only costs when used.
    llm_max_tokens: int = Field(default=2048, ge=64, le=8192)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    # Below this best-match similarity we refuse WITHOUT calling the LLM.
    # Provisional value; calibrated against the eval set in M7.
    min_relevance_score: float = Field(default=0.20, ge=-1.0, le=1.0)
    # Cap on source text sent to the LLM per question (cost + focus).
    max_context_tokens: int = Field(default=6000, ge=500, le=100_000)

    # ----- Ingestion -----
    chunk_size_tokens: int = Field(default=500, ge=50, le=8000)
    chunk_overlap_tokens: int = Field(default=75, ge=0)
    max_upload_mb: int = Field(default=10, ge=1, le=100)
    max_pdf_pages: int = Field(default=500, ge=1)

    # ----- Abuse / cost protection (public demo) -----
    # Per client IP, per app instance (in-memory sliding window).
    rate_limit_query_per_minute: int = Field(default=10, ge=1, le=10_000)
    rate_limit_search_per_minute: int = Field(default=30, ge=1, le=10_000)
    # Global cap on LLM calls per UTC day. Each call reserves a slot atomically
    # in Postgres before it runs, so it holds across instances and under
    # concurrency, and failed calls count too. Worst-case daily spend is bounded
    # by this (times 1 + llm_max_retries HTTP attempts per call).
    daily_llm_query_limit: int = Field(default=300, ge=1)
    # How many proxies in front of the app append to X-Forwarded-For.
    # 0 = trust nothing, use the TCP peer (local / docker compose).
    # 1 = Cloud Run: use the LAST X-Forwarded-For entry (added by Google's front end).
    # Never use the FIRST entry: clients can put anything there.
    trusted_proxy_hops: int = Field(default=0, ge=0, le=5)

    # ----- Retrieval -----
    retrieval_top_k: int = Field(default=6, ge=1, le=20)
    # HNSW search breadth: higher = better recall, slower. pgvector's default is 40.
    hnsw_ef_search: int = Field(default=100, ge=10, le=1000)

    @field_validator("database_url")
    @classmethod
    def normalize_database_url(cls, value: str) -> str:
        """Accept the connection string Postgres hosts (e.g. Neon) give you, as-is.

        - postgres:// or postgresql://  ->  postgresql+asyncpg://  (async driver)
        - sslmode=require  ->  ssl=require  (asyncpg's spelling)
        - channel_binding=...  dropped (libpq-only option asyncpg doesn't accept)
        """
        scheme, sep, rest = value.partition("://")
        if not sep or scheme not in ("postgres", "postgresql", "postgresql+asyncpg"):
            raise ValueError("DATABASE_URL must be a PostgreSQL URL (postgresql://...)")
        base, _, query = rest.partition("?")
        params = []
        for param in filter(None, query.split("&")):
            key, _, val = param.partition("=")
            if key == "channel_binding":
                continue
            params.append(f"ssl={val}" if key == "sslmode" else param)
        if "-pooler." in base:
            raise ValueError(
                "DATABASE_URL points at a connection pooler (PgBouncer). asyncpg's prepared "
                "statements break in transaction pooling mode: use the direct connection string."
            )
        return f"postgresql+asyncpg://{base}" + (f"?{'&'.join(params)}" if params else "")

    @field_validator("admin_api_key", "openai_api_key", "anthropic_api_key", mode="before")
    @classmethod
    def empty_means_unset(cls, value: object) -> object:
        # Docker Compose passes unset variables through as "" rather than omitting them.
        return None if value == "" else value

    @field_validator("admin_api_key")
    @classmethod
    def require_strong_admin_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < 16:
            raise ValueError("ADMIN_API_KEY must be at least 16 characters")
        return value

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS")
        if self.embedding_dim != EMBEDDING_DIM:
            raise ValueError(
                f"EMBEDDING_DIM={self.embedding_dim} but the database column is "
                f"vector({EMBEDDING_DIM}); changing it requires a migration"
            )
        # Production must be fully configured. Local dev can run without keys.
        if self.app_env == "prod":
            missing = [
                name
                for name, value in (
                    ("OPENAI_API_KEY", self.openai_api_key),
                    ("ANTHROPIC_API_KEY", self.anthropic_api_key),
                    ("ADMIN_API_KEY", self.admin_api_key),
                )
                if value is None
            ]
            if missing:
                raise ValueError(f"Missing required settings for prod: {', '.join(missing)}")
        return self

    @property
    def json_logs(self) -> bool:
        """Machine-readable logs everywhere except a developer's terminal."""
        return self.app_env != "local"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Build settings once per process and reuse them."""
    return Settings()  # type: ignore[call-arg]  # values come from the environment
