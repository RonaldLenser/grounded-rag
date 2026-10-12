"""Database engine and session factory.

The engine owns a pool of open connections to Postgres (reused across
requests, because opening a connection per request is slow). A session is a
short-lived unit of work borrowed from that pool for one request.
"""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from grounded.config import Settings


def create_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        pool_pre_ping=True,  # detect dead connections (e.g. after a DB restart) before using them
        echo=settings.db_echo,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False: objects stay readable after commit, which async code needs
    # because lazy-reloading an expired attribute would require an implicit DB call.
    return async_sessionmaker(engine, expire_on_commit=False)
