"""FastAPI dependencies: shared resources handed to route functions.

Routes declare what they need (e.g. `session: SessionDep`) and FastAPI
provides it. Tests can override any dependency with a fake, which is how
the test suite runs ingestion without calling OpenAI.
"""

import secrets
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from grounded.config import Settings
from grounded.errors import AuthenticationError, ProviderNotConfiguredError, WritesDisabledError
from grounded.ingestion.service import IngestionService
from grounded.providers.base import Embedder
from grounded.retrieval.retriever import Retriever

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_settings_dep(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_engine(request: Request) -> AsyncEngine:
    engine: AsyncEngine = request.app.state.engine
    return engine


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """One session per request, always closed afterward (even on errors)."""
    factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    async with factory() as session:
        yield session


def get_embedder(request: Request) -> Embedder:
    embedder: Embedder | None = request.app.state.embedder
    if embedder is None:
        raise ProviderNotConfiguredError("Embeddings are not configured (OPENAI_API_KEY is unset)")
    return embedder


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
EngineDep = Annotated[AsyncEngine, Depends(get_engine)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]


def require_admin(
    settings: SettingsDep,
    api_key: Annotated[str | None, Security(API_KEY_HEADER)],
) -> None:
    """Guard for write endpoints. Constant-time comparison prevents timing attacks."""
    if settings.admin_api_key is None:
        raise WritesDisabledError("Write endpoints are disabled (ADMIN_API_KEY is unset)")
    expected = settings.admin_api_key.get_secret_value()
    if api_key is None or not secrets.compare_digest(api_key, expected):
        raise AuthenticationError("Missing or invalid X-API-Key header")


def get_ingestion_service(
    session: SessionDep, embedder: EmbedderDep, settings: SettingsDep
) -> IngestionService:
    return IngestionService(session=session, embedder=embedder, settings=settings)


AdminDep = Depends(require_admin)
IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]


def get_retriever(session: SessionDep, embedder: EmbedderDep, settings: SettingsDep) -> Retriever:
    return Retriever(session=session, embedder=embedder, settings=settings)


RetrieverDep = Annotated[Retriever, Depends(get_retriever)]
