"""Application entry point.

create_app() is an app factory: it builds a fresh, fully configured FastAPI
instance each time it is called. Tests pass in their own Settings to get an
isolated app. Uvicorn calls it with --factory, so nothing (like reading
DATABASE_URL) happens merely because this module was imported.

Run locally:  uv run uvicorn grounded.main:create_app --factory --reload
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from anthropic import AsyncAnthropic
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

from grounded import __version__
from grounded.api.errors import register_error_handlers
from grounded.api.middleware import RequestContextMiddleware
from grounded.api.rate_limit import SlidingWindowRateLimiter
from grounded.api.routes import collections, health, query, search, web
from grounded.config import Settings, get_settings
from grounded.db.session import create_engine, create_session_factory
from grounded.logging_config import configure_logging
from grounded.providers.anthropic_llm import AnthropicLLM
from grounded.providers.base import Embedder, LLMClient
from grounded.providers.openai_embedder import OpenAIEmbedder

logger = structlog.get_logger(__name__)


def build_embedder(settings: Settings) -> Embedder | None:
    """None when no API key is set: the app still runs; ingestion returns 503."""
    if settings.openai_api_key is None:
        return None
    client = AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=settings.embedding_timeout_seconds,
        max_retries=settings.embedding_max_retries,
    )
    return OpenAIEmbedder(
        client=client,
        model=settings.embedding_model,
        dimension=settings.embedding_dim,
        batch_size=settings.embedding_batch_size,
    )


def build_llm(settings: Settings) -> LLMClient | None:
    """None when no API key is set: the app still runs; /query returns 503."""
    if settings.anthropic_api_key is None:
        return None
    client = AsyncAnthropic(
        api_key=settings.anthropic_api_key.get_secret_value(),
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
    return AnthropicLLM(client=client, model=settings.llm_model)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_logs=settings.json_logs)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup: create long-lived resources once for the whole process.
        # (Creating the engine does not connect yet, so the app still starts
        # if the database is down; /ready reports it instead.)
        engine = create_engine(settings)
        app.state.engine = engine
        app.state.session_factory = create_session_factory(engine)
        app.state.embedder = build_embedder(settings)
        app.state.llm = build_llm(settings)
        logger.info(
            "app_started",
            env=settings.app_env,
            version=__version__,
            embeddings_enabled=app.state.embedder is not None,
            answers_enabled=app.state.llm is not None,
            writes_enabled=settings.admin_api_key is not None,
        )
        try:
            yield
        finally:
            # Shutdown: close pooled connections cleanly.
            await engine.dispose()
            logger.info("app_stopped")

    app = FastAPI(
        title="Grounded",
        description="Answers questions only from your documents, with verified citations.",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.rate_limiters = {
        "query": SlidingWindowRateLimiter(settings.rate_limit_query_per_minute),
        "search": SlidingWindowRateLimiter(settings.rate_limit_search_per_minute),
    }
    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(collections.router)
    app.include_router(search.router)
    app.include_router(query.router)
    app.include_router(web.router)
    app.mount("/static", StaticFiles(directory=web.WEB_DIR), name="static")
    return app
