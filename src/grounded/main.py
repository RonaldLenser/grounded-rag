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
from fastapi import FastAPI

from grounded import __version__
from grounded.api.middleware import RequestContextMiddleware
from grounded.api.routes import health
from grounded.config import Settings, get_settings
from grounded.db.session import create_engine, create_session_factory
from grounded.logging_config import configure_logging

logger = structlog.get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_logs=settings.json_logs)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup: create the connection pool once for the whole process.
        # (Creating the engine does not connect yet, so the app still starts
        # if the database is down; /ready reports it instead.)
        engine = create_engine(settings)
        app.state.engine = engine
        app.state.session_factory = create_session_factory(engine)
        logger.info("app_started", env=settings.app_env, version=__version__)
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
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health.router)
    return app
