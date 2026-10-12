"""Application entry point.

create_app() is an app factory: it builds a fresh FastAPI instance each
time it is called. Tests call it to get an isolated app, and later
milestones use it to inject settings and swap real providers for fakes.
"""

from fastapi import FastAPI

from grounded import __version__
from grounded.api.routes import health


def create_app() -> FastAPI:
    app = FastAPI(
        title="Grounded",
        description="Answers questions only from your documents, with verified citations.",
        version=__version__,
    )
    app.include_router(health.router)
    return app


app = create_app()
