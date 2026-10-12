"""Health endpoints used by load balancers, Cloud Run, and humans.

/health (liveness): "is the process up?" Never touches the database, so a
slow database can't make the platform kill an otherwise healthy container.

/ready (readiness): "can this instance serve real traffic?" Pings the
database with a short timeout. Returns 503 when it can't, so traffic is
routed elsewhere until the database is back.
"""

import asyncio
from typing import Literal

import structlog
from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from grounded import __version__
from grounded.api.deps import EngineDep

router = APIRouter(tags=["health"])
logger = structlog.get_logger(__name__)

READINESS_TIMEOUT_SECONDS = 2.0


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "unavailable"]
    database: Literal["ok", "unreachable"]


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadyResponse,
            "content": {
                "application/json": {
                    "example": {"status": "unavailable", "database": "unreachable"}
                }
            },
        }
    },
)
async def ready(engine: EngineDep, response: Response) -> ReadyResponse:
    try:
        async with asyncio.timeout(READINESS_TIMEOUT_SECONDS):
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
    except (TimeoutError, SQLAlchemyError, OSError) as exc:
        # Log the error type only: DB error messages can contain hostnames or credentials.
        logger.warning("readiness_check_failed", error_type=type(exc).__name__)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(status="unavailable", database="unreachable")

    return ReadyResponse(status="ready", database="ok")
