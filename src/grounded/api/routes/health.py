"""Health endpoints used by load balancers, Cloud Run, and humans.

/health is liveness: "is the process up?" It must never touch external
dependencies, so a slow database cannot make the platform kill a healthy
container. A separate /ready (readiness) check that pings the database
arrives in M1.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from grounded import __version__

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)
