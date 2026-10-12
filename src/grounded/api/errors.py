"""Map exceptions to HTTP responses with ONE consistent error shape:

    {"error": {"code": "duplicate_document", "message": "...", "request_id": "..."}}

Clients branch on the stable `code`; humans read `message`; `request_id`
lets anyone find the matching server log line. Unexpected exceptions return
a generic 500: stack traces stay in the logs, never in responses.
"""

from typing import Any

import structlog
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from grounded.errors import AppError, DuplicateDocumentError

logger = structlog.get_logger(__name__)


def _error_response(status_code: int, code: str, message: str, **extra: Any) -> JSONResponse:
    request_id = structlog.contextvars.get_contextvars().get("request_id")
    body: dict[str, Any] = {"code": code, "message": message, "request_id": request_id, **extra}
    return JSONResponse(status_code=status_code, content={"error": jsonable_encoder(body)})


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("request_failed", code=exc.code, message=exc.message)
        extra = (
            {"existing_document_id": exc.existing_document_id}
            if isinstance(exc, DuplicateDocumentError)
            else {}
        )
        return _error_response(exc.status_code, exc.code, exc.message, **extra)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "validation_error",
            "Request validation failed",
            details=exc.errors(),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error_response(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception", error_type=type(exc).__name__)
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", "Internal server error"
        )
