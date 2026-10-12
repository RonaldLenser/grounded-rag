"""Domain errors.

Services raise these; they know nothing about HTTP. The API layer
(api/errors.py) maps each one to a status code and a consistent JSON body.
That keeps business logic reusable from scripts and evals, not just routes.
"""


class AppError(Exception):
    """Base class. `code` is a stable, machine-readable identifier for clients."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
        self.headers: dict[str, str] = {}


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class DuplicateDocumentError(ConflictError):
    code = "duplicate_document"

    def __init__(self, existing_document_id: str) -> None:
        super().__init__(f"This file was already uploaded as document {existing_document_id}")
        self.existing_document_id = existing_document_id


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"


class UnsupportedDocumentError(AppError):
    status_code = 415
    code = "unsupported_document"


class UnprocessableDocumentError(AppError):
    """The file type is supported but its contents can't be used (e.g. scanned PDF)."""

    status_code = 422
    code = "unprocessable_document"


class AuthenticationError(AppError):
    status_code = 401
    code = "unauthorized"


class WritesDisabledError(AppError):
    status_code = 403
    code = "writes_disabled"


class ProviderNotConfiguredError(AppError):
    status_code = 503
    code = "provider_not_configured"


class EmbeddingProviderError(AppError):
    """The embedding API failed after retries. Our fault to report, not the client's."""

    status_code = 502
    code = "embedding_provider_error"


class LLMProviderError(AppError):
    """The LLM API failed after retries, or returned output we can't use."""

    status_code = 502
    code = "llm_provider_error"


class RateLimitedError(AppError):
    """Too many requests from one client. Retry-After tells them when to retry."""

    status_code = 429
    code = "rate_limited"

    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__(f"Too many requests. Try again in {retry_after_seconds} seconds.")
        self.headers = {"Retry-After": str(retry_after_seconds)}


class DailyLimitError(AppError):
    """The demo's global daily LLM budget is used up."""

    status_code = 429
    code = "daily_limit_reached"
