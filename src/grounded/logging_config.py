"""Structured logging setup.

Every log line is a set of key/value pairs instead of a free-text sentence.
In production they are rendered as JSON so a log platform can filter on
fields (request_id, status, duration_ms). Locally they render as colored,
human-readable lines.

Named logging_config.py (not logging.py) so it never shadows Python's
standard-library `logging` module.
"""

import logging
import sys

import structlog


def configure_logging(level: str, json_logs: bool) -> None:
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,  # pulls in request_id bound per request
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    renderer: list[structlog.typing.Processor] = (
        [structlog.processors.dict_tracebacks, structlog.processors.JSONRenderer()]
        if json_logs
        else [structlog.dev.ConsoleRenderer()]
    )

    structlog.configure(
        processors=[*shared_processors, *renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelNamesMapping()[level]),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=False,
    )
