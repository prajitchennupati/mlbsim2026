"""Structured logging configuration.

Call :func:`configure_logging` once at process start (CLI, API startup, test
session). Everywhere else use ``get_logger(__name__)``.
"""

from __future__ import annotations

import logging
import sys
from typing import cast

import structlog

from mlbsim_core.config import get_settings

_CONFIGURED = False


def configure_logging(*, force: bool = False) -> None:
    """Configure stdlib + structlog once, honoring ``MLBSIM_LOG_*`` settings."""
    global _CONFIGURED  # noqa: PLW0603 -- module-level idempotency guard
    if _CONFIGURED and not force:
        return

    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level)

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, configuring logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    return cast("structlog.stdlib.BoundLogger", structlog.get_logger(name))
