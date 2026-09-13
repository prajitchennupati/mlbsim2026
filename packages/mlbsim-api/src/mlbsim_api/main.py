"""FastAPI application factory — read-only API over the ``serving`` schema."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mlbsim_api import __version__
from mlbsim_api.routers import ROUTERS
from mlbsim_core import configure_logging, get_settings

API_PREFIX = "/api/v1"


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()

    app = FastAPI(
        title="mlbplayoffs2026 API",
        version=__version__,
        summary="AI MLB prediction & World Series simulation platform",
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_deployed else [],
        allow_origin_regex=r"https://.*\.vercel\.app" if settings.is_deployed else None,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    for router in ROUTERS:
        app.include_router(router, prefix=API_PREFIX)

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/version", tags=["ops"])
    def version() -> dict[str, Any]:
        return {"version": __version__, "environment": settings.environment}

    return app


app = create_app()
