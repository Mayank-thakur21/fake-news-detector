"""Application factory and ASGI entrypoint.

Run locally with::

    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import __version__
from backend.api.routes import api, public
from backend.core.config import Settings, get_settings
from backend.core.errors import register_exception_handlers
from backend.core.logging import configure_logging, request_id_var
from backend.services.analyzer import NewsAnalyzer
from backend.services.model_registry import ModelRegistry
from backend.services.prediction_store import PredictionStore
from backend.services.scraper import ArticleScraper

logger = logging.getLogger("backend")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        registry = ModelRegistry(settings.model_dir)
        registry.load()
        store = PredictionStore(settings.db_path)
        scraper = ArticleScraper(
            timeout=settings.scraper_timeout,
            max_bytes=settings.scraper_max_bytes,
            user_agent=settings.scraper_user_agent,
            allow_private_hosts=settings.scraper_allow_private_hosts,
        )
        app.state.registry = registry
        app.state.store = store
        app.state.analyzer = NewsAnalyzer(registry, store, scraper, settings)
        app.state.started_at = time.monotonic()
        if not settings.auth_enabled:
            logger.warning("API_KEYS is empty - authentication is DISABLED")
        logger.info("%s v%s started (%s)", settings.app_name, __version__, settings.environment)
        try:
            yield
        finally:
            store.close()
            logger.info("Shutdown complete")

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Detect fake news with three classical ML models (Random Forest, Logistic "
            "Regression, Linear SVM). Supports raw text, URLs, batches, model comparison "
            "and per-feature explanations.\n\n"
            "Authenticate with the `X-API-Key` header when the server has `API_KEYS` set."
        ),
        lifespan=lifespan,
    )
    # Must be set before the first request (overrides are used by tests).
    app.dependency_overrides[get_settings] = lambda: settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,  # API-key auth uses headers, not cookies
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-ID",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "Retry-After",
        ],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
            duration = (time.perf_counter() - start) * 1000
            response.headers["X-Request-ID"] = rid
            logger.info(
                "%s %s -> %s (%.1fms)",
                request.method,
                request.url.path,
                response.status_code,
                duration,
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round(duration, 1),
                },
            )
            return response
        finally:
            request_id_var.reset(token)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Flatten pydantic's error list into one readable message.
        msgs = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", []) if p != "body")
            msgs.append(f"{loc}: {err.get('msg')}" if loc else err.get("msg", "invalid"))
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "; ".join(msgs),
                    "request_id": request_id_var.get(),
                }
            },
        )

    register_exception_handlers(app)
    app.include_router(public)
    app.include_router(api)
    return app


app = create_app()
