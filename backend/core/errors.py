"""Domain exceptions and their HTTP mapping."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.core.logging import request_id_var

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for errors that map cleanly onto an HTTP response."""

    status_code = status.HTTP_400_BAD_REQUEST
    code = "bad_request"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidInputError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "invalid_input"


class ModelNotReadyError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "model_not_ready"


class UnknownModelError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "unknown_model"


class ScrapeError(AppError):
    """The URL could not be fetched or no article text could be extracted."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "scrape_failed"


class UpstreamError(ScrapeError):
    """The remote site misbehaved (timeout, 5xx, ...)."""

    status_code = status.HTTP_502_BAD_GATEWAY
    code = "upstream_error"


def _body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message, "request_id": request_id_var.get()}}


_HTTP_CODES = {
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    429: "rate_limited",
}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, "http_error")
        return JSONResponse(
            status_code=exc.status_code,
            content=_body(code, str(exc.detail)),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        logger.info("%s: %s", exc.code, exc.message)
        return JSONResponse(status_code=exc.status_code, content=_body(exc.code, exc.message))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_body("internal_error", "An unexpected error occurred"),
        )
