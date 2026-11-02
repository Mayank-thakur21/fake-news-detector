"""HTTP endpoints.

Handlers are plain ``def`` functions on purpose: model inference and
scraping are blocking, so FastAPI runs them in its threadpool instead of
blocking the event loop.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import PlainTextResponse

from backend import __version__
from backend.api.deps import get_analyzer, get_registry, get_store
from backend.core.config import Settings, get_settings
from backend.core.errors import AppError, InvalidInputError
from backend.core.rate_limit import rate_limit
from backend.core.security import require_api_key
from backend.schemas import (
    BatchRequest,
    BatchResponse,
    CompareRequest,
    CompareResponse,
    ErrorResponse,
    HealthResponse,
    ModelsResponse,
    PredictionResponse,
    PredictRequest,
    ScrapeRequest,
    ScrapeResponse,
    StatsResponse,
)
from backend.services.analyzer import NewsAnalyzer
from backend.services.model_registry import MODEL_KEYS, MODEL_NAMES, ModelRegistry
from backend.services.prediction_store import PredictionStore

_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (401, 422, 429, 502, 503)
}

public = APIRouter(tags=["meta"])
api = APIRouter(responses=_ERRORS)

predict_limit = rate_limit("predict", lambda s: s.rate_limit_per_minute)
batch_limit = rate_limit("batch", lambda s: s.batch_rate_limit_per_minute)


# --------------------------------------------------------------------------- #
# Public (no auth) endpoints
# --------------------------------------------------------------------------- #
@public.get("/", summary="API information")
def root(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    return {
        "name": settings.app_name,
        "version": __version__,
        "docs": "/docs",
        "auth_required": settings.auth_enabled,
    }


@public.get("/health", response_model=HealthResponse, summary="Liveness / readiness probe")
def health(registry: ModelRegistry = Depends(get_registry)) -> HealthResponse:
    return HealthResponse(
        status="ok" if registry.ready else "degraded",
        models_loaded=list(registry.models),
        detail=registry.load_error,
        version=__version__,
    )


# --------------------------------------------------------------------------- #
# Prediction endpoints
# --------------------------------------------------------------------------- #
@api.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary="Classify one article (text or URL)",
    dependencies=[Depends(predict_limit)],
)
def predict(
    body: PredictRequest,
    request: Request,
    analyzer: NewsAnalyzer = Depends(get_analyzer),
) -> dict[str, Any]:
    return analyzer.predict(
        text=body.text,
        url=str(body.url) if body.url else None,
        model_key=body.model,
        want_explain=body.explain,
        top_k=body.top_k,
        client_id=getattr(request.state, "api_key_id", None),
    )


@api.post(
    "/predict/batch",
    response_model=BatchResponse,
    tags=["prediction"],
    summary="Classify many articles in one request",
    dependencies=[Depends(batch_limit)],
)
def predict_batch(
    body: BatchRequest,
    request: Request,
    analyzer: NewsAnalyzer = Depends(get_analyzer),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    if len(body.items) > settings.batch_max_items:
        raise InvalidInputError(
            f"Batch too large: {len(body.items)} items (max {settings.batch_max_items})"
        )
    analyzer.registry.ensure_ready()  # fail fast with 503 rather than N item errors
    results, confidences, fake = [], [], 0
    for i, item in enumerate(body.items):
        try:
            res = analyzer.predict(
                text=item.text,
                url=str(item.url) if item.url else None,
                model_key=body.model,
                want_explain=body.explain,
                top_k=body.top_k,
                source_kind="batch",
                client_id=getattr(request.state, "api_key_id", None),
            )
            results.append({"index": i, "id": item.id, "ok": True, "result": res})
            confidences.append(res["confidence"])
            fake += res["is_fake"]
        except AppError as exc:  # one bad item must not sink the whole batch
            results.append({"index": i, "id": item.id, "ok": False, "error": exc.message})
    ok = len(confidences)
    return {
        "results": results,
        "summary": {
            "total": len(body.items),
            "succeeded": ok,
            "failed": len(body.items) - ok,
            "fake": fake,
            "real": ok - fake,
            "average_confidence": round(sum(confidences) / ok, 2) if ok else None,
        },
    }


@api.post(
    "/compare",
    response_model=CompareResponse,
    tags=["prediction"],
    summary="Run every model on the same article and compare",
    dependencies=[Depends(predict_limit)],
)
def compare(
    body: CompareRequest,
    request: Request,
    analyzer: NewsAnalyzer = Depends(get_analyzer),
) -> dict[str, Any]:
    return analyzer.compare(
        text=body.text,
        url=str(body.url) if body.url else None,
        want_explain=body.explain,
        top_k=body.top_k,
        client_id=getattr(request.state, "api_key_id", None),
    )


@api.post(
    "/scrape",
    response_model=ScrapeResponse,
    tags=["prediction"],
    summary="Fetch a URL and return the extracted article (no prediction)",
    dependencies=[Depends(predict_limit)],
)
def scrape(body: ScrapeRequest, analyzer: NewsAnalyzer = Depends(get_analyzer)) -> dict[str, Any]:
    art = analyzer.scraper.scrape(str(body.url))
    return {
        "url": art.url,
        "final_url": art.final_url,
        "title": art.title,
        "site_name": art.site_name,
        "published_at": art.published_at,
        "word_count": art.word_count,
        "text": art.text,
    }


# --------------------------------------------------------------------------- #
# Models, stats & history
# --------------------------------------------------------------------------- #
def _models_payload(registry: ModelRegistry, settings: Settings) -> dict[str, Any]:
    report = registry.report or {}
    default = registry.resolve_default(settings.default_model) if registry.ready else None
    return {
        "default_model": default,
        "best_model": report.get("best_model"),
        "trained_at": report.get("trained_at"),
        "dataset": report.get("dataset"),
        "vectorizer": report.get("vectorizer"),
        "models": [
            {
                "key": key,
                "name": MODEL_NAMES[key],
                "loaded": key in registry.models,
                "is_default": key == default,
                "metrics": report.get("models", {}).get(key, {}),
            }
            for key in MODEL_KEYS
        ],
    }


@api.get("/models", response_model=ModelsResponse, tags=["models"], summary="Available models")
def models(
    registry: ModelRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
    _: str | None = Depends(require_api_key),
) -> dict[str, Any]:
    return _models_payload(registry, settings)


@api.get(
    "/stats",
    response_model=StatsResponse,
    tags=["models"],
    summary="Model accuracy and prediction counters",
)
def stats(
    request: Request,
    registry: ModelRegistry = Depends(get_registry),
    store: PredictionStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
    _: str | None = Depends(require_api_key),
) -> dict[str, Any]:
    models_info = _models_payload(registry, settings)
    return {
        **store.stats(),
        "model_accuracy": {m["key"]: m["metrics"].get("accuracy") for m in models_info["models"]},
        "models": models_info,
        "uptime_seconds": round(time.monotonic() - request.app.state.started_at, 1),
        "version": __version__,
    }


@api.get("/history", tags=["history"], summary="Most recent predictions")
def history(
    limit: int = Query(50, ge=1, le=1000),
    store: PredictionStore = Depends(get_store),
    _: str | None = Depends(require_api_key),
) -> list[dict[str, Any]]:
    return store.recent(limit)


@api.get(
    "/history/export",
    tags=["history"],
    summary="Download prediction history as CSV",
    response_class=PlainTextResponse,
)
def history_export(
    limit: int = Query(10_000, ge=1, le=100_000),
    store: PredictionStore = Depends(get_store),
    _: str | None = Depends(require_api_key),
) -> PlainTextResponse:
    filename = f"predictions_{datetime.now(UTC):%Y%m%d_%H%M%S}.csv"
    return PlainTextResponse(
        store.export_csv(limit),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@api.post("/admin/reload", tags=["models"], summary="Reload model artifacts from disk")
def reload_models(
    registry: ModelRegistry = Depends(get_registry),
    _: str | None = Depends(require_api_key),
) -> dict[str, Any]:
    registry.load()
    return {"ready": registry.ready, "models": list(registry.models), "detail": registry.load_error}
