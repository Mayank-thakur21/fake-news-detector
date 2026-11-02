"""Business logic: turn an article (text or URL) into predictions."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any

from backend.core.config import Settings
from backend.core.errors import InvalidInputError
from backend.services.model_registry import LoadedModel, ModelRegistry
from backend.services.prediction_store import PredictionStore
from backend.services.scraper import ArticleScraper
from ml import FAKE, LABELS, REAL
from ml.explain import explain
from ml.preprocessing import prepare

logger = logging.getLogger(__name__)


class NewsAnalyzer:
    def __init__(
        self,
        registry: ModelRegistry,
        store: PredictionStore,
        scraper: ArticleScraper,
        settings: Settings,
    ):
        self.registry = registry
        self.store = store
        self.scraper = scraper
        self.settings = settings

    # ------------------------------------------------------------------ #
    # Input handling
    # ------------------------------------------------------------------ #
    def resolve_article(self, text: str | None, url: str | None) -> tuple[str, dict[str, Any]]:
        """Return ``(article_text, source_info)``; scrapes the URL when given."""
        if url:
            art = self.scraper.scrape(str(url))
            body = f"{art.title}. {art.text}" if art.title else art.text
            source = {
                "type": "url",
                "url": art.url,
                "final_url": art.final_url,
                "title": art.title,
                "site_name": art.site_name,
                "published_at": art.published_at,
            }
        else:
            body = (text or "").strip()
            source = {"type": "text"}

        if len(body) > self.settings.max_text_chars:
            body = body[: self.settings.max_text_chars]
            logger.info("Article truncated to %d characters", self.settings.max_text_chars)
        cleaned = prepare(body)
        n_words = len(cleaned.split()) if cleaned else 0
        if n_words < self.settings.min_words:
            raise InvalidInputError(
                f"Article is too short to analyse ({n_words} usable words; "
                f"need at least {self.settings.min_words})"
            )
        source["word_count"] = len(body.split())
        return body, source

    # ------------------------------------------------------------------ #
    # Core prediction
    # ------------------------------------------------------------------ #
    def _vectorize(self, text: str):
        self.registry.ensure_ready()
        return self.registry.vectorizer.transform([prepare(text)])

    def _verdict(self, model: LoadedModel, x, *, want_explain: bool, top_k: int) -> dict[str, Any]:
        est = model.estimator
        proba = est.predict_proba(x)[0]
        classes = list(est.classes_)
        p_fake, p_real = float(proba[classes.index(FAKE)]), float(proba[classes.index(REAL)])
        is_fake = p_fake >= p_real
        out: dict[str, Any] = {
            "model": model.key,
            "model_name": model.name,
            "prediction": LABELS[FAKE if is_fake else REAL],
            "is_fake": is_fake,
            "confidence": round(100 * max(p_fake, p_real), 2),
            "probabilities": {"fake": round(p_fake, 4), "real": round(p_real, 4)},
            "explanation": None,
        }
        if want_explain:
            try:
                exp = explain(est, x, self.registry.feature_names, top_k=top_k)
                out["explanation"] = asdict(exp)
            except Exception:  # explanations are best-effort; never fail the prediction
                logger.exception("Explanation failed for model %s", model.key)
        return out

    def predict(
        self,
        *,
        text: str | None = None,
        url: str | None = None,
        model_key: str | None = None,
        want_explain: bool = True,
        top_k: int = 10,
        source_kind: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        key = model_key or self.registry.resolve_default(self.settings.default_model)
        model = self.registry.get(key)  # validates key before any scraping work
        body, source = self.resolve_article(text, url)
        result = self._verdict(model, self._vectorize(body), want_explain=want_explain, top_k=top_k)
        result["source"] = source
        result["processing_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["prediction_id"] = self.store.record(
            model=key,
            label=result["prediction"],
            is_fake=result["is_fake"],
            confidence=result["confidence"],
            source=source_kind or source["type"],
            text=body,
            url=source.get("url"),
            title=source.get("title"),
            client_id=client_id,
        )
        logger.info(
            "Prediction %s: %s (%.1f%%) via %s in %.0fms",
            result["prediction_id"],
            result["prediction"],
            result["confidence"],
            key,
            result["processing_ms"],
        )
        return result

    def compare(
        self,
        *,
        text: str | None = None,
        url: str | None = None,
        want_explain: bool = False,
        top_k: int = 5,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        """Run every loaded model on the same article."""
        t0 = time.perf_counter()
        self.registry.ensure_ready()
        body, source = self.resolve_article(text, url)
        x = self._vectorize(body)
        verdicts = []
        for model in self.registry.models.values():
            v = self._verdict(model, x, want_explain=want_explain, top_k=top_k)
            v.pop("is_fake")
            v["test_accuracy"] = model.metrics.get("accuracy")
            v["test_f1"] = model.metrics.get("f1")
            verdicts.append(v)

        mean_fake = sum(v["probabilities"]["fake"] for v in verdicts) / len(verdicts)
        fake_votes = sum(v["prediction"] == LABELS[FAKE] for v in verdicts)
        # Majority vote; ties are broken by the mean probability.
        if fake_votes * 2 == len(verdicts):
            consensus = LABELS[FAKE] if mean_fake >= 0.5 else LABELS[REAL]
        else:
            consensus = LABELS[FAKE] if fake_votes * 2 > len(verdicts) else LABELS[REAL]
        agreeing = sum(v["prediction"] == consensus for v in verdicts)

        self.store.record(
            model="ensemble",
            label=consensus,
            is_fake=consensus == LABELS[FAKE],
            confidence=round(100 * max(mean_fake, 1 - mean_fake), 2),
            source=f"compare:{source['type']}",
            text=body,
            url=source.get("url"),
            title=source.get("title"),
            client_id=client_id,
        )
        return {
            "results": verdicts,
            "consensus": consensus,
            "agreement": round(100 * agreeing / len(verdicts), 1),
            "mean_fake_probability": round(mean_fake, 4),
            "source": source,
            "processing_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
