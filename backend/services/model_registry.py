"""Loads the vectorizer, every trained model and their metrics from disk."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from backend.core.errors import ModelNotReadyError, UnknownModelError

logger = logging.getLogger(__name__)

MODEL_KEYS = ("rf", "lr", "svm")
MODEL_NAMES = {"rf": "Random Forest", "lr": "Logistic Regression", "svm": "Linear SVM"}


@dataclass
class LoadedModel:
    key: str
    name: str
    estimator: Any
    metrics: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Thread-safe container for the inference artifacts."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.vectorizer: Any = None
        self.feature_names: np.ndarray | None = None
        self.models: dict[str, LoadedModel] = {}
        self.report: dict[str, Any] = {}
        self.load_error: str | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """(Re)load artifacts. Never raises - failures are recorded in ``load_error``."""
        with self._lock:
            self.models.clear()
            self.vectorizer = None
            self.load_error = None
            vec_path = self.model_dir / "vectorizer.joblib"
            if not vec_path.exists():
                self.load_error = (
                    f"No trained models found in {self.model_dir}. Run `python -m ml.train` first."
                )
                logger.error(self.load_error)
                return
            try:
                self.vectorizer = joblib.load(vec_path)
                self.feature_names = self.vectorizer.get_feature_names_out()
                metrics_path = self.model_dir / "metrics.json"
                self.report = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
                for key in MODEL_KEYS:
                    path = self.model_dir / f"{key}.joblib"
                    if not path.exists():
                        logger.warning("Model %s not found at %s - skipping", key, path)
                        continue
                    self.models[key] = LoadedModel(
                        key=key,
                        name=MODEL_NAMES[key],
                        estimator=joblib.load(path),
                        metrics=self.report.get("models", {}).get(key, {}),
                    )
                if not self.models:
                    self.load_error = f"Vectorizer found but no model files in {self.model_dir}"
                    logger.error(self.load_error)
                else:
                    logger.info(
                        "Loaded %d model(s) [%s] with %d features",
                        len(self.models),
                        ", ".join(self.models),
                        len(self.feature_names),
                    )
            except Exception as exc:  # corrupt pickle, version mismatch, ...
                self.models.clear()
                self.vectorizer = None
                self.load_error = f"Failed to load models: {exc}"
                logger.exception(self.load_error)

    # ------------------------------------------------------------------ #
    @property
    def ready(self) -> bool:
        return self.vectorizer is not None and bool(self.models)

    def ensure_ready(self) -> None:
        if not self.ready:
            raise ModelNotReadyError(self.load_error or "Models are not loaded")

    def get(self, key: str) -> LoadedModel:
        self.ensure_ready()
        try:
            return self.models[key]
        except KeyError:
            raise UnknownModelError(
                f"Model '{key}' is not available. Loaded models: {', '.join(self.models)}"
            ) from None

    def resolve_default(self, preferred: str) -> str:
        """Use the configured default if loaded, else the best-scoring available model."""
        self.ensure_ready()
        if preferred in self.models:
            return preferred
        best = self.report.get("best_model")
        return best if best in self.models else next(iter(self.models))
