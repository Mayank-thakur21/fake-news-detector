"""Train and compare fake-news classifiers.

Trains three models on a shared TF-IDF representation and writes every
artifact the API needs into ``--model-dir``:

    vectorizer.joblib   TF-IDF vectorizer
    rf.joblib           Random Forest
    lr.joblib           Logistic Regression
    svm.joblib          Linear SVM (probability-calibrated)
    metrics.json        held-out metrics for every model + dataset metadata

Usage::

    # Kaggle / ISOT dataset (Fake.csv + True.csv)
    python -m ml.train --fake data/Fake.csv --true data/True.csv

    # Any CSV with ``text`` and ``label`` columns (label: 0/1 or FAKE/REAL)
    python -m ml.train --csv data/my_dataset.csv

    # No arguments: uses data/Fake.csv + data/True.csv when present,
    # otherwise falls back to the bundled demo sample.
    python -m ml.train
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from ml import FAKE, REAL
from ml.preprocessing import prepare

logger = logging.getLogger("ml.train")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_SAMPLE = DEFAULT_DATA_DIR / "sample" / "news_sample.csv"
DEFAULT_MODEL_DIR = ROOT / "models"

MODEL_NAMES: dict[str, str] = {
    "rf": "Random Forest",
    "lr": "Logistic Regression",
    "svm": "Linear SVM",
}

_LABEL_ALIASES = {
    "0": FAKE,
    "fake": FAKE,
    "false": FAKE,
    "fake news": FAKE,
    "1": REAL,
    "real": REAL,
    "true": REAL,
    "real news": REAL,
}


@dataclass
class TrainConfig:
    """Hyper-parameters. Defaults are tuned for the ~44k-article ISOT dataset."""

    test_size: float = 0.2
    random_state: int = 42
    max_features: int = 50_000
    ngram_max: int = 2
    rf_estimators: int = 200
    strip_leakage: bool = True
    models: tuple[str, ...] = ("rf", "lr", "svm")
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _combine_title(df: pd.DataFrame) -> pd.Series:
    text = df["text"].fillna("").astype(str)
    if "title" in df.columns:
        title = df["title"].fillna("").astype(str)
        text = (title + ". " + text).str.strip(". ")
    return text


def load_isot(fake_path: Path, true_path: Path) -> pd.DataFrame:
    """Load the Kaggle ISOT dataset (two CSVs with title/text/subject/date)."""
    fake = pd.read_csv(fake_path)
    real = pd.read_csv(true_path)
    fake = pd.DataFrame({"text": _combine_title(fake), "label": FAKE})
    real = pd.DataFrame({"text": _combine_title(real), "label": REAL})
    return pd.concat([fake, real], ignore_index=True)


def load_labelled_csv(path: Path) -> pd.DataFrame:
    """Load a single CSV with ``text`` (+ optional ``title``) and ``label`` columns."""
    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")
    labels = df["label"].astype(str).str.strip().str.lower().map(_LABEL_ALIASES)
    if labels.isna().any():
        bad = df.loc[labels.isna(), "label"].unique()[:5]
        raise ValueError(f"Unrecognised label values in {path}: {list(bad)}")
    return pd.DataFrame({"text": _combine_title(df), "label": labels.astype(int)})


def resolve_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    """Pick a dataset from CLI args, falling back to sensible defaults."""
    if args.csv:
        return load_labelled_csv(Path(args.csv)), str(args.csv)
    if args.fake and args.true:
        return load_isot(Path(args.fake), Path(args.true)), f"{args.fake} + {args.true}"

    fake, true = DEFAULT_DATA_DIR / "Fake.csv", DEFAULT_DATA_DIR / "True.csv"
    if fake.exists() and true.exists():
        return load_isot(fake, true), "ISOT (data/Fake.csv + data/True.csv)"

    logger.warning(
        "Kaggle dataset not found in %s - training on the small bundled demo sample. "
        "Predictions will NOT be reliable; see README for how to get the full dataset.",
        DEFAULT_DATA_DIR,
    )
    return load_labelled_csv(DEFAULT_SAMPLE), "demo sample (data/sample/news_sample.csv)"


def preprocess(df: pd.DataFrame, *, strip_leakage: bool) -> pd.DataFrame:
    """Clean text, drop empties and exact duplicates (which leak across splits)."""
    out = df.copy()
    out["clean"] = out["text"].map(lambda t: prepare(t, strip_leakage=strip_leakage))
    before = len(out)
    out = out[out["clean"].str.len() > 0].drop_duplicates(subset="clean")
    logger.info("Preprocessed %d rows (%d dropped as empty/duplicate)", len(out), before - len(out))
    if out["label"].nunique() < 2:
        raise ValueError("Dataset must contain both fake and real examples")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_vectorizer(n_docs: int, cfg: TrainConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=(1, cfg.ngram_max),
        stop_words="english",
        sublinear_tf=True,
        # Ignore ultra-rare terms on real datasets; keep everything on tiny ones.
        min_df=2 if n_docs >= 1000 else 1,
        max_df=0.9,
    )


def build_model(key: str, cfg: TrainConfig, n_train: int) -> ClassifierMixin:
    if key == "rf":
        return RandomForestClassifier(
            n_estimators=cfg.rf_estimators,
            n_jobs=-1,
            random_state=cfg.random_state,
            class_weight="balanced",
        )
    if key == "lr":
        return LogisticRegression(
            max_iter=2000, C=4.0, class_weight="balanced", random_state=cfg.random_state
        )
    if key == "svm":
        # LinearSVC has no predict_proba; calibration gives us real probabilities.
        folds = 3 if n_train >= 30 else 2
        return CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced", random_state=cfg.random_state),
            cv=folds,
            method="sigmoid",
        )
    raise ValueError(f"Unknown model key: {key!r}")


def evaluate(model: ClassifierMixin, x_test, y_test) -> dict[str, Any]:
    """Held-out metrics. The positive class for precision/recall is FAKE."""
    y_pred = model.predict(x_test)
    proba_fake = model.predict_proba(x_test)[:, list(model.classes_).index(FAKE)]
    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=FAKE, zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label=FAKE, zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label=FAKE, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[FAKE, REAL]).tolist(),
    }
    try:
        metrics["roc_auc"] = roc_auc_score((np.asarray(y_test) == FAKE).astype(int), proba_fake)
    except ValueError:  # only one class present in y_test (tiny datasets)
        metrics["roc_auc"] = None
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in metrics.items()}


def train(
    df: pd.DataFrame,
    model_dir: Path,
    cfg: TrainConfig | None = None,
    dataset_name: str = "custom",
) -> dict[str, Any]:
    """Train every configured model, save artifacts and return the metrics report."""
    cfg = cfg or TrainConfig()
    model_dir.mkdir(parents=True, exist_ok=True)
    data = preprocess(df, strip_leakage=cfg.strip_leakage)

    stratify = data["label"] if data["label"].value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        data["clean"],
        data["label"],
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify,
    )

    vectorizer = build_vectorizer(len(data), cfg)
    t0 = time.perf_counter()
    xv_train = vectorizer.fit_transform(x_train)
    xv_test = vectorizer.transform(x_test)
    logger.info(
        "TF-IDF fitted: %d features in %.1fs", len(vectorizer.vocabulary_), time.perf_counter() - t0
    )
    joblib.dump(vectorizer, model_dir / "vectorizer.joblib", compress=3)

    report: dict[str, Any] = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "dataset": {
            "name": dataset_name,
            "rows": len(data),
            "fake": int((data["label"] == FAKE).sum()),
            "real": int((data["label"] == REAL).sum()),
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "strip_leakage": cfg.strip_leakage,
        },
        "vectorizer": {"features": len(vectorizer.vocabulary_), "ngram_range": [1, cfg.ngram_max]},
        "models": {},
    }

    for key in cfg.models:
        logger.info("Training %s ...", MODEL_NAMES[key])
        model = build_model(key, cfg, len(x_train))
        t0 = time.perf_counter()
        model.fit(xv_train, y_train)
        train_seconds = time.perf_counter() - t0
        metrics = evaluate(model, xv_test, y_test)
        metrics.update(name=MODEL_NAMES[key], train_seconds=round(train_seconds, 2))
        joblib.dump(model, model_dir / f"{key}.joblib", compress=3)
        report["models"][key] = metrics
        logger.info(
            "%-20s accuracy=%.4f f1=%.4f (%.1fs)",
            MODEL_NAMES[key],
            metrics["accuracy"],
            metrics["f1"],
            train_seconds,
        )

    best = max(report["models"], key=lambda k: report["models"][k]["f1"])
    report["best_model"] = best
    (model_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    logger.info("Saved artifacts to %s (best model: %s)", model_dir, MODEL_NAMES[best])
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--fake", help="Path to Fake.csv (ISOT format)")
    p.add_argument("--true", help="Path to True.csv (ISOT format)")
    p.add_argument("--csv", help="Path to a single CSV with text,label columns")
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Output directory")
    p.add_argument("--models", default="rf,lr,svm", help="Comma-separated subset of rf,lr,svm")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--max-features", type=int, default=50_000)
    p.add_argument("--rf-estimators", type=int, default=200)
    p.add_argument(
        "--keep-leakage",
        action="store_true",
        help="Do NOT strip Reuters datelines (reproduces the inflated notebook accuracy)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    args = parse_args(argv)
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    unknown = set(models) - set(MODEL_NAMES)
    if unknown:
        logger.error("Unknown model(s): %s", ", ".join(sorted(unknown)))
        return 2
    cfg = TrainConfig(
        test_size=args.test_size,
        max_features=args.max_features,
        rf_estimators=args.rf_estimators,
        strip_leakage=not args.keep_leakage,
        models=models,
    )
    try:
        df, name = resolve_dataset(args)
        train(df, Path(args.model_dir), cfg, dataset_name=name)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Training failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
