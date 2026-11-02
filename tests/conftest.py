"""Shared fixtures: a tiny synthetic corpus, trained models and a configured app."""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.core.config import Settings
from backend.core.rate_limit import limiter
from backend.main import create_app
from ml.train import TrainConfig, train

API_KEY = "test-key-123"

_FAKE_WORDS = [
    "shocking",
    "secret",
    "miracle",
    "cure",
    "exposed",
    "hidden",
    "truth",
    "banned",
    "elites",
    "conspiracy",
    "share",
    "before",
    "deleted",
    "they",
    "dont",
    "want",
    "you",
    "know",
    "insiders",
    "leaked",
    "scandal",
    "wake",
    "up",
]
_REAL_WORDS = [
    "officials",
    "said",
    "statement",
    "announced",
    "percent",
    "quarter",
    "ministry",
    "report",
    "published",
    "according",
    "data",
    "committee",
    "government",
    "spokesman",
    "analysts",
    "tuesday",
    "budget",
]
_COMMON = [
    "the",
    "a",
    "of",
    "and",
    "in",
    "to",
    "for",
    "on",
    "with",
    "at",
    "by",
    "this",
    "that",
    "is",
    "was",
    "it",
]

FAKE_TEXT = (
    "SHOCKING secret miracle cure exposed! The hidden truth the elites banned. "
    "Share before deleted, insiders leaked the scandal. Wake up!"
)
REAL_TEXT = (
    "Officials said in a statement on Tuesday that the ministry published a report showing "
    "the budget grew 3 percent according to committee data, analysts said."
)


def _doc(rng: random.Random, words: list[str], n: int = 40) -> str:
    vocab = words * 3 + _COMMON
    return " ".join(rng.choice(vocab) for _ in range(n))


def synthetic_corpus(n_per_class: int = 60, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = [{"text": _doc(rng, _FAKE_WORDS), "label": 0} for _ in range(n_per_class)]
    rows += [{"text": _doc(rng, _REAL_WORDS), "label": 1} for _ in range(n_per_class)]
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("models")
    train(synthetic_corpus(), out, TrainConfig(rf_estimators=15), dataset_name="synthetic")
    return out


@pytest.fixture
def settings(model_dir: Path, tmp_path: Path) -> Settings:
    return Settings(
        model_dir=model_dir,
        db_path=tmp_path / "predictions.db",
        api_keys=API_KEY,
        environment="test",
        rate_limit_per_minute=1000,
        batch_rate_limit_per_minute=1000,
        log_level="WARNING",
    )


@pytest.fixture(autouse=True)
def _reset_limiter() -> Iterator[None]:
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture
def make_client():
    """Factory so tests can build clients with custom settings."""
    clients: list[TestClient] = []

    def _make(settings: Settings) -> TestClient:
        c = TestClient(create_app(settings))
        c.__enter__()  # run lifespan
        clients.append(c)
        return c

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client, settings: Settings) -> TestClient:
    c = make_client(settings)
    c.headers["X-API-Key"] = API_KEY
    return c
