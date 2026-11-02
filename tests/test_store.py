import csv
import io

import pytest

from backend.services.prediction_store import PredictionStore


def _record(store, **kw):
    defaults = {
        "model": "rf",
        "label": "Fake News",
        "is_fake": True,
        "confidence": 91.234,
        "source": "text",
        "text": "some article text",
    }
    return store.record(**{**defaults, **kw})


def test_record_and_stats(tmp_path):
    store = PredictionStore(tmp_path / "p.db")
    assert _record(store) == 1
    _record(store, label="Real News", is_fake=False, model="lr", confidence=60, source="url")
    s = store.stats()
    assert s["total_predictions"] == 2
    assert s["last_24h"] == 2
    assert s["by_label"] == {"Fake News": 1, "Real News": 1}
    assert s["by_model"] == {"lr": 1, "rf": 1}
    assert s["by_source"] == {"text": 1, "url": 1}
    assert s["average_confidence"] == round((91.23 + 60) / 2, 2)
    store.close()


@pytest.fixture
def store():
    s = PredictionStore(":memory:")
    yield s
    s.close()


def test_empty_stats(store):
    s = store.stats()
    assert s["total_predictions"] == 0
    assert s["average_confidence"] is None


def test_recent_is_newest_first_and_truncates_preview(store):
    _record(store, text="first")
    _record(store, text="x" * 500)
    rows = store.recent(10)
    assert rows[0]["id"] == 2
    assert len(rows[0]["text_preview"]) == 200


def test_export_csv(store):
    _record(store, url="https://e.com/a", title="A title, with comma")
    rows = list(csv.DictReader(io.StringIO(store.export_csv())))
    assert rows[0]["title"] == "A title, with comma"
    assert rows[0]["url"] == "https://e.com/a"
    assert "text_sha256" not in rows[0]  # internal column not exported


def test_record_failure_is_swallowed():
    store = PredictionStore(":memory:")
    store.close()
    assert _record(store) is None
