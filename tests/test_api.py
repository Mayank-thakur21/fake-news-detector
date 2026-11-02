from __future__ import annotations

import csv
import io

import pytest

from backend.core.config import Settings
from backend.services.scraper import ScrapedArticle
from tests.conftest import API_KEY, FAKE_TEXT, REAL_TEXT


# --------------------------------------------------------------------------- #
# Public endpoints & auth
# --------------------------------------------------------------------------- #
def test_root_and_health_are_public(make_client, settings):
    c = make_client(settings)
    assert c.get("/").json()["auth_required"] is True
    h = c.get("/health").json()
    assert h["status"] == "ok"
    assert set(h["models_loaded"]) == {"rf", "lr", "svm"}


@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/predict"),
        ("post", "/predict/batch"),
        ("post", "/compare"),
        ("get", "/stats"),
        ("get", "/history"),
        ("get", "/history/export"),
        ("get", "/models"),
    ],
)
def test_protected_endpoints_require_key(make_client, settings, method, path):
    c = make_client(settings)
    resp = getattr(c, method)(path, **({"json": {"text": REAL_TEXT}} if method == "post" else {}))
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_wrong_key_rejected(make_client, settings):
    c = make_client(settings)
    r = c.post("/predict", json={"text": REAL_TEXT}, headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_auth_disabled_when_no_keys(make_client, settings):
    c = make_client(settings.model_copy(update={"api_keys": ""}))
    assert c.post("/predict", json={"text": REAL_TEXT}).status_code == 200


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #
def test_predict_fake_and_real(client):
    fake = client.post("/predict", json={"text": FAKE_TEXT}).json()
    real = client.post("/predict", json={"text": REAL_TEXT}).json()
    assert fake["prediction"] == "Fake News" and fake["is_fake"] is True
    assert real["prediction"] == "Real News" and real["is_fake"] is False
    assert 50 <= fake["confidence"] <= 100
    assert fake["probabilities"]["fake"] + fake["probabilities"]["real"] == pytest.approx(
        1, abs=1e-3
    )
    assert fake["prediction_id"] is not None
    assert fake["source"] == {
        "type": "text",
        "url": None,
        "final_url": None,
        "title": None,
        "site_name": None,
        "published_at": None,
        "word_count": len(FAKE_TEXT.split()),
    }


def test_predict_legacy_news_field(client):
    r = client.post("/predict", json={"news": FAKE_TEXT})
    assert r.status_code == 200


@pytest.mark.parametrize("model", ["rf", "lr", "svm"])
def test_predict_with_each_model_and_explanation(client, model):
    r = client.post("/predict", json={"text": FAKE_TEXT, "model": model, "top_k": 3}).json()
    assert r["model"] == model
    exp = r["explanation"]
    assert exp is not None
    assert len(exp["toward_fake"]) <= 3
    assert exp["toward_fake"], "fake article should have evidence toward fake"


def test_predict_without_explanation(client):
    r = client.post("/predict", json={"text": FAKE_TEXT, "explain": False}).json()
    assert r["explanation"] is None


def test_predict_sets_rate_limit_headers(client):
    r = client.post("/predict", json={"text": REAL_TEXT})
    assert r.headers["X-RateLimit-Limit"] == "1000"
    assert int(r.headers["X-RateLimit-Remaining"]) == 999
    assert r.headers["X-Request-ID"]


@pytest.mark.parametrize(
    "payload,fragment",
    [
        ({}, "exactly one"),
        ({"text": "   "}, "exactly one"),
        ({"text": REAL_TEXT, "url": "https://example.com"}, "exactly one"),
        ({"text": REAL_TEXT, "model": "xgboost"}, "model"),
        ({"text": REAL_TEXT, "top_k": 0}, "top_k"),
        ({"url": "not a url"}, "url"),
    ],
)
def test_predict_validation_errors(client, payload, fragment):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    body = r.json()["error"]
    assert body["code"] == "validation_error"
    assert fragment.lower() in body["message"].lower()


def test_predict_too_short(client):
    r = client.post("/predict", json={"text": "fake? 123"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_input"


def test_predict_url_uses_scraper(client, monkeypatch):
    article = ScrapedArticle(
        url="https://news.example.com/a",
        final_url="https://news.example.com/a",
        title="Shocking secret",
        text=FAKE_TEXT * 3,
        site_name="Example",
        published_at=None,
        word_count=60,
    )
    scraper = client.app.state.analyzer.scraper
    monkeypatch.setattr(scraper, "scrape", lambda url: article)
    r = client.post("/predict", json={"url": "https://news.example.com/a"}).json()
    assert r["source"]["type"] == "url"
    assert r["source"]["title"] == "Shocking secret"
    assert r["prediction"] == "Fake News"


def test_predict_url_ssrf_blocked(client):
    r = client.post("/predict", json={"url": "http://127.0.0.1:9999/admin"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "scrape_failed"


def test_unknown_but_valid_model_not_loaded(make_client, settings, tmp_path):
    # Copy only the LR model so 'rf' is a valid key that isn't loaded.
    import shutil

    d = tmp_path / "partial"
    d.mkdir()
    for f in ("vectorizer.joblib", "lr.joblib", "metrics.json"):
        shutil.copy(settings.model_dir / f, d / f)
    c = make_client(settings.model_copy(update={"model_dir": d, "api_keys": ""}))
    assert c.post("/predict", json={"text": REAL_TEXT, "model": "rf"}).status_code == 404
    # Default 'rf' falls back to an available model.
    assert c.post("/predict", json={"text": REAL_TEXT}).json()["model"] == "lr"


def test_models_missing_returns_503(make_client, settings, tmp_path):
    c = make_client(settings.model_copy(update={"model_dir": tmp_path / "empty", "api_keys": ""}))
    assert c.get("/health").json()["status"] == "degraded"
    r = c.post("/predict", json={"text": REAL_TEXT})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "model_not_ready"
    assert c.post("/predict/batch", json={"items": [{"text": REAL_TEXT}]}).status_code == 503


# --------------------------------------------------------------------------- #
# Batch, compare, scrape
# --------------------------------------------------------------------------- #
def test_batch_mixed_results(client):
    r = client.post(
        "/predict/batch",
        json={
            "items": [
                {"text": FAKE_TEXT, "id": "a"},
                {"text": "too short"},
                {"text": REAL_TEXT, "id": "c"},
            ],
            "model": "lr",
        },
    ).json()
    assert r["summary"] == {
        "total": 3,
        "succeeded": 2,
        "failed": 1,
        "fake": 1,
        "real": 1,
        "average_confidence": r["summary"]["average_confidence"],
    }
    assert r["results"][0]["id"] == "a" and r["results"][0]["ok"]
    assert r["results"][1]["ok"] is False and "too short" in r["results"][1]["error"]
    assert r["results"][2]["result"]["model"] == "lr"


def test_batch_size_limit(client):
    items = [{"text": REAL_TEXT}] * 26
    r = client.post("/predict/batch", json={"items": items})
    assert r.status_code == 422
    assert "max 25" in r.json()["error"]["message"]


def test_batch_requires_items(client):
    assert client.post("/predict/batch", json={"items": []}).status_code == 422


def test_compare(client):
    r = client.post("/compare", json={"text": FAKE_TEXT, "explain": True}).json()
    assert {v["model"] for v in r["results"]} == {"rf", "lr", "svm"}
    assert r["consensus"] == "Fake News"
    assert r["agreement"] in (pytest.approx(66.7), 100.0)
    assert all(v["test_accuracy"] is not None for v in r["results"])
    assert all(v["explanation"] for v in r["results"])


def test_scrape_endpoint(client, monkeypatch):
    article = ScrapedArticle(
        "https://e.com/a", "https://e.com/b", "T", "body " * 30, None, None, 30
    )
    monkeypatch.setattr(client.app.state.analyzer.scraper, "scrape", lambda url: article)
    r = client.post("/scrape", json={"url": "https://e.com/a"}).json()
    assert r["final_url"] == "https://e.com/b" and r["word_count"] == 30


# --------------------------------------------------------------------------- #
# Stats, history, models
# --------------------------------------------------------------------------- #
def test_stats_counts_predictions(client):
    before = client.get("/stats").json()
    assert before["total_predictions"] == 0
    client.post("/predict", json={"text": FAKE_TEXT})
    client.post("/predict", json={"text": REAL_TEXT, "model": "svm"})
    s = client.get("/stats").json()
    assert s["total_predictions"] == 2
    assert s["by_label"] == {"Fake News": 1, "Real News": 1}
    assert s["by_model"] == {"rf": 1, "svm": 1}
    assert set(s["model_accuracy"]) == {"rf", "lr", "svm"}
    assert all(0 <= v <= 1 for v in s["model_accuracy"].values())
    assert s["models"]["dataset"]["name"] == "synthetic"
    assert s["uptime_seconds"] >= 0


def test_models_endpoint(client):
    r = client.get("/models").json()
    assert r["default_model"] == "rf"
    assert [m["key"] for m in r["models"]] == ["rf", "lr", "svm"]
    assert all(m["loaded"] for m in r["models"])


def test_history_and_csv_export(client):
    client.post("/predict", json={"text": FAKE_TEXT})
    hist = client.get("/history?limit=5").json()
    assert len(hist) == 1 and hist[0]["label"] == "Fake News"
    r = client.get("/history/export")
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert rows[0]["model"] == "rf"


def test_admin_reload(client):
    r = client.post("/admin/reload").json()
    assert r["ready"] is True


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
def test_rate_limit_returns_429(make_client, settings: Settings):
    c = make_client(settings.model_copy(update={"rate_limit_per_minute": 2}))
    c.headers["X-API-Key"] = API_KEY
    assert c.post("/predict", json={"text": REAL_TEXT}).status_code == 200
    assert c.post("/predict", json={"text": REAL_TEXT}).status_code == 200
    r = c.post("/predict", json={"text": REAL_TEXT})
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1
    assert r.headers["X-RateLimit-Remaining"] == "0"
    assert r.json()["error"]["code"] == "rate_limited"


def test_rate_limit_is_per_scope(make_client, settings: Settings):
    c = make_client(settings.model_copy(update={"rate_limit_per_minute": 1}))
    c.headers["X-API-Key"] = API_KEY
    assert c.post("/predict", json={"text": REAL_TEXT}).status_code == 200
    # batch has its own bucket
    assert c.post("/predict/batch", json={"items": [{"text": REAL_TEXT}]}).status_code == 200
