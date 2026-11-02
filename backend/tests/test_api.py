"""Unit tests for Fake News Detection API"""
import pytest
from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_real_news():
    r = client.post("/predict", json={"news": "Scientists at NASA have confirmed a new discovery about the formation of black holes using the James Webb Space Telescope. The findings were published in Nature journal and peer-reviewed by 12 independent researchers."})
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert data["prediction"] in ["Real News", "Fake News"]


def test_predict_fake_news():
    r = client.post("/predict", json={"news": "SHOCKING: You won't believe this secret cure doctors don't want you to know! Share before it gets deleted!!!"})
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data


def test_predict_empty():
    r = client.post("/predict", json={"news": ""})
    assert r.status_code == 400


def test_predict_batch():
    r = client.post("/predict/batch", json={
        "articles": [
            "Scientists publish new research on climate change.",
            "SHOCKING secret the government is hiding from you!!!"
        ]
    })
    assert r.status_code == 200
    assert "results" in r.json()
    assert len(r.json()["results"]) == 2


def test_predict_batch_too_many():
    r = client.post("/predict/batch", json={"articles": ["text"] * 25})
    assert r.status_code == 400


def test_models_list():
    r = client.get("/models")
    assert r.status_code == 200
    assert "models" in r.json()
    assert len(r.json()["models"]) == 3


def test_stats():
    r = client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_predictions" in data
    assert "model_accuracies" in data
