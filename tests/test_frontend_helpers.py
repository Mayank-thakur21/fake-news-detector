"""Tests for the framework-free parts of the Streamlit frontend."""

from __future__ import annotations

import io

import pandas as pd
import pytest
import requests

from api_client import ApiClient, ApiError
from history import make_entry, parse_batch_csv, parse_batch_text, to_csv_bytes
from theme import DARK, LIGHT, css, palette

RESULT = {
    "prediction": "Fake News",
    "confidence": 88.5,
    "model": "rf",
    "model_name": "Random Forest",
}


# --------------------------------------------------------------------------- #
# history.py
# --------------------------------------------------------------------------- #
def test_make_entry_truncates_and_flattens():
    e = make_entry(RESULT, input_preview="line1\nline2 " + "x" * 200, source="Text")
    assert "\n" not in e["Input"]
    assert e["Input"].endswith("…") and len(e["Input"]) == 121
    assert e["Model"] == "Random Forest" and e["Confidence (%)"] == 88.5


def test_csv_export_roundtrip():
    rows = [make_entry(RESULT, input_preview="héllo, wörld", source="URL")]
    data = to_csv_bytes(rows)
    assert data.startswith(b"\xef\xbb\xbf")  # BOM for Excel
    df = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
    assert df.loc[0, "Input"] == "héllo, wörld"
    assert df.columns[0] == "Time"


def test_csv_export_empty_has_header():
    assert to_csv_bytes([]).decode("utf-8-sig").startswith("Time,Source,Input")


def test_parse_batch_text():
    raw = "First article\nline two\n---\n\n---\nhttps://example.com/a\n---\nThird"
    assert parse_batch_text(raw) == [
        {"text": "First article\nline two"},
        {"url": "https://example.com/a"},
        {"text": "Third"},
    ]


def test_parse_batch_csv():
    df = pd.DataFrame({"Text": ["a", None, ""], "URL": [None, "https://e.com", None]})
    assert parse_batch_csv(df) == [{"text": "a"}, {"url": "https://e.com"}]


def test_parse_batch_csv_requires_columns():
    with pytest.raises(ValueError):
        parse_batch_csv(pd.DataFrame({"body": ["x"]}))


# --------------------------------------------------------------------------- #
# theme.py
# --------------------------------------------------------------------------- #
def test_palettes():
    assert palette("dark") is DARK and palette("light") is LIGHT and palette("other") is LIGHT
    assert DARK.bg in css(DARK) and LIGHT.bg in css(LIGHT)
    assert css(DARK).strip().startswith("<style>")


# --------------------------------------------------------------------------- #
# api_client.py
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self.ok = status < 400
        self._payload = payload
        self.text = text
        self.content = text.encode()

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _client(monkeypatch, resp_or_exc, captured=None):
    c = ApiClient("http://api:8000/", api_key="k")

    def fake_request(method, url, **kw):
        if captured is not None:
            captured.update(method=method, url=url, **kw)
        if isinstance(resp_or_exc, Exception):
            raise resp_or_exc
        return resp_or_exc

    monkeypatch.setattr(c.session, "request", fake_request)
    return c


def test_client_sends_key_and_strips_none(monkeypatch):
    cap: dict = {}
    c = _client(monkeypatch, _Resp(200, {"ok": True}), cap)
    assert c.session.headers["X-API-Key"] == "k"
    c.predict(text="hello", model=None)
    assert cap["url"] == "http://api:8000/predict"
    assert cap["json"] == {"text": "hello", "explain": True, "top_k": 10}


def test_client_error_envelope(monkeypatch):
    c = _client(monkeypatch, _Resp(429, {"error": {"code": "x", "message": "Slow down"}}))
    with pytest.raises(ApiError) as e:
        c.stats()
    assert e.value.message == "Slow down" and e.value.status == 429


def test_client_fastapi_detail(monkeypatch):
    c = _client(monkeypatch, _Resp(401, {"detail": "Missing or invalid API key"}))
    with pytest.raises(ApiError, match="invalid API key"):
        c.models()


def test_client_non_json_error(monkeypatch):
    c = _client(monkeypatch, _Resp(502, None, "Bad gateway"))
    with pytest.raises(ApiError, match="HTTP 502"):
        c.health()


def test_client_connection_error(monkeypatch):
    c = _client(monkeypatch, requests.ConnectionError("refused"))
    with pytest.raises(ApiError, match="Cannot reach"):
        c.health()


def test_client_timeout(monkeypatch):
    c = _client(monkeypatch, requests.Timeout())
    with pytest.raises(ApiError, match="too long"):
        c.health()
