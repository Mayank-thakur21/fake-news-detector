"""Thin, typed HTTP client for the Fake News Detection API."""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """Raised for any failed API call; ``message`` is safe to show to users."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.message = message
        self.status = status


class ApiClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers["X-API-Key"] = api_key

    # ------------------------------------------------------------------ #
    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
        except requests.ConnectionError as exc:
            raise ApiError(f"Cannot reach the backend at {self.base_url}. Is it running?") from exc
        except requests.Timeout as exc:
            raise ApiError("The backend took too long to respond. Try again.") from exc
        except requests.RequestException as exc:
            raise ApiError(f"Request failed: {exc}") from exc

        if resp.ok:
            return resp
        raise ApiError(self._error_message(resp), resp.status_code)

    @staticmethod
    def _error_message(resp: requests.Response) -> str:
        try:
            body = resp.json()
        except ValueError:
            return f"HTTP {resp.status_code}: {resp.text[:200]}"
        if isinstance(body, dict):
            if isinstance(body.get("error"), dict):
                return body["error"].get("message", f"HTTP {resp.status_code}")
            if "detail" in body:
                return str(body["detail"])
        return f"HTTP {resp.status_code}"

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        return self._request(method, path, **kwargs).json()

    # ------------------------------------------------------------------ #
    def health(self) -> dict[str, Any]:
        return self._json("GET", "/health")

    def stats(self) -> dict[str, Any]:
        return self._json("GET", "/stats")

    def models(self) -> dict[str, Any]:
        return self._json("GET", "/models")

    def predict(
        self,
        *,
        text: str | None = None,
        url: str | None = None,
        model: str | None = None,
        explain: bool = True,
        top_k: int = 10,
    ) -> dict[str, Any]:
        payload = {"text": text, "url": url, "model": model, "explain": explain, "top_k": top_k}
        return self._json(
            "POST", "/predict", json={k: v for k, v in payload.items() if v is not None}
        )

    def predict_batch(
        self, items: list[dict[str, str]], model: str | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"items": items, "explain": False}
        if model:
            payload["model"] = model
        return self._json("POST", "/predict/batch", json=payload)

    def compare(
        self,
        *,
        text: str | None = None,
        url: str | None = None,
        explain: bool = True,
        top_k: int = 5,
    ) -> dict[str, Any]:
        payload = {"text": text, "url": url, "explain": explain, "top_k": top_k}
        return self._json(
            "POST", "/compare", json={k: v for k, v in payload.items() if v is not None}
        )

    def history(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._json("GET", "/history", params={"limit": limit})

    def history_csv(self) -> bytes:
        return self._request("GET", "/history/export").content
