"""API-key authentication via the ``X-API-Key`` header."""

from __future__ import annotations

import hashlib
import hmac
import logging

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from backend.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"
_api_key_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False, description="API key")


def key_fingerprint(key: str) -> str:
    """Short, non-reversible identifier for a key (safe to log / store)."""
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def require_api_key(
    request: Request,
    api_key: str | None = Security(_api_key_scheme),
    settings: Settings = Depends(get_settings),
) -> str | None:
    """Validate the API key. Returns the key fingerprint (or ``None`` if auth is off).

    Uses constant-time comparison so response timing does not leak key bytes.
    """
    if not settings.auth_enabled:
        return None
    if not api_key or not any(hmac.compare_digest(api_key, k) for k in settings.api_key_set):
        logger.warning("Rejected request with missing/invalid API key from %s", _client_ip(request))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )
    fp = key_fingerprint(api_key)
    request.state.api_key_id = fp
    return fp


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"
