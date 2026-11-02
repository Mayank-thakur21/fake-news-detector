"""Application settings loaded from environment variables / ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

ModelKey = Literal["rf", "lr", "svm"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", protected_namespaces=()
    )

    # --- general --------------------------------------------------------- #
    app_name: str = "Fake News Detection API"
    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    log_json: bool = False
    cors_origins: str = "*"

    # --- models ---------------------------------------------------------- #
    model_dir: Path = ROOT / "models"
    default_model: ModelKey = "rf"

    # --- security -------------------------------------------------------- #
    # Comma separated. Empty => auth disabled (a warning is logged at startup).
    api_keys: str = ""
    rate_limit_per_minute: int = Field(60, ge=1)
    batch_rate_limit_per_minute: int = Field(10, ge=1)

    # --- persistence ----------------------------------------------------- #
    db_path: Path = ROOT / "storage" / "predictions.db"

    # --- input limits ---------------------------------------------------- #
    min_words: int = Field(5, ge=1)
    max_text_chars: int = Field(100_000, ge=100)
    batch_max_items: int = Field(25, ge=1, le=500)

    # --- URL scraping ---------------------------------------------------- #
    scraper_timeout: float = Field(10.0, gt=0)
    scraper_max_bytes: int = Field(5_000_000, ge=10_000)
    scraper_allow_private_hosts: bool = False  # SSRF guard; only enable for local testing
    scraper_user_agent: str = "Mozilla/5.0 (compatible; FakeNewsDetector/2.0; +https://github.com/Mayank-thakur21/fake-news-detector)"

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @property
    def api_key_set(self) -> frozenset[str]:
        return frozenset(k.strip() for k in self.api_keys.split(",") if k.strip())

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key_set)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()] or ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
