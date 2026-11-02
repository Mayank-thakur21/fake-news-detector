"""Dependency providers that pull shared services off ``app.state``."""

from __future__ import annotations

from fastapi import Request

from backend.services.analyzer import NewsAnalyzer
from backend.services.model_registry import ModelRegistry
from backend.services.prediction_store import PredictionStore


def get_analyzer(request: Request) -> NewsAnalyzer:
    return request.app.state.analyzer


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def get_store(request: Request) -> PredictionStore:
    return request.app.state.store
