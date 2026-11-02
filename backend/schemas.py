"""Pydantic request / response models (these also drive the OpenAPI docs)."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, HttpUrl, model_validator

ModelKey = Literal["rf", "lr", "svm"]


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class ArticleInput(BaseModel):
    """Exactly one of ``text`` or ``url`` must be provided."""

    text: str | None = Field(
        None,
        # ``news`` keeps backwards compatibility with the v1 API payload.
        validation_alias=AliasChoices("text", "news"),
        description="Full article text",
        examples=["Scientists confirm that drinking lemon water cures all diseases..."],
    )
    url: HttpUrl | None = Field(None, description="URL of a news article to fetch and analyse")

    @model_validator(mode="after")
    def _one_of(self) -> ArticleInput:
        has_text = bool(self.text and self.text.strip())
        if has_text == bool(self.url):
            raise ValueError("Provide exactly one of 'text' or 'url'")
        return self


class PredictRequest(ArticleInput):
    model: ModelKey | None = Field(None, description="rf | lr | svm (default: server setting)")
    explain: bool = Field(True, description="Include per-feature confidence breakdown")
    top_k: int = Field(10, ge=1, le=30, description="Number of features per direction")


class CompareRequest(ArticleInput):
    explain: bool = False
    top_k: int = Field(5, ge=1, le=30)


class BatchItem(ArticleInput):
    id: str | None = Field(None, max_length=100, description="Optional client-side identifier")


class BatchRequest(BaseModel):
    items: list[BatchItem] = Field(..., min_length=1)
    model: ModelKey | None = None
    explain: bool = False
    top_k: int = Field(5, ge=1, le=30)


class ScrapeRequest(BaseModel):
    url: HttpUrl


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class FeatureContribution(BaseModel):
    term: str
    weight: float = Field(..., description="Signed contribution: <0 pushes fake, >0 pushes real")
    share: float = Field(..., description="% of the total evidence behind this prediction")
    tfidf: float
    present: bool = Field(..., description="False if the term's absence drove the decision")


class Explanation(BaseModel):
    method: str
    base_value: float | None
    toward_fake: list[FeatureContribution]
    toward_real: list[FeatureContribution]


class SourceInfo(BaseModel):
    type: Literal["text", "url"]
    url: str | None = None
    final_url: str | None = None
    title: str | None = None
    site_name: str | None = None
    published_at: str | None = None
    word_count: int


class Probabilities(BaseModel):
    fake: float
    real: float


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: Literal["Fake News", "Real News"]
    is_fake: bool
    confidence: float = Field(..., description="Probability of the predicted class, 0-100")
    probabilities: Probabilities
    model: ModelKey
    model_name: str
    explanation: Explanation | None = None
    source: SourceInfo
    processing_ms: float
    prediction_id: int | None = None


class BatchResult(BaseModel):
    index: int
    id: str | None = None
    ok: bool
    result: PredictionResponse | None = None
    error: str | None = None


class BatchSummary(BaseModel):
    total: int
    succeeded: int
    failed: int
    fake: int
    real: int
    average_confidence: float | None


class BatchResponse(BaseModel):
    results: list[BatchResult]
    summary: BatchSummary


class ModelVerdict(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: ModelKey
    model_name: str
    prediction: Literal["Fake News", "Real News"]
    confidence: float
    probabilities: Probabilities
    test_accuracy: float | None = None
    test_f1: float | None = None
    explanation: Explanation | None = None


class CompareResponse(BaseModel):
    results: list[ModelVerdict]
    consensus: Literal["Fake News", "Real News"]
    agreement: float = Field(..., description="% of models agreeing with the consensus")
    mean_fake_probability: float
    source: SourceInfo
    processing_ms: float


class ScrapeResponse(BaseModel):
    url: str
    final_url: str
    title: str | None
    site_name: str | None
    published_at: str | None
    word_count: int
    text: str


class ModelInfo(BaseModel):
    key: ModelKey
    name: str
    loaded: bool
    is_default: bool
    metrics: dict


class ModelsResponse(BaseModel):
    default_model: str | None
    best_model: str | None
    models: list[ModelInfo]
    trained_at: str | None
    dataset: dict | None
    vectorizer: dict | None


class StatsResponse(BaseModel):
    total_predictions: int
    last_24h: int
    average_confidence: float | None
    by_label: dict[str, int]
    by_model: dict[str, int]
    by_source: dict[str, int]
    model_accuracy: dict[str, float | None]
    models: ModelsResponse
    uptime_seconds: float
    version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models_loaded: list[str]
    detail: str | None = None
    version: str


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody
