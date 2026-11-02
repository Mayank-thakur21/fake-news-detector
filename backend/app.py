from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import time
import logging
import os
import json
from datetime import datetime
from uuid import uuid4

from predictor import load_models, predict_news, predict_batch, get_model_stats

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("FakeNewsAPI")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fake News Detection API",
    description="Advanced ML-powered fake news detection with multi-model support",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Simple in-memory stats tracker ───────────────────────────────────────────
prediction_log: List[dict] = []
API_KEY = os.getenv("API_KEY", "dev-key-123")  # Set in .env for production


# ── Auth dependency ───────────────────────────────────────────────────────────
def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if os.getenv("REQUIRE_API_KEY", "false").lower() == "true":
        if x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


# ── Request/Response models ───────────────────────────────────────────────────
class NewsInput(BaseModel):
    news: str
    model: Optional[str] = "random_forest"  # random_forest | logistic | svm

class BatchNewsInput(BaseModel):
    articles: List[str]
    model: Optional[str] = "random_forest"

class URLInput(BaseModel):
    url: str
    model: Optional[str] = "random_forest"


# ── Middleware: request timing ────────────────────────────────────────────────
@app.middleware("http")
async def add_timing(request, call_next):
    start = time.time()
    rid = str(uuid4())[:8]
    logger.info(f"[{rid}] {request.method} {request.url.path}")
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000, 2)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time"] = f"{elapsed}ms"
    logger.info(f"[{rid}] {response.status_code} in {elapsed}ms")
    return response


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    logger.info("Loading ML models...")
    load_models()
    logger.info("Models loaded successfully.")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "message": "Fake News Detection API v2.0",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
    }

@app.post("/predict", tags=["Prediction"])
def predict(input_data: NewsInput, _key=Depends(verify_api_key)):
    """Predict if a single news article is real or fake."""
    if not input_data.news.strip():
        raise HTTPException(status_code=400, detail="News text cannot be empty")
    try:
        result = predict_news(input_data.news, model_name=input_data.model)
        # Log prediction
        prediction_log.append({
            "id": str(uuid4())[:8],
            "timestamp": datetime.utcnow().isoformat(),
            "model": input_data.model,
            "prediction": result["prediction"],
            "confidence": result.get("confidence"),
            "text_length": len(input_data.news),
        })
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch_endpoint(input_data: BatchNewsInput, _key=Depends(verify_api_key)):
    """Predict multiple articles at once (max 20)."""
    if not input_data.articles:
        raise HTTPException(status_code=400, detail="No articles provided")
    if len(input_data.articles) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 articles per batch")
    try:
        results = predict_batch(input_data.articles, model_name=input_data.model)
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/url", tags=["Prediction"])
def predict_from_url(input_data: URLInput, _key=Depends(verify_api_key)):
    """Scrape a news URL and predict if it's real or fake."""
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(input_data.url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Extract article text
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs[:30])
        if len(text.strip()) < 50:
            raise HTTPException(status_code=422, detail="Could not extract enough text from URL")
        result = predict_news(text, model_name=input_data.model)
        result["scraped_text_preview"] = text[:300] + "..."
        result["source_url"] = input_data.url
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process URL: {str(e)}")

@app.get("/stats", tags=["Analytics"])
def get_stats(_key=Depends(verify_api_key)):
    """Get API usage stats and model performance metrics."""
    model_stats = get_model_stats()
    total = len(prediction_log)
    fake_count = sum(1 for p in prediction_log if p["prediction"] == "Fake News")
    real_count = total - fake_count
    return {
        "total_predictions": total,
        "fake_count": fake_count,
        "real_count": real_count,
        "fake_percentage": round(fake_count / total * 100, 1) if total > 0 else 0,
        "model_accuracies": model_stats,
        "recent_predictions": prediction_log[-10:][::-1],
    }

@app.get("/models", tags=["Analytics"])
def list_models():
    """List available prediction models."""
    return {
        "models": [
            {"id": "random_forest", "name": "Random Forest", "description": "Ensemble tree-based classifier"},
            {"id": "logistic", "name": "Logistic Regression", "description": "Fast linear classifier"},
            {"id": "svm", "name": "Support Vector Machine", "description": "High-accuracy kernel-based classifier"},
        ]
    }
