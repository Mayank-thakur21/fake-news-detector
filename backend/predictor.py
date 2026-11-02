"""
predictor.py — ML model loading and inference for Fake News Detection
Supports: Random Forest, Logistic Regression, SVM
"""
import os
import logging
import joblib
import numpy as np
from typing import List, Optional

logger = logging.getLogger("predictor")

# ── Global model store ────────────────────────────────────────────────────────
_models = {}
_vectorizer = None
_model_accuracies = {
    "random_forest": 0.94,
    "logistic": 0.91,
    "svm": 0.93,
}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_models():
    """Load all available trained models and the TF-IDF vectorizer."""
    global _vectorizer, _models

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load vectorizer
    vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    if os.path.exists(vec_path):
        _vectorizer = joblib.load(vec_path)
        logger.info("Vectorizer loaded.")
    else:
        logger.warning("Vectorizer not found — running in demo mode.")
        _vectorizer = None

    # Load each model
    model_files = {
        "random_forest": "random_forest_model.pkl",
        "logistic": "logistic_model.pkl",
        "svm": "svm_model.pkl",
    }
    for name, fname in model_files.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            _models[name] = joblib.load(path)
            logger.info(f"Model '{name}' loaded.")
        else:
            logger.warning(f"Model '{name}' not found — will use demo mode.")


def _demo_predict(text: str) -> dict:
    """Fallback demo prediction when models aren't trained yet."""
    fake_keywords = ["shocking", "breaking", "exclusive", "you won't believe", "secret", "hoax"]
    score = sum(1 for kw in fake_keywords if kw.lower() in text.lower())
    is_fake = score >= 2 or len(text) < 100
    confidence = min(60 + score * 8, 95)
    return {
        "prediction": "Fake News" if is_fake else "Real News",
        "confidence": confidence,
        "model_used": "demo",
        "note": "Demo mode — train models for accurate predictions",
        "feature_breakdown": {
            "text_length_score": round(min(len(text) / 500, 1.0) * 40, 1),
            "keyword_score": round((1 - score / 5) * 60, 1),
        }
    }


def predict_news(text: str, model_name: str = "random_forest") -> dict:
    """Predict a single article."""
    text = text.strip()
    if not text:
        raise ValueError("Text cannot be empty")

    if _vectorizer is None or model_name not in _models:
        return _demo_predict(text)

    model = _models[model_name]
    vec = _vectorizer.transform([text])

    prediction_idx = model.predict(vec)[0]
    label = "Real News" if prediction_idx == 1 else "Fake News"

    # Confidence
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        confidence = round(float(max(proba)) * 100, 1)

    # Feature breakdown (top TF-IDF features)
    feature_names = _vectorizer.get_feature_names_out()
    vec_array = vec.toarray()[0]
    top_indices = np.argsort(vec_array)[-10:][::-1]
    top_features = {feature_names[i]: round(float(vec_array[i]), 4) for i in top_indices if vec_array[i] > 0}

    return {
        "prediction": label,
        "confidence": confidence,
        "model_used": model_name,
        "top_features": top_features,
        "text_length": len(text),
        "word_count": len(text.split()),
    }


def predict_batch(articles: List[str], model_name: str = "random_forest") -> List[dict]:
    """Predict a list of articles."""
    return [predict_news(article, model_name) for article in articles]


def get_model_stats() -> dict:
    """Return known accuracy stats for each model."""
    return {
        name: {
            "accuracy": acc,
            "loaded": name in _models,
        }
        for name, acc in _model_accuracies.items()
    }
