# AI-Powered Fake News Detection System

---

## 1. Introduction

**Problem:** Misinformation spreads rapidly online. Readers struggle to distinguish real news from fake.

**Solution:** An ML-powered web app that analyzes news articles and predicts authenticity — **Fake** or **Real**.

**Goal:** Provide a fast, accessible tool for anyone to verify news credibility.

---

## 2. Objective

- Build a complete Fake News Detection system
- Use a trained ML model for real-time prediction
- Simple web interface for end users
- REST API for integration with other platforms

---

## 3. Dataset

| Source | Articles | Label |
|--------|----------|-------|
| True.csv | ~21,000 | Real News (1) |
| Fake.csv | ~23,000 | Fake News (0) |

**Total:** ~44,000 labeled news articles from Kaggle.

---

## 4. Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Backend | FastAPI + Uvicorn |
| ML Model | Random Forest Classifier |
| Text Processing | TF-IDF Vectorizer |
| Model Serialization | joblib |
| Data Processing | pandas, numpy |
| Language | Python 3 |

---

## 5. System Architecture

```
┌─────────────┐     HTTP      ┌──────────────┐     Load      ┌──────────────────┐
│  Streamlit  │ ──────────▶   │   FastAPI    │ ──────────▶   │  ML Model +      │
│  Frontend   │ ◀──────────   │   Backend    │ ◀──────────   │  TF-IDF Vector   │
│  (User UI)  │    JSON       │   (API)      │    Return     └──────────────────┘
└─────────────┘               └──────────────┘
```

---

## 6. Workflow

```
User Input → Clean Text → TF-IDF Vectorize → Predict (RFC) → Output Result
                                                                    │
                                                          ┌─────────┴──────────┐
                                                          ▼                    ▼
                                                     ✅ Real News         ❌ Fake News
                                                     + Confidence %       + Confidence %
```

**Steps:**
1. User pastes news article in the web UI
2. Text is cleaned and transformed using TF-IDF vectorizer
3. Trained Random Forest model makes prediction
4. Result (Fake/Real) with confidence score displayed
5. Prediction saved in session history

---

## 7. Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **99.65%** |
| Algorithm | Random Forest Classifier |
| Estimators | 100 |
| Max Features | 5,000 (TF-IDF) |

---

## 8. Features

- **Real-time prediction** – Analyze any news article instantly
- **Confidence score** – See how certain the model is
- **Session history** – Track all predictions made
- **Clean UI** – Modern Streamlit interface
- **REST API** – Backend can be used independently
- **Error handling** – Graceful handling of empty/invalid input

---

## 9. API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Analyze news article |

**Request:**
```json
{ "news": "Full article text here..." }
```

**Response:**
```json
{
  "prediction": "Fake News",
  "confidence": 96.50
}
```

---

## 10. Applications

- **Social media platforms** – Flag potentially fake content
- **News aggregators** – Verify sources before display
- **Journalism** – Fact-checking assistance
- **Education** – Teach media literacy
- **Browser extensions** – Real-time article verification

---

## 11. Future Improvements

- Deep learning models (LSTM, BERT) for better accuracy
- Multi-language support
- Source credibility scoring
- Explainable AI – Show why a prediction was made
- Browser extension integration
- Mobile app version

---

## 12. Conclusion

Successfully built a complete Fake News Detection system with:
- ✅ 99.65% accurate ML model
- ✅ FastAPI backend with clean REST API
- ✅ Streamlit frontend with modern UI
- ✅ End-to-end prediction pipeline
- ✅ Ready for real-world use

---
