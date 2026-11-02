# 📰 Fake News Detection System v2.0

![CI](https://github.com/Mayank-thakur21/fake-news-detector/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![License](https://img.shields.io/badge/license-MIT-yellow)

An advanced ML-powered system to detect fake news articles using **Random Forest**, **Logistic Regression**, and **SVM** classifiers with TF-IDF vectorization.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🔍 Single Analysis | Paste any article and get instant prediction |
| 🔗 URL Scraping | Enter a URL — we fetch and analyze it |
| 📦 Batch Analysis | Analyze up to 20 articles at once |
| ⚖️ Model Comparison | Compare results across 3 ML models |
| 📊 Feature Breakdown | See which words drove the prediction |
| 📜 History Export | Download prediction history as CSV |
| 📈 Stats Dashboard | Track total predictions and model accuracy |
| 🐳 Docker Support | One-command deploy with docker-compose |

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
git clone https://github.com/Mayank-thakur21/fake-news-detector
cd fake-news-detector
cp .env.example .env
docker-compose up --build
```
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000/docs

### Option 2: Manual
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app:app --reload

# Frontend (new terminal)
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 🛠️ Tech Stack

- **Backend:** FastAPI, scikit-learn, joblib, BeautifulSoup4
- **Frontend:** Streamlit, pandas
- **Models:** Random Forest, Logistic Regression, SVM
- **Vectorizer:** TF-IDF
- **CI/CD:** GitHub Actions

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Single article prediction |
| POST | `/predict/batch` | Batch prediction (max 20) |
| POST | `/predict/url` | Predict from URL |
| GET | `/stats` | Usage statistics |
| GET | `/models` | List available models |

### Example
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"news": "Your article text here", "model": "random_forest"}'
```

### Response
```json
{
  "prediction": "Real News",
  "confidence": 94.2,
  "model_used": "random_forest",
  "word_count": 142,
  "top_features": {"climate": 0.42, "research": 0.38}
}
```

---

## 🧪 Running Tests
```bash
cd backend
pip install pytest httpx
pytest tests/ -v
```

---

## 🏗️ Project Structure
```
fake-news-detector/
├── backend/
│   ├── app.py              # FastAPI app
│   ├── predictor.py        # ML inference engine
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── models/             # Trained model files (.pkl)
│   └── tests/
│       └── test_api.py
├── frontend/
│   ├── streamlit_app.py    # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
├── .env.example
├── docker-compose.yml
└── README.md
```

---

## 📄 License
MIT © Mayank Thakur
