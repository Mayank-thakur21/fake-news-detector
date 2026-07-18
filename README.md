# AI-Powered Fake News Detection System

A machine learning web application that determines whether a news article is **Fake** or **Real** with 99.65% accuracy.

## Features

- Real-time news article analysis
- Confidence score for each prediction
- Prediction history (session-based)
- Clean, modern Streamlit UI
- REST API backend (FastAPI)
- Handles empty/invalid input gracefully

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Backend | FastAPI, Uvicorn |
| ML Model | Random Forest (scikit-learn) |
| Text Processing | TF-IDF Vectorizer |
| Data | pandas, numpy |

## Project Structure

```
fake-news-detection/
├── backend/
│   ├── app.py                 # FastAPI server
│   ├── predictor.py           # Model loading & prediction
│   ├── fake_news_model.pkl    # Trained Random Forest model
│   ├── tfidf_vectorizer.pkl   # TF-IDF vectorizer
│   └── requirements.txt
├── frontend/
│   └── streamlit_app.py       # Streamlit UI
├── requirements.txt           # All dependencies
├── demo_news.md               # Sample articles for testing
├── presentation.md            # PPT content
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Running the App

**Terminal 1 – Backend:**
```bash
cd backend
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 – Frontend:**
```bash
streamlit run frontend/streamlit_app.py --server.port 8501
```

Open **http://localhost:8501** in your browser.

## API Documentation

### `GET /`
Returns API health status.

### `POST /predict`
Analyze a news article.

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

## Model Performance

- **Algorithm:** Random Forest Classifier
- **Accuracy:** 99.65%
- **Features:** TF-IDF (5,000 max features)
- **Estimators:** 100

## Dataset

~44,000 labeled news articles from Kaggle (True.csv + Fake.csv), split into real and fake categories.

## Future Improvements

- Deep learning models (LSTM, BERT)
- Multi-language support
- Source credibility scoring
- Browser extension
