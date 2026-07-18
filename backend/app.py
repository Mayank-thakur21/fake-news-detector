from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predictor import load_models, predict_news

app = FastAPI(title="Fake News Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NewsInput(BaseModel):
    news: str


@app.on_event("startup")
def startup():
    load_models()


@app.get("/")
def root():
    return {"message": "Fake News Detection API Running"}


@app.post("/predict")
def predict(input_data: NewsInput):
    try:
        result = predict_news(input_data.news)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
