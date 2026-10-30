
from fastapi import FastAPI
from pydantic import BaseModel

from src.prediction.predict import predict_sentiment


# Initialize FastAPI
app = FastAPI(
    title="Amazon Review Sentiment Analysis API",
    description="API for predicting sentiment of Amazon book reviews",
    version="1.0.0"
)


# Request schema
class ReviewRequest(BaseModel):
    review: str


# Root endpoint
@app.get("/")
def home():
    return {
        "message": "Amazon Review Sentiment Analysis API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict(request: ReviewRequest):

    result = predict_sentiment(request.review)

    return {
        "review": request.review,
        "sentiment": result
    }
