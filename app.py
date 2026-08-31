"""
app.py
A FastAPI web service that loads the trained model (model.pkl) and
exposes a /predict endpoint. This is what gets deployed to Render.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Loan Approval Predictor",
    description="A simple ML API that predicts whether a loan should be approved.",
    version="1.0.0",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]


class LoanApplication(BaseModel):
    income: float = Field(..., example=300000, description="Monthly income")
    credit_score: float = Field(..., example=680, description="Credit score, 300-850")
    loan_amount: float = Field(..., example=750000, description="Amount requested")
    employment_years: float = Field(..., example=3.5, description="Years employed")
    existing_debt: float = Field(..., example=100000, description="Current outstanding debt")


class PredictionResponse(BaseModel):
    approved: bool
    approval_probability: float


@app.get("/")
def root():
    return {
        "message": "Loan Approval Predictor API is running.",
        "docs": "/docs",
        "predict_endpoint": "/predict (POST)",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    input_df = pd.DataFrame([application.model_dump()])[FEATURES]
    probability = model.predict_proba(input_df)[0][1]
    prediction = bool(model.predict(input_df)[0])

    return PredictionResponse(
        approved=prediction,
        approval_probability=round(float(probability), 4),
    )
