"""
FastAPI backend for churn prediction.
Run: uvicorn api.main:app --reload
"""

from pathlib import Path
import sys

# Allow importing from src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd

from predict import load_model, predict_proba

app = FastAPI(
    title="Churn Early-Warning API",
    description="Predict customer churn probability",
    version="1.0.0",
)

# Load model once at startup
try:
    model = load_model()
except FileNotFoundError as e:
    model = None
    print(f"WARNING: {e}")


class CustomerFeatures(BaseModel):
    tenure: int = Field(..., example=12, description="Months with the company")
    MonthlyCharges: float = Field(..., example=70.5)
    TotalCharges: float = Field(..., example=850.0)
    Contract: str = Field(..., example="Month-to-month")
    InternetService: str = Field(..., example="Fiber optic")
    PaymentMethod: str = Field(..., example="Electronic check")
    # Optional extra fields (will be ignored if not in training columns)
    gender: Optional[str] = None
    SeniorCitizen: Optional[int] = None
    Partner: Optional[str] = None
    Dependents: Optional[str] = None
    PhoneService: Optional[str] = None
    MultipleLines: Optional[str] = None
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    PaperlessBilling: Optional[str] = None


@app.get("/")
def root():
    return {
        "message": "Churn Early-Warning API",
        "docs": "/docs",
        "predict": "POST /predict",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(customer: CustomerFeatures):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run: python src/train_model.py",
        )

    # Convert to DataFrame with a single row
    data = customer.model_dump(exclude_none=True)
    df = pd.DataFrame([data])

    try:
        proba = float(predict_proba(df).iloc[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    risk = "High" if proba >= 0.5 else "Low"

    return {
        "churn_probability": round(proba, 4),
        "risk_label": risk,
        "message": f"Customer has a {proba:.1%} chance of churning.",
    }
