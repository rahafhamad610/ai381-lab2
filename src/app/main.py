from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "model.joblib"


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenure_months: int = Field(ge=0, strict=True)
    monthly_charges: float = Field(ge=0, allow_inf_nan=False)
    contract: Literal["month-to-month", "one-year", "two-year"]
    internet_service: Literal["dsl", "fiber", "none"]
    support_calls: int = Field(ge=0, strict=True)


class Prediction(BaseModel):
    churn_probability: float = Field(ge=0, le=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="AI381 Lab 2 Churn API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    features = pd.DataFrame([customer.model_dump()])
    probability = float(app.state.model.predict_proba(features)[0, 1])
    return Prediction(churn_probability=round(probability, 3))
