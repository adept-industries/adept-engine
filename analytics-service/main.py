from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path(__file__).resolve().parent / "pr_risk_model.joblib"

FEATURES = [
    "la",
    "ld",
    "nf",
    "ns",
    "nd",
    "entropy",
    "ndev",
    "lt",
    "nuc",
    "age",
    "exp",
    "rexp",
    "sexp",
    "fix",
]

model: Any = None


def load_model():
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run train_model.py first."
        )
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="PR Risk Analytics Service",
    description="Lightweight Tabular ML microservice for PR risk scoring using RandomForest.",
    version="1.0.0",
    lifespan=lifespan,
)


class PRFeatures(BaseModel):
    la: float = 0.0
    ld: float = 0.0
    nf: float = 0.0
    ns: float = 0.0
    nd: float = 0.0
    entropy: float = 0.0
    ndev: float = 0.0
    lt: float = 0.0
    nuc: float = 0.0
    age: float = 0.0
    exp: float = 0.0
    rexp: float = 0.0
    sexp: float = 0.0
    fix: float = 0.0


class PredictResponse(BaseModel):
    probability: float
    riskScore: int
    riskLevel: str


def compute_risk_level(risk_score: int) -> str:
    if risk_score <= 30:
        return "LOW"
    elif risk_score <= 70:
        return "MEDIUM"
    else:
        return "HIGH"


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(features: PRFeatures) -> PredictResponse:
    global model
    if model is None:
        load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    # Convert features to DataFrame ensuring exact feature order
    feature_dict = features.model_dump()
    df = pd.DataFrame([feature_dict])[FEATURES]

    # Predict defect probability (class 1)
    probabilities = model.predict_proba(df)
    prob = float(probabilities[0, 1])

    risk_score = int(round(prob * 100))
    risk_level = compute_risk_level(risk_score)

    return PredictResponse(
        probability=prob,
        riskScore=risk_score,
        riskLevel=risk_level,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
