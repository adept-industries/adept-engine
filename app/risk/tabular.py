"""Tabular Random Forest PR Risk scoring service and schema."""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel

logger = logging.getLogger("pr_risk_tabular")

# Candidate search paths for the Random Forest model artifact
_SEARCH_PATHS = [
    Path(__file__).resolve().parent / "pr_risk_model.joblib",
    Path(__file__).resolve().parent.parent.parent / "analytics-service" / "pr_risk_model.joblib",
    Path("/app/app/risk/pr_risk_model.joblib"),
    Path("/app/analytics-service/pr_risk_model.joblib"),
]

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

tabular_model: Any = None


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


def load_tabular_model() -> Any:
    global tabular_model
    target_path: Path | None = None
    for p in _SEARCH_PATHS:
        if p.exists():
            target_path = p
            break

    if target_path is None:
        logger.warning(
            "Trained Random Forest model not found in %s. Initializing fallback baseline model.",
            [str(p) for p in _SEARCH_PATHS],
        )
        from sklearn.ensemble import RandomForestClassifier

        fallback = RandomForestClassifier(n_estimators=100, random_state=42)
        x_dummy = np.array(
            [
                [15, 3, 1, 1, 1, 0.1, 5, 100, 2, 2, 120, 25, 15, 1],
                [850, 320, 18, 6, 6, 0.9, 1, 15, 1, 120, 0, 0, 0, 0],
            ]
        )
        y_dummy = np.array([0, 1])
        fallback.fit(x_dummy, y_dummy)
        tabular_model = fallback
        logger.info("Initialized fallback tabular model successfully.")
        return tabular_model

    tabular_model = joblib.load(target_path)
    logger.info("Loaded Random Forest model successfully from %s", target_path)
    return tabular_model


def predict_pr_risk(features: PRFeatures) -> PredictResponse:
    global tabular_model
    if tabular_model is None:
        load_tabular_model()
    if tabular_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    feature_dict = features.model_dump()
    df = pd.DataFrame([feature_dict])[FEATURES]

    probabilities = tabular_model.predict_proba(df)
    prob = float(probabilities[0, 1])

    risk_score = int(round(prob * 100))
    risk_level = compute_risk_level(risk_score)

    return PredictResponse(
        probability=prob,
        riskScore=risk_score,
        riskLevel=risk_level,
    )
