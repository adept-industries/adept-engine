"""FastAPI application for Adept Engine including PR Risk Analytics."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Query

from app.core.logging import configure_logging
from app.db.session import current_schema_version, get_database_engine
from app.risk.dx import compute_dx_score
from app.risk.model import risk_model
from app.risk.schemas import (
    DXRequest,
    DXResponse,
    ModelMetadataResponse,
    OutcomeRecordRequest,
    OutcomeRecordResponse,
    RiskPredictionResponse,
    RiskPredictRequest,
    StaleCheckRequest,
    StaleCheckResponse,
)
from app.risk.service import (
    check_stale_for_pr,
    compute_repo_dx_score,
    get_latest_risk_for_pr,
    list_risky_prs_for_repository,
    predict_and_persist,
    record_outcome,
)
from app.risk.stale import check_stale
from app.risk.synthetic import generate_synthetic_pr_dataset
from app.risk.tabular import (
    PredictResponse,
    PRFeatures,
    load_tabular_model,
    predict_pr_risk,
)
from app.risk.trainer import train_risk_model

configure_logging()
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Attempt to load model on startup
    loaded = risk_model.load()
    if not loaded:
        logger.info("risk_model_not_yet_loaded_at_startup")
    load_tabular_model()
    yield


app = FastAPI(title="Adept Engine", version="0.1.0", lifespan=lifespan)


@app.post("/predict", response_model=PredictResponse)
def predict_tabular(features: PRFeatures) -> PredictResponse:
    """Inference entry point for real-time tabular Random Forest PR risk scoring."""
    return predict_pr_risk(features)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "UP", "modelReady": risk_model.ready}


@app.get("/ready")
def ready() -> dict[str, str]:
    try:
        version = current_schema_version(get_database_engine())
    except Exception:
        logger.warning("engine_readiness_failed")
        raise HTTPException(
            status_code=503,
            detail="database or Flyway schema is not ready",
        ) from None

    return {"status": "READY", "schemaVersion": version}


# ── PR Risk Analytics Endpoints ───────────────────────────────────────────────


@app.post("/v1/risk/predict", response_model=RiskPredictionResponse)
def predict_risk(req: RiskPredictRequest) -> dict[str, Any]:
    """Inference entry point: runs calibrated XGBoost risk model with SHAP explanations."""
    if not risk_model.ready:
        raise HTTPException(
            status_code=503,
            detail="Risk model is not loaded. Train a model first via 'train-demo'.",
        )

    try:
        if req.persist:
            db_engine = get_database_engine()
            return predict_and_persist(
                database_engine=db_engine,
                repo_identifier=req.repository_id,
                pr_number=req.pr_number,
                custom_features=req.features,
                stage=req.stage,
                snapshot_at=req.snapshot_at,
            )
        else:
            pred = risk_model.predict(req.features)
            return {
                "repository_id": req.repository_id,
                "pr_number": req.pr_number,
                "risk_probability": pred["probability"],
                "risk_level": pred["risk_level"],
                "model_version": pred["model_version"],
                "top_factors": pred["top_factors"],
                "predicted_at": pred["predicted_at"],
                "stage": req.stage,
            }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("predict_risk_endpoint_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v1/repositories/{repo_id}/pull-requests/{pr_number}/risk")
def get_pr_risk(repo_id: str, pr_number: int) -> dict[str, Any]:
    """Retrieves the latest risk prediction and factors for a PR."""
    try:
        db_engine = get_database_engine()
        result = get_latest_risk_for_pr(db_engine, repo_id, pr_number)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"No risk prediction found for repository '{repo_id}' PR #{pr_number}",
            )
        return result
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/repositories/{repo_id}/pull-requests/{pr_number}/risk/recalculate")
def recalculate_pr_risk(repo_id: str, pr_number: int) -> dict[str, Any]:
    """Recalculates risk from latest database state and persists the new prediction."""
    if not risk_model.ready:
        raise HTTPException(status_code=503, detail="Risk model is not loaded.")
    try:
        db_engine = get_database_engine()
        return predict_and_persist(
            database_engine=db_engine,
            repo_identifier=repo_id,
            pr_number=pr_number,
            stage="live",
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v1/repositories/{repo_id}/pull-requests/risky")
def list_risky_prs(
    repo_id: str,
    min_level: str = Query(default="MEDIUM", pattern="^(MEDIUM|HIGH)$"),
) -> list[dict[str, Any]]:
    """Lists currently open PRs with HIGH/MEDIUM risk in a repository."""
    try:
        db_engine = get_database_engine()
        return list_risky_prs_for_repository(db_engine, repo_id, min_level=min_level)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/stale/check", response_model=StaleCheckResponse)
def check_stale_endpoint(req: StaleCheckRequest) -> dict[str, Any]:
    """Deterministic stale PR detection."""
    thresh = req.threshold_hours if req.threshold_hours is not None else 120.0
    res = check_stale(
        is_open=req.is_open,
        hours_since_last_activity=req.hours_since_last_activity,
        threshold_hours=thresh,
    )
    return {
        "repository_id": req.repository_id,
        "pr_number": req.pr_number,
        **res,
    }


@app.get("/v1/repositories/{repo_id}/pull-requests/{pr_number}/stale")
def get_pr_stale_status(
    repo_id: str,
    pr_number: int,
    threshold_hours: float | None = Query(default=None),
) -> dict[str, Any]:
    """Evaluates staleness of a PR from database activity timestamps."""
    try:
        db_engine = get_database_engine()
        return check_stale_for_pr(db_engine, repo_id, pr_number, threshold_hours=threshold_hours)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/outcomes", response_model=OutcomeRecordResponse)
def record_outcome_endpoint(req: OutcomeRecordRequest) -> dict[str, Any]:
    """Records an adverse outcome or clean observation for a merged PR."""
    try:
        db_engine = get_database_engine()
        return record_outcome(
            database_engine=db_engine,
            repo_identifier=req.repository_id,
            pr_number=req.pr_number,
            is_risky=req.is_risky,
            reason=req.reason,
            evidence=req.evidence,
            merged_at=req.merged_at,
            observed_until=req.observed_until,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/dx/score", response_model=DXResponse)
def dx_score_endpoint(req: DXRequest) -> dict[str, Any]:
    """Calculates repository/team Developer Experience workflow score."""
    return compute_dx_score(
        median_first_review_hours=req.median_first_review_hours,
        median_pr_cycle_hours=req.median_pr_cycle_hours,
        stale_pr_rate=req.stale_pr_rate,
        change_failure_rate=req.change_failure_rate,
        ci_success_rate=req.ci_success_rate,
        custom_weights=req.weights,
    )


@app.get("/v1/repositories/{repo_id}/dx-score")
def get_repo_dx_score(repo_id: str) -> dict[str, Any]:
    """Computes DX workflow score for a repository from historical database records."""
    try:
        db_engine = get_database_engine()
        return compute_repo_dx_score(db_engine, repo_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v1/risk/model/latest", response_model=ModelMetadataResponse)
def get_latest_model_metadata() -> dict[str, Any]:
    """Retrieves active model registry metadata and metrics."""
    if not risk_model.ready:
        raise HTTPException(status_code=404, detail="No active model loaded.")
    meta = risk_model.metadata or {}
    return {
        "model_name": meta.get("model_name", "pr-code-change-risk-xgb"),
        "model_version": meta.get("model_version", "unknown"),
        "trained_at": meta.get("trained_at", ""),
        "feature_schema_version": meta.get("feature_schema_version", "v1"),
        "feature_names": meta.get("feature_names", []),
        "thresholds": {
            "medium": meta.get("medium_threshold", 0.15),
            "high": meta.get("high_threshold", 0.30),
        },
        "metrics": meta.get("metrics", {}),
        "is_demo": meta.get("is_demo", False),
    }


@app.post("/v1/risk/train")
def train_model_endpoint(demo: bool = Query(default=True)) -> dict[str, Any]:
    """Triggers model training (demo synthetic dataset or real database data)."""
    if demo:
        df = generate_synthetic_pr_dataset(n_samples=5000, seed=42)
        db_engine = None
        with suppress(Exception):
            db_engine = get_database_engine()
        res = train_risk_model(df, is_demo=True, database_engine=db_engine)
        risk_model.load()
        return {
            "status": "success",
            "model_version": res["model_version"],
            "metrics": res["metrics"],
            "high_threshold": res["high_threshold"],
            "medium_threshold": res["medium_threshold"],
        }
    else:
        raise HTTPException(
            status_code=400,
            detail="Real historical training requires at least 30 labeled outcomes in DB.",
        )
