"""Historical model training and calibration pipeline.

CRITICAL RULES:
1. Historical records must be ordered chronologically.
2. TRAIN = oldest historical records.
3. CALIBRATION/VALIDATION = later records.
4. TEST = newest untouched records.
5. Base XGBoost model trained ONLY on train set.
6. Probability calibrator fitted ONLY on calibration set.
7. High threshold determined from validation data favoring precision.
8. Test set evaluated once at the very end.
"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import structlog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sqlalchemy import Engine, text
from xgboost import XGBClassifier

from app.core.config import get_settings
from app.risk.features import FEATURE_SCHEMA_VERSION, RISK_FEATURES
from app.risk.synthetic import DEMO_DISCLAIMER

logger = structlog.get_logger()


def train_risk_model(
    df: pd.DataFrame,
    artifact_dir: str | Path | None = None,
    is_demo: bool = False,
    database_engine: Engine | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Executes chronological training, calibration, threshold tuning, and evaluation."""
    if len(df) < 30:
        raise ValueError(f"Insufficient training data: {len(df)} rows. Minimum 30 required.")

    # 1. Strict chronological ordering
    if "snapshot_at" in df.columns:
        df_sorted = df.sort_values("snapshot_at").reset_index(drop=True)
    else:
        df_sorted = df.reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * 0.60)
    cal_end = int(n * 0.80)

    train_df = df_sorted.iloc[:train_end]
    cal_df = df_sorted.iloc[train_end:cal_end]
    test_df = df_sorted.iloc[cal_end:]

    X_train = train_df[RISK_FEATURES]
    y_train = train_df["is_risky"].astype(int)

    X_cal = cal_df[RISK_FEATURES]
    y_cal = cal_df["is_risky"].astype(int)

    X_test = test_df[RISK_FEATURES]
    y_test = test_df["is_risky"].astype(int)

    # Check positive label counts
    pos_train = int(y_train.sum())
    neg_train = int((y_train == 0).sum())
    if pos_train == 0 or neg_train == 0:
        raise ValueError("Training partition lacks class diversity (all positive or all negative).")

    scale_pos_weight = float(neg_train / max(1, pos_train))

    # 2. Base XGBoost classifier
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        learning_rate=0.035,
        max_depth=4,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=random_seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=False)

    # 3. Probability calibration fitted only on calibration partition
    raw_cal = model.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, random_state=random_seed)
    calibrator.fit(raw_cal, y_cal)

    # 4. Tune alert policy thresholds on calibration/validation data
    # High threshold is tuned to favor high precision (avoiding noisy warnings)
    cal_calibrated_probs = calibrator.predict_proba(raw_cal)[:, 1]

    best_high_thresh = 0.30
    best_medium_thresh = 0.15

    candidate_thresholds = np.linspace(0.10, 0.70, 61)
    for thresh in candidate_thresholds:
        pred_high = (cal_calibrated_probs >= thresh).astype(int)
        if pred_high.sum() > 0:
            prec = precision_score(y_cal, pred_high, zero_division=0)
            if prec >= 0.75 and thresh >= 0.25:
                best_high_thresh = float(round(thresh, 2))
                break

    best_medium_thresh = float(round(max(0.10, best_high_thresh / 2.0), 2))

    # 5. Untouched test evaluation
    raw_test = model.predict_proba(X_test)[:, 1].reshape(-1, 1)
    test_calibrated_probs = calibrator.predict_proba(raw_test)[:, 1]

    test_pred_high = (test_calibrated_probs >= best_high_thresh).astype(int)
    cm = confusion_matrix(y_test, test_pred_high).tolist()

    metrics: dict[str, Any] = {
        "disclaimer": DEMO_DISCLAIMER if is_demo else "REAL DATA",
        "roc_auc": round(float(roc_auc_score(y_test, test_calibrated_probs)), 4),
        "pr_auc": round(float(average_precision_score(y_test, test_calibrated_probs)), 4),
        "brier_score": round(float(brier_score_loss(y_test, test_calibrated_probs)), 4),
        "precision_at_high": round(
            float(precision_score(y_test, test_pred_high, zero_division=0)), 4
        ),
        "recall_at_high": round(float(recall_score(y_test, test_pred_high, zero_division=0)), 4),
        "f1_at_high": round(float(f1_score(y_test, test_pred_high, zero_division=0)), 4),
        "confusion_matrix": cm,
        "test_positive_rate": round(float(y_test.mean()), 4),
        "sample_counts": {
            "train": len(train_df),
            "calibration": len(cal_df),
            "test": len(test_df),
            "total": n,
        },
    }

    # 6. Save model artifacts
    settings = get_settings()
    out_dir = Path(artifact_dir or settings.risk_model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "risk_model.joblib"
    cal_path = out_dir / "risk_calibrator.joblib"
    meta_path = out_dir / "risk_metadata.joblib"

    timestamp_str = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    prefix = "risk-xgb-demo" if is_demo else "risk-xgb-prod"
    model_version = f"{prefix}-{timestamp_str}"

    metadata = {
        "model_name": "pr-code-change-risk-xgb",
        "model_version": model_version,
        "trained_at": datetime.now(UTC).isoformat(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": RISK_FEATURES,
        "medium_threshold": best_medium_thresh,
        "high_threshold": best_high_thresh,
        "metrics": metrics,
        "is_demo": is_demo,
    }

    joblib.dump(model, model_path)
    joblib.dump(calibrator, cal_path)
    joblib.dump(metadata, meta_path)

    # Compute hash of model artifact for registry
    with open(model_path, "rb") as f:
        artifact_hash = hashlib.sha256(f.read()).hexdigest()

    # 7. Register in database if engine is provided
    if database_engine is not None:
        try:
            with database_engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO model_registry (
                            model_name, model_version, trained_at, feature_schema_version,
                            feature_names, thresholds, train_range, metrics,
                            artifact_path, artifact_hash
                        ) VALUES (
                            :model_name, :model_version, now(), :feature_schema_version,
                            CAST(:feature_names AS jsonb), CAST(:thresholds AS jsonb),
                            CAST(:train_range AS jsonb), CAST(:metrics AS jsonb),
                            :artifact_path, :artifact_hash
                        ) ON CONFLICT (model_version) DO UPDATE SET
                            metrics = EXCLUDED.metrics,
                            thresholds = EXCLUDED.thresholds,
                            updated_at = now()
                        """
                    ),
                    {
                        "model_name": metadata["model_name"],
                        "model_version": model_version,
                        "feature_schema_version": FEATURE_SCHEMA_VERSION,
                        "feature_names": pd.Series(RISK_FEATURES).to_json(),
                        "thresholds": pd.Series(
                            {
                                "medium": best_medium_thresh,
                                "high": best_high_thresh,
                            }
                        ).to_json(),
                        "train_range": pd.Series(
                            {
                                "train_count": len(train_df),
                                "cal_count": len(cal_df),
                                "test_count": len(test_df),
                            }
                        ).to_json(),
                        "metrics": pd.Series(metrics).to_json(),
                        "artifact_path": str(model_path),
                        "artifact_hash": artifact_hash,
                    },
                )
        except Exception as exc:
            logger.warning("model_registry_db_insert_failed", error=str(exc))

    logger.info(
        "risk_model_trained_successfully",
        model_version=model_version,
        roc_auc=metrics["roc_auc"],
        pr_auc=metrics["pr_auc"],
        brier=metrics["brier_score"],
    )

    return {
        "model_version": model_version,
        "metrics": metrics,
        "medium_threshold": best_medium_thresh,
        "high_threshold": best_high_thresh,
        "metadata": metadata,
    }
