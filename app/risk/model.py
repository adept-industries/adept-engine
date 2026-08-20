"""PR Risk ML Model Service using XGBoost, probability calibration, and SHAP explanations."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import structlog

from app.core.config import get_settings
from app.risk.features import RISK_FEATURES, risk_level, validate_feature_dict

logger = structlog.get_logger()


class RiskModelService:
    def __init__(self, artifact_dir: str | Path | None = None) -> None:
        settings = get_settings()
        self.artifact_dir = Path(artifact_dir or settings.risk_model_dir)
        self.model: Any = None
        self.calibrator: Any = None
        self.metadata: dict[str, Any] | None = None
        self.explainer: Any = None

    def load(self) -> bool:
        """Loads trained XGBoost model, calibrator, metadata, and SHAP explainer."""
        model_path = self.artifact_dir / "risk_model.joblib"
        cal_path = self.artifact_dir / "risk_calibrator.joblib"
        meta_path = self.artifact_dir / "risk_metadata.joblib"

        if not (model_path.exists() and cal_path.exists() and meta_path.exists()):
            logger.info("risk_model_not_found_auto_training_demo", dir=str(self.artifact_dir))
            try:
                from app.risk.synthetic import generate_synthetic_pr_dataset
                from app.risk.trainer import train_risk_model

                df = generate_synthetic_pr_dataset(n_samples=5000, seed=42)
                train_risk_model(df, artifact_dir=self.artifact_dir, is_demo=True)
            except Exception as exc:
                logger.error("risk_model_auto_train_failed", error=str(exc))
                return False

        try:
            self.model = joblib.load(model_path)
            self.calibrator = joblib.load(cal_path)
            self.metadata = joblib.load(meta_path)
            self.explainer = shap.TreeExplainer(self.model)
            logger.info(
                "risk_model_loaded_successfully",
                version=self.metadata.get("model_version"),
                medium_threshold=self.metadata.get("medium_threshold"),
                high_threshold=self.metadata.get("high_threshold"),
            )
            return True
        except Exception as exc:
            logger.error("risk_model_load_failed", error=str(exc))
            self.model = None
            self.calibrator = None
            self.metadata = None
            self.explainer = None
            return False

    @property
    def ready(self) -> bool:
        return (
            self.model is not None
            and self.calibrator is not None
            and self.metadata is not None
            and self.explainer is not None
        )

    def predict(self, feature_dict: dict[str, Any]) -> dict[str, Any]:
        """Runs inference on validated features, returning calibrated risk and SHAP factors."""
        if not self.ready:
            raise RuntimeError(
                "Risk model is not loaded. Train a model first via "
                "'uv run python -m app.risk.cli train-demo'."
            )

        validated = validate_feature_dict(feature_dict)
        X = pd.DataFrame([[validated[f] for f in RISK_FEATURES]], columns=RISK_FEATURES)

        # 1. Base XGBoost probability
        raw_prob_arr = self.model.predict_proba(X)
        raw_p = (
            float(raw_prob_arr[:, 1][0]) if raw_prob_arr.shape[1] > 1 else float(raw_prob_arr[0][0])
        )

        # 2. Probability calibration
        cal_input = np.array([[raw_p]])
        cal_prob_arr = self.calibrator.predict_proba(cal_input)
        calibrated_p = (
            float(cal_prob_arr[:, 1][0]) if cal_prob_arr.shape[1] > 1 else float(cal_prob_arr[0][0])
        )
        calibrated_p = max(0.0, min(1.0, calibrated_p))

        # 3. SHAP TreeExplainer factors
        shap_values = self.explainer.shap_values(X)
        shap_arr = np.asarray(shap_values)
        if shap_arr.ndim == 3:
            shap_arr = shap_arr[:, :, -1]
        elif shap_arr.ndim == 1:
            shap_arr = shap_arr.reshape(1, -1)

        row = shap_arr[0]
        feature_values = X.iloc[0].tolist()
        ranked = sorted(
            zip(RISK_FEATURES, row, feature_values, strict=False),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:5]

        top_factors = [
            {
                "feature": name,
                "value": round(float(val), 4),
                "impact": round(float(imp), 4),
                "direction": "raises_risk" if float(imp) > 0 else "lowers_risk",
            }
            for name, imp, val in ranked
        ]

        meta = self.metadata or {}
        medium_thresh = float(meta.get("medium_threshold", 0.15))
        high_thresh = float(meta.get("high_threshold", 0.30))
        level = risk_level(calibrated_p, medium_threshold=medium_thresh, high_threshold=high_thresh)

        return {
            "probability": round(calibrated_p, 4),
            "risk_level": level,
            "model_version": meta.get("model_version", "unknown"),
            "top_factors": top_factors,
            "predicted_at": datetime.now(UTC),
        }


# Global singleton instance
risk_model = RiskModelService()
