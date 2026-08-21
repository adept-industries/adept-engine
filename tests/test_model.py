"""Tests for XGBoost model inference, calibration, and SHAP factor generation."""

from app.risk.features import RISK_FEATURES
from app.risk.model import RiskModelService
from app.risk.synthetic import generate_synthetic_pr_dataset
from app.risk.trainer import train_risk_model


def test_model_training_and_inference(tmp_path: str) -> None:
    # 1. Train model on small synthetic dataset
    df = generate_synthetic_pr_dataset(n_samples=200, seed=42)
    train_res = train_risk_model(df, artifact_dir=tmp_path, is_demo=True)

    assert "model_version" in train_res
    assert "metrics" in train_res
    assert 0.0 <= train_res["metrics"]["roc_auc"] <= 1.0
    assert 0.0 <= train_res["metrics"]["pr_auc"] <= 1.0
    assert 0.0 <= train_res["metrics"]["brier_score"] <= 1.0

    # 2. Load model service from disk
    service = RiskModelService(artifact_dir=tmp_path)
    assert service.load() is True
    assert service.ready is True

    # 3. Predict on sample features
    sample_features = {f: 1.0 for f in RISK_FEATURES}
    sample_features["hotspot_score"] = 0.9
    sample_features["ci_failures"] = 3
    sample_features["lines_added"] = 2000

    pred = service.predict(sample_features)

    assert "probability" in pred
    assert 0.0 <= pred["probability"] <= 1.0
    assert pred["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert pred["model_version"] == train_res["model_version"]

    # 4. Check SHAP explanations
    top_factors = pred["top_factors"]
    assert len(top_factors) <= 5
    for factor in top_factors:
        assert factor["feature"] in RISK_FEATURES
        assert isinstance(factor["value"], float)
        assert isinstance(factor["impact"], float)
        assert factor["direction"] in ("raises_risk", "lowers_risk")
