"""Tests for PR Risk FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.risk.features import RISK_FEATURES
from app.risk.model import risk_model
from app.risk.synthetic import generate_synthetic_pr_dataset
from app.risk.trainer import train_risk_model


@pytest.fixture(scope="module", autouse=True)
def ensure_model_loaded(tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_dir = tmp_path_factory.mktemp("models")
    df = generate_synthetic_pr_dataset(n_samples=150, seed=42)
    train_risk_model(df, artifact_dir=tmp_dir, is_demo=True)
    risk_model.artifact_dir = tmp_dir
    risk_model.load()


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "UP"
    assert data["modelReady"] is True


def test_stale_check_endpoint() -> None:
    client = TestClient(app)
    payload = {
        "repository_id": "repo-123",
        "pr_number": 10,
        "is_open": True,
        "hours_since_last_activity": 130.5,
        "threshold_hours": 120.0,
    }
    response = client.post("/v1/stale/check", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_stale"] is True
    assert data["hours_since_last_activity"] == 130.5
    assert data["threshold_hours"] == 120.0


def test_dx_score_endpoint() -> None:
    client = TestClient(app)
    payload = {
        "median_first_review_hours": 8.0,
        "median_pr_cycle_hours": 36.0,
        "stale_pr_rate": 0.05,
        "change_failure_rate": 0.05,
        "ci_success_rate": 0.95,
    }
    response = client.post("/v1/dx/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["score"] <= 100.0
    assert "components" in data
    assert "weights" in data


def test_risk_predict_endpoint_unpersisted() -> None:
    client = TestClient(app)
    features = {f: 1.0 for f in RISK_FEATURES}
    features["hotspot_score"] = 0.8
    features["ci_failures"] = 1

    payload = {
        "repository_id": "repo-123",
        "pr_number": 50,
        "features": features,
        "persist": False,
    }
    response = client.post("/v1/risk/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert "top_factors" in data
    assert len(data["top_factors"]) <= 5


def test_model_metadata_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/v1/risk/model/latest")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "model_version" in data
    assert "feature_names" in data
    assert "thresholds" in data
    assert data["is_demo"] is True
