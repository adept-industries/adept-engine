"""Unit tests for the tabular PR Risk Analytics microservice."""

import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure analytics-service is importable
analytics_path = Path(__file__).resolve().parent.parent / "analytics-service"
if str(analytics_path) not in sys.path:
    sys.path.insert(0, str(analytics_path))

from main import app, compute_risk_level  # type: ignore[import-not-found] # noqa: E402


@pytest.fixture(scope="module")
def client() -> Generator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def test_analytics_service_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_analytics_service_predict_low_risk(client: TestClient) -> None:
    payload = {
        "la": 15.0,
        "ld": 3.0,
        "nf": 1.0,
        "ns": 1.0,
        "nd": 1.0,
        "entropy": 0.1,
        "ndev": 5.0,
        "lt": 50.0,
        "nuc": 10.0,
        "age": 2.0,
        "exp": 120.0,
        "rexp": 25.0,
        "sexp": 15.0,
        "fix": 1.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["probability"] <= 1.0
    assert 0 <= data["riskScore"] <= 100
    assert data["riskLevel"] in ("LOW", "MEDIUM", "HIGH")
    assert data["riskLevel"] == "LOW"


def test_analytics_service_predict_high_risk(client: TestClient) -> None:
    payload = {
        "la": 850.0,
        "ld": 320.0,
        "nf": 18.0,
        "ns": 6.0,
        "nd": 6.0,
        "entropy": 0.85,
        "ndev": 15.0,
        "lt": 2500.0,
        "nuc": 120.0,
        "age": 360.0,
        "exp": 0.0,
        "rexp": 0.0,
        "sexp": 0.0,
        "fix": 0.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["probability"] <= 1.0
    assert 0 <= data["riskScore"] <= 100
    assert data["riskLevel"] in ("LOW", "MEDIUM", "HIGH")


def test_compute_risk_level() -> None:
    assert compute_risk_level(0) == "LOW"
    assert compute_risk_level(30) == "LOW"
    assert compute_risk_level(31) == "MEDIUM"
    assert compute_risk_level(70) == "MEDIUM"
    assert compute_risk_level(71) == "HIGH"
    assert compute_risk_level(100) == "HIGH"
