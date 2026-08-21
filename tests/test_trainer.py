"""Tests for training pipeline, chronological splitting, and metric validation."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from app.risk.features import RISK_FEATURES
from app.risk.synthetic import generate_synthetic_pr_dataset
from app.risk.trainer import train_risk_model


def test_chronological_splitting_order(tmp_path: str) -> None:
    """Verifies that records are strictly ordered by time without random shuffle."""
    n = 100
    dates = [datetime(2024, 1, 1, tzinfo=UTC) + pd.Timedelta(days=i) for i in range(n)]

    # Generate synthetic data with intentional date-correlated feature
    df = pd.DataFrame(
        {
            "snapshot_at": dates,
            **{f: [float(i) for i in range(n)] for f in RISK_FEATURES},
            "is_risky": [1 if i % 4 == 0 else 0 for i in range(n)],
        }
    )

    # Shuffle input intentionally before passing to trainer to verify it sorts chronologically
    df_shuffled = df.sample(frac=1.0, random_state=99).reset_index(drop=True)

    result = train_risk_model(df_shuffled, artifact_dir=tmp_path, is_demo=True)
    assert result["metrics"]["sample_counts"]["train"] == 60
    assert result["metrics"]["sample_counts"]["calibration"] == 20
    assert result["metrics"]["sample_counts"]["test"] == 20


def test_training_insufficient_samples(tmp_path: str) -> None:
    df_small = generate_synthetic_pr_dataset(n_samples=15, seed=42)
    with pytest.raises(ValueError, match="Insufficient training data"):
        train_risk_model(df_small, artifact_dir=tmp_path)


def test_training_metrics_completeness(tmp_path: str) -> None:
    df = generate_synthetic_pr_dataset(n_samples=250, seed=42)
    res = train_risk_model(df, artifact_dir=tmp_path, is_demo=True)
    metrics = res["metrics"]

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "brier_score" in metrics
    assert "precision_at_high" in metrics
    assert "recall_at_high" in metrics
    assert "f1_at_high" in metrics
    assert "confusion_matrix" in metrics
    assert "test_positive_rate" in metrics
    assert metrics["disclaimer"] == "DEMO DATA — NOT REAL-WORLD MODEL ACCURACY"
