from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml_training.src.artifact_export import export_artifact, verify_exported_artifact
from ml_training.src.constants import (
    FEATURE_ORDER,
    FEATURE_SCHEMA_VERSION,
    MODEL_NAME,
    MODEL_VERSION,
)
from ml_training.src.dataset_loader import sha256_file
from ml_training.src.evaluation import ThresholdSelection, evaluate_probabilities, select_thresholds


def test_validation_threshold_selection_is_ordered_and_deterministic() -> None:
    labels = np.array([0] * 30 + [1] * 10, dtype=np.int8)
    generator = np.random.default_rng(42)
    generator.random(40)
    probabilities = generator.random(40)

    first = select_thresholds(labels, probabilities)
    second = select_thresholds(labels, probabilities)

    assert first == second
    assert 0 <= first.medium < first.high < first.critical <= 1


def test_evaluation_reports_core_calibration_and_effort_metrics() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])
    lines_changed = np.array([10.0, 10.0, 10.0, 10.0])

    result = evaluate_probabilities(
        labels,
        probabilities,
        classification_threshold=0.5,
        lines_changed=lines_changed,
    )

    assert result["rocAuc"] == 1.0
    assert result["prAuc"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["confusionMatrix"] == {
        "trueNegative": 2,
        "falsePositive": 0,
        "falseNegative": 0,
        "truePositive": 2,
    }
    assert result["calibration"]
    assert result["effortAware"]["pOpt"] is not None  # type: ignore[index]


def test_exported_artifact_has_verifiable_identity_order_width_and_checksums(
    tmp_path: Path,
) -> None:
    features = np.array(
        [
            [1, 1, 1, 0, 1, 0, 0],
            [2, 2, 2, 1, 20, 5, 1],
            [1, 1, 1, 0, 2, 0, 0],
            [3, 3, 4, 2, 40, 10, 1],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 1, 0, 1], dtype=np.int8)
    model = RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=1).fit(features, labels)
    thresholds = ThresholdSelection(
        medium=0.1,
        high=0.4,
        critical=0.8,
        method="test-only fixed thresholds",
    )

    metadata = export_artifact(
        model=model,
        report={"test": True},
        thresholds=thresholds,
        output_dir=tmp_path,
    )

    stored_metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    loaded_model = joblib.load(tmp_path / "model.joblib")
    assert metadata == stored_metadata
    assert stored_metadata["modelName"] == MODEL_NAME
    assert stored_metadata["modelVersion"] == MODEL_VERSION
    assert stored_metadata["featureSchemaVersion"] == FEATURE_SCHEMA_VERSION
    assert stored_metadata["featureOrder"] == list(FEATURE_ORDER)
    assert stored_metadata["sha256"]["model.joblib"] == sha256_file(tmp_path / "model.joblib")
    assert stored_metadata["sha256"]["training-report.json"] == sha256_file(
        tmp_path / "training-report.json"
    )
    assert loaded_model.n_features_in_ == len(FEATURE_ORDER)
    assert loaded_model.classes_.tolist() == [0, 1]
    probability = loaded_model.predict_proba(features[:1])[0, 1]
    assert np.isfinite(probability)
    verify_exported_artifact(tmp_path)
