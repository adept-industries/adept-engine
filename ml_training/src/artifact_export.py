from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy
import numpy as np
import sklearn

from ml_training.src.constants import (
    FEATURE_ORDER,
    FEATURE_SCHEMA_VERSION,
    MODEL_NAME,
    MODEL_VERSION,
)
from ml_training.src.dataset_loader import sha256_file
from ml_training.src.evaluation import ThresholdSelection

LIMITATIONS = (
    "The training records are commit-level while runtime predictions aggregate a whole "
    "pull request.",
    "Training projects and languages may differ from repositories analysed by Adept.",
    "The score ranks estimated review risk and does not prove that a defect exists.",
    "The prepared dataset FIX field is used for training; runtime keyword matching is an "
    "MVP approximation.",
    "The prepared change metrics exclude some non-source files; the later runtime extractor "
    "must make and document a file-scope decision before claiming training/runtime parity.",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def export_artifact(
    *,
    model: Any,
    report: dict[str, Any],
    thresholds: ThresholdSelection,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    report_path = output_dir / "training-report.json"
    metadata_path = output_dir / "metadata.json"

    joblib.dump(model, model_path, compress=3)
    _write_json(report_path, report)
    metadata: dict[str, Any] = {
        "modelName": MODEL_NAME,
        "modelVersion": MODEL_VERSION,
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "featureOrder": list(FEATURE_ORDER),
        "trainingUnit": "commit",
        "runtimePredictionUnit": "pull_request_aggregate",
        "pythonVersion": platform.python_version(),
        "scikitLearnVersion": sklearn.__version__,
        "numpyVersion": numpy.__version__,
        "joblibVersion": joblib.__version__,
        "trainingDataset": "JIT-Fine replication package data.zip",
        "thresholds": thresholds.as_dict(),
        "sha256": {
            "model.joblib": sha256_file(model_path),
            "training-report.json": sha256_file(report_path),
        },
        "limitations": list(LIMITATIONS),
    }
    _write_json(metadata_path, metadata)
    verify_exported_artifact(output_dir)
    return metadata


def verify_exported_artifact(output_dir: Path) -> None:
    metadata_path = output_dir / "metadata.json"
    model_path = output_dir / "model.joblib"
    report_path = output_dir / "training-report.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_identity = {
        "modelName": MODEL_NAME,
        "modelVersion": MODEL_VERSION,
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "featureOrder": list(FEATURE_ORDER),
        "pythonVersion": platform.python_version(),
        "scikitLearnVersion": sklearn.__version__,
        "numpyVersion": numpy.__version__,
        "joblibVersion": joblib.__version__,
    }
    for key, expected in expected_identity.items():
        if metadata.get(key) != expected:
            raise ValueError(f"artifact metadata mismatch for {key}")
    for filename, path in (("model.joblib", model_path), ("training-report.json", report_path)):
        if metadata.get("sha256", {}).get(filename) != sha256_file(path):
            raise ValueError(f"artifact checksum mismatch for {filename}")
    thresholds = metadata.get("thresholds", {})
    values = [thresholds.get(level) for level in ("medium", "high", "critical")]
    if not all(isinstance(value, (int, float)) for value in values):
        raise ValueError("artifact thresholds must be numeric")
    medium, high, critical = (float(value) for value in values)
    if not 0.0 <= medium < high < critical <= 1.0:
        raise ValueError("artifact thresholds are not strictly ordered")

    # Checksums and current-environment identity are verified before deserialization.
    model = joblib.load(model_path)
    if getattr(model, "n_features_in_", None) != len(FEATURE_ORDER):
        raise ValueError("artifact input width does not match the feature contract")
    classes = np.asarray(getattr(model, "classes_", []))
    positive_positions = np.flatnonzero(classes == 1)
    if positive_positions.size != 1:
        raise ValueError("artifact positive class cannot be identified")
    probability = model.predict_proba(np.zeros((1, len(FEATURE_ORDER)), dtype=np.float64))[
        0, int(positive_positions[0])
    ]
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("artifact did not produce a finite probability")
