"""Verified inference runtime for the approved seven-feature model artifact."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
import structlog

from app.risk.features import FEATURE_ORDER, FEATURE_SCHEMA_VERSION, PullRequestRiskFeatures

MODEL_NAME = "jitfine-expert-pr-risk-mvp"
MODEL_VERSION = "jitfine-expert-pr-risk-mvp-v1"
DEFAULT_ARTIFACT_DIRECTORY = Path(__file__).resolve().parent / "artifacts" / MODEL_VERSION

logger = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class RiskPredictionResult:
    probability: float
    level: str
    threshold_used: float
    top_factors: list[dict[str, Any]]


class JitFineRiskModel:
    def __init__(self, artifact_directory: Path | str = DEFAULT_ARTIFACT_DIRECTORY) -> None:
        self.artifact_directory = Path(artifact_directory)
        self.model: Any | None = None
        self.metadata: dict[str, Any] | None = None
        self.positive_class_position: int | None = None

    @property
    def ready(self) -> bool:
        return (
            self.model is not None
            and self.metadata is not None
            and self.positive_class_position is not None
        )

    def load(self) -> None:
        metadata_path = self.artifact_directory / "metadata.json"
        model_path = self.artifact_directory / "model.joblib"
        report_path = self.artifact_directory / "training-report.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self._verify_metadata(metadata, model_path, report_path)

        model = joblib.load(model_path)
        if int(getattr(model, "n_features_in_", -1)) != len(FEATURE_ORDER):
            raise RuntimeError("PR-risk artifact input width does not match its contract")
        classes = np.asarray(getattr(model, "classes_", []))
        positive_positions = np.flatnonzero(classes == 1)
        if positive_positions.size != 1:
            raise RuntimeError("PR-risk artifact positive class cannot be identified")

        self.model = model
        self.metadata = metadata
        self.positive_class_position = int(positive_positions[0])
        logger.info(
            "pr_risk_model_loaded",
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
        )

    def predict(self, features: PullRequestRiskFeatures) -> RiskPredictionResult:
        if not self.ready:
            self.load()
        if self.model is None or self.metadata is None or self.positive_class_position is None:
            raise RuntimeError("PR-risk model did not initialize")

        row = np.asarray([features.as_ordered_values()], dtype=np.float64)
        probabilities = np.asarray(self.model.predict_proba(row), dtype=np.float64)
        probability = float(probabilities[0, self.positive_class_position])
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError("PR-risk model returned an invalid probability")

        thresholds = self._thresholds()
        level, threshold_used = _classify_probability(probability, thresholds)
        importances = np.asarray(getattr(self.model, "feature_importances_", []), dtype=float)
        top_factors = _global_importance_factors(features, importances)
        return RiskPredictionResult(probability, level, threshold_used, top_factors)

    def _verify_metadata(
        self,
        metadata: dict[str, Any],
        model_path: Path,
        report_path: Path,
    ) -> None:
        expected = {
            "modelName": MODEL_NAME,
            "modelVersion": MODEL_VERSION,
            "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
            "featureOrder": list(FEATURE_ORDER),
            "scikitLearnVersion": sklearn.__version__,
            "numpyVersion": np.__version__,
            "joblibVersion": joblib.__version__,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(f"PR-risk artifact metadata mismatch for {key}")
        artifact_python_version = metadata.get("pythonVersion")
        if not _same_python_minor(artifact_python_version, platform.python_version()):
            raise RuntimeError("PR-risk artifact metadata mismatch for pythonVersion")
        checksums = metadata.get("sha256")
        if not isinstance(checksums, dict):
            raise RuntimeError("PR-risk artifact checksums are missing")
        for filename, path in (("model.joblib", model_path), ("training-report.json", report_path)):
            if checksums.get(filename) != _sha256(path):
                raise RuntimeError(f"PR-risk artifact checksum mismatch for {filename}")
        self._validated_thresholds(metadata)

    def _thresholds(self) -> dict[str, float]:
        if self.metadata is None:
            raise RuntimeError("PR-risk model metadata is not loaded")
        return self._validated_thresholds(self.metadata)

    @staticmethod
    def _validated_thresholds(metadata: dict[str, Any]) -> dict[str, float]:
        raw = metadata.get("thresholds")
        if not isinstance(raw, dict):
            raise RuntimeError("PR-risk artifact thresholds are missing")
        try:
            values = {name: float(raw[name]) for name in ("medium", "high", "critical")}
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("PR-risk artifact thresholds are invalid") from exc
        if not 0.0 <= values["medium"] < values["high"] < values["critical"] <= 1.0:
            raise RuntimeError("PR-risk artifact thresholds are not strictly ordered")
        return values


def _classify_probability(probability: float, thresholds: dict[str, float]) -> tuple[str, float]:
    if probability >= thresholds["critical"]:
        return "CRITICAL", thresholds["critical"]
    if probability >= thresholds["high"]:
        return "HIGH", thresholds["high"]
    if probability >= thresholds["medium"]:
        return "MEDIUM", thresholds["medium"]
    return "LOW", thresholds["medium"]


def _global_importance_factors(
    features: PullRequestRiskFeatures,
    importances: np.ndarray[Any, Any],
) -> list[dict[str, Any]]:
    if importances.shape != (len(FEATURE_ORDER),):
        return []
    values = features.as_dict()
    ranked = sorted(
        zip(FEATURE_ORDER, importances, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:3]
    return [
        {
            "feature": name,
            "value": values[name],
            "globalImportance": round(float(importance), 6),
            "explanationType": "global_model_importance",
        }
        for name, importance in ranked
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_python_minor(artifact_version: object, runtime_version: str) -> bool:
    if not isinstance(artifact_version, str):
        return False
    try:
        artifact_parts = tuple(int(part) for part in artifact_version.split(".")[:2])
        runtime_parts = tuple(int(part) for part in runtime_version.split(".")[:2])
    except ValueError:
        return False
    return len(artifact_parts) == 2 and artifact_parts == runtime_parts


risk_model = JitFineRiskModel()
