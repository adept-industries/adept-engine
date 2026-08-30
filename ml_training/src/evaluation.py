from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ThresholdSelection:
    medium: float
    high: float
    critical: float
    method: str

    def as_dict(self) -> dict[str, float]:
        return {"medium": self.medium, "high": self.high, "critical": self.critical}


def _best_fbeta_threshold(labels: np.ndarray, probabilities: np.ndarray, beta: float) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        raise ValueError("validation probabilities do not provide a threshold")
    precision = precision[:-1]
    recall = recall[:-1]
    beta_squared = beta**2
    denominator = beta_squared * precision + recall
    scores = np.divide(
        (1 + beta_squared) * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    # np.argmax intentionally picks the lower threshold on a score tie.
    return float(thresholds[int(np.argmax(scores))])


def select_thresholds(labels: np.ndarray, probabilities: np.ndarray) -> ThresholdSelection:
    """Select recall-balanced-to-precision-focused bands using validation only."""
    selected = (
        _best_fbeta_threshold(labels, probabilities, beta=2.0),
        _best_fbeta_threshold(labels, probabilities, beta=1.0),
        _best_fbeta_threshold(labels, probabilities, beta=0.5),
    )
    medium, high, critical = selected
    if not 0.0 <= medium < high < critical <= 1.0:
        raise ValueError(
            f"validation-selected F2/F1/F0.5 thresholds are not strictly ordered: {selected}"
        )
    return ThresholdSelection(
        medium=medium,
        high=high,
        critical=critical,
        method=(
            "Validation-only thresholds maximizing F-beta: medium=F2 (recall weighted), "
            "high=F1, critical=F0.5 (precision weighted); lower threshold wins ties."
        ),
    )


def _effort_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    lines_changed: np.ndarray,
) -> dict[str, float | None]:
    if not np.isfinite(lines_changed).all() or np.any(lines_changed <= 0):
        return {
            "recallAt20PercentLoc": None,
            "effortAt20PercentRecall": None,
            "pOpt": None,
        }

    def ordered_indices(actual: bool, ascending: bool = False) -> np.ndarray:
        density = (labels if actual else probabilities) / lines_changed
        return np.argsort(density) if ascending else np.argsort(-density)

    def recall_curve_for(order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ordered_effort = lines_changed[order]
        ordered_labels = labels[order]
        total_effort = float(ordered_effort.sum())
        total_defects = int(labels.sum())
        effort = np.concatenate(([0.0], np.cumsum(ordered_effort) / total_effort, [1.0]))
        recall = np.concatenate(([0.0], np.cumsum(ordered_labels) / total_defects, [1.0]))
        return effort, recall

    predicted_order = ordered_indices(actual=False)
    best_order = ordered_indices(actual=True)
    worst_order = ordered_indices(actual=True, ascending=True)
    predicted_effort, predicted_recall = recall_curve_for(predicted_order)
    best_effort, best_recall = recall_curve_for(best_order)
    worst_effort, worst_recall = recall_curve_for(worst_order)

    grid = np.linspace(0.0, 1.0, 101)
    predicted_area = float(np.trapezoid(np.interp(grid, predicted_effort, predicted_recall), grid))
    best_area = float(np.trapezoid(np.interp(grid, best_effort, best_recall), grid))
    worst_area = float(np.trapezoid(np.interp(grid, worst_effort, worst_recall), grid))
    p_opt = 1.0 - ((best_area - predicted_area) / (best_area - worst_area))

    cumulative_effort = np.cumsum(lines_changed[predicted_order])
    cumulative_defects = np.cumsum(labels[predicted_order])
    twenty_percent_effort = 0.2 * float(lines_changed.sum())
    within_effort = cumulative_effort <= twenty_percent_effort
    recall_at_effort = (
        float(cumulative_defects[within_effort][-1] / labels.sum()) if within_effort.any() else 0.0
    )
    target_defects = math.ceil(0.2 * int(labels.sum()))
    target_index = int(np.searchsorted(cumulative_defects, target_defects, side="left"))
    effort_at_recall = float(cumulative_effort[target_index] / lines_changed.sum())
    return {
        "recallAt20PercentLoc": recall_at_effort,
        "effortAt20PercentRecall": effort_at_recall,
        "pOpt": p_opt,
    }


def evaluate_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    classification_threshold: float,
    lines_changed: np.ndarray,
) -> dict[str, object]:
    if labels.shape != probabilities.shape or labels.shape != lines_changed.shape:
        raise ValueError("labels, probabilities, and line effort must have equal shape")
    predictions = (probabilities >= classification_threshold).astype(np.int8)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    calibration_true, calibration_predicted = calibration_curve(
        labels, probabilities, n_bins=10, strategy="quantile"
    )
    return {
        "classificationThreshold": classification_threshold,
        "rocAuc": float(roc_auc_score(labels, probabilities)),
        "prAuc": float(average_precision_score(labels, probabilities)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "brierScore": float(brier_score_loss(labels, probabilities)),
        "confusionMatrix": {
            "trueNegative": int(matrix[0, 0]),
            "falsePositive": int(matrix[0, 1]),
            "falseNegative": int(matrix[1, 0]),
            "truePositive": int(matrix[1, 1]),
        },
        "calibration": [
            {"meanPredictedProbability": float(predicted), "observedPositiveRate": float(actual)}
            for predicted, actual in zip(calibration_predicted, calibration_true, strict=True)
        ],
        "effortAware": _effort_metrics(labels, probabilities, lines_changed),
    }
