"""Developer Experience (DX) Score Calculation.

Provides a transparent, team/repository-level 0-100 workflow score.
Never used to rank individual developers.
"""

import math
from typing import Any

DEFAULT_DX_WEIGHTS = {
    "review_wait": 0.25,
    "flow": 0.20,
    "staleness": 0.20,
    "delivery_stability": 0.20,
    "ci_reliability": 0.15,
}


def _lower_better(value: float, target: float) -> float:
    """Exponential decay score for metrics where lower is better."""
    if value <= target:
        return 100.0
    return max(0.0, min(100.0, 100.0 * math.exp(-0.45 * max(0.0, value / max(1e-6, target) - 1.0))))


def _rate_lower_better(rate: float, target: float) -> float:
    """Linear scaling score for rate metrics where lower is better."""
    if rate <= target:
        return 100.0
    return max(0.0, min(100.0, 100.0 * (1.0 - (rate - target) / max(1e-6, 1.0 - target))))


def compute_dx_score(
    median_first_review_hours: float,
    median_pr_cycle_hours: float,
    stale_pr_rate: float,
    change_failure_rate: float,
    ci_success_rate: float,
    custom_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Computes transparent 0..100 Developer Experience score and component breakdown."""
    components = {
        "review_wait": _lower_better(median_first_review_hours, target=12.0),
        "flow": _lower_better(median_pr_cycle_hours, target=72.0),
        "staleness": _rate_lower_better(stale_pr_rate, target=0.10),
        "delivery_stability": _rate_lower_better(change_failure_rate, target=0.15),
        "ci_reliability": max(0.0, min(100.0, ci_success_rate * 100.0)),
    }

    weights = custom_weights or DEFAULT_DX_WEIGHTS
    # Normalize weights to sum to 1.0
    total_weight = sum(weights.get(k, DEFAULT_DX_WEIGHTS.get(k, 0.0)) for k in components)
    if total_weight <= 0:
        total_weight = 1.0

    normalized_weights = {
        k: weights.get(k, DEFAULT_DX_WEIGHTS.get(k, 0.0)) / total_weight for k in components
    }

    score = sum(components[k] * normalized_weights[k] for k in components)

    return {
        "score": round(score, 1),
        "components": {k: round(v, 1) for k, v in components.items()},
        "weights": {k: round(v, 3) for k, v in normalized_weights.items()},
    }
