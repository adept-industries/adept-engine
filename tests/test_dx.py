"""Tests for Developer Experience (DX) workflow score calculations."""

from app.risk.dx import compute_dx_score


def test_dx_score_standard_range() -> None:
    res = compute_dx_score(
        median_first_review_hours=6.0,
        median_pr_cycle_hours=24.0,
        stale_pr_rate=0.05,
        change_failure_rate=0.05,
        ci_success_rate=0.98,
    )
    assert 0.0 <= res["score"] <= 100.0
    assert "components" in res
    assert "review_wait" in res["components"]
    assert "flow" in res["components"]
    assert "staleness" in res["components"]
    assert "delivery_stability" in res["components"]
    assert "ci_reliability" in res["components"]


def test_dx_score_perfect_metrics() -> None:
    res = compute_dx_score(
        median_first_review_hours=1.0,
        median_pr_cycle_hours=12.0,
        stale_pr_rate=0.0,
        change_failure_rate=0.0,
        ci_success_rate=1.0,
    )
    assert res["score"] == 100.0


def test_dx_score_custom_weights() -> None:
    custom_weights = {
        "review_wait": 0.50,
        "flow": 0.50,
        "staleness": 0.0,
        "delivery_stability": 0.0,
        "ci_reliability": 0.0,
    }
    res = compute_dx_score(
        median_first_review_hours=12.0,
        median_pr_cycle_hours=72.0,
        stale_pr_rate=0.50,
        change_failure_rate=0.50,
        ci_success_rate=0.0,
        custom_weights=custom_weights,
    )
    assert res["score"] == 100.0
