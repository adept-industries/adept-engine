"""Tests for post-merge PR adverse outcome evaluation."""

from datetime import UTC, datetime

from app.risk.outcomes import evaluate_pr_outcome


def test_outcome_explicit_revert() -> None:
    merged_at = datetime(2025, 5, 1, 10, 0, 0, tzinfo=UTC)
    pr_row = {"number": 100, "merged_at": merged_at}

    later_prs = [
        {
            "number": 105,
            "title": 'Revert "Feature X" (reverts #100)',
            "opened_at": datetime(2025, 5, 3, 12, 0, 0, tzinfo=UTC),  # 2 days later
        }
    ]

    res = evaluate_pr_outcome(pr_row, later_prs=later_prs)
    assert res["is_risky"] is True
    assert res["reason"] == "EXPLICIT_REVERT"
    assert res["evidence"]["revert_pr_number"] == 105


def test_outcome_revert_outside_observation_window() -> None:
    merged_at = datetime(2025, 5, 1, 10, 0, 0, tzinfo=UTC)
    pr_row = {"number": 100, "merged_at": merged_at}

    later_prs = [
        {
            "number": 120,
            "title": 'Revert "Feature X" (reverts #100)',
            "opened_at": datetime(2025, 5, 20, 12, 0, 0, tzinfo=UTC),  # 19 days later > 14d
        }
    ]

    res = evaluate_pr_outcome(pr_row, later_prs=later_prs, observation_window_days=14)
    assert res["is_risky"] is False
    assert res["reason"] == "NO_ADVERSE_EVENT_OBSERVED"


def test_outcome_failed_production_deployment() -> None:
    merged_at = datetime(2025, 5, 1, 10, 0, 0, tzinfo=UTC)
    pr_row = {"number": 100, "merged_at": merged_at}

    deployments = [
        {
            "id": "dep-123",
            "is_production": True,
            "status": "FAILURE",
            "environment": "production",
            "finished_at": datetime(2025, 5, 1, 11, 0, 0, tzinfo=UTC),
        }
    ]

    res = evaluate_pr_outcome(pr_row, deployments=deployments)
    assert res["is_risky"] is True
    assert res["reason"] == "FAILED_DEPLOYMENT"


def test_outcome_clean_observation() -> None:
    merged_at = datetime(2025, 5, 1, 10, 0, 0, tzinfo=UTC)
    pr_row = {"number": 100, "merged_at": merged_at}

    res = evaluate_pr_outcome(pr_row)
    assert res["is_risky"] is False
    assert res["reason"] == "NO_ADVERSE_EVENT_OBSERVED"
