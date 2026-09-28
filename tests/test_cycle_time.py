"""Unit tests for code-review cycle-time stages."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.metrics.cycle_time import (
    CYCLE_TIME_CALCULATION_VERSION,
    calculate_cycle_time_stages,
    pickup_start,
    pull_request_stage_hours,
    size_bucket,
)

MONDAY = datetime(2026, 9, 21, tzinfo=UTC)


def _pull_request(**overrides: Any) -> dict[str, Any]:
    # Mon 09:00 first commit, 15:00 opened, Wed 11:00 first review,
    # Wed 17:00 merged, Thu 10:00 deployed.
    base: dict[str, Any] = {
        "id": "pr-42",
        "first_commit_at": MONDAY + timedelta(hours=9),
        "opened_at": MONDAY + timedelta(hours=15),
        "ready_for_review_at": None,
        "first_review_at": MONDAY + timedelta(days=2, hours=11),
        "merged_at": MONDAY + timedelta(days=2, hours=17),
        "deployed_at": MONDAY + timedelta(days=3, hours=10),
        "additions": 120,
        "deletions": 30,
    }
    base.update(overrides)
    return base


def test_stage_hours_follow_the_pull_request_timeline() -> None:
    assert pull_request_stage_hours(_pull_request()) == {
        "coding": 6.0,
        "pickup": 44.0,
        "review": 6.0,
        "deploy": 17.0,
    }


def test_ready_for_review_moves_draft_time_from_pickup_to_coding() -> None:
    ready_at = MONDAY + timedelta(days=1, hours=15)
    stages = pull_request_stage_hours(_pull_request(ready_for_review_at=ready_at))
    assert stages["coding"] == 30.0
    assert stages["pickup"] == 20.0


def test_ready_after_first_review_keeps_opening_as_pickup_start() -> None:
    pull_request = _pull_request(ready_for_review_at=MONDAY + timedelta(days=3))
    assert pickup_start(pull_request) == pull_request["opened_at"]


def test_unreviewed_pull_request_only_reports_measurable_stages() -> None:
    stages = pull_request_stage_hours(_pull_request(first_review_at=None, deployed_at=None))
    assert stages == {"coding": 6.0}


def test_out_of_order_boundaries_are_excluded() -> None:
    # A rebased commit authored after the PR opened must not produce negative coding time.
    stages = pull_request_stage_hours(_pull_request(first_commit_at=MONDAY + timedelta(days=1)))
    assert "coding" not in stages
    assert stages["pickup"] == 44.0


@pytest.mark.parametrize(
    ("changed_lines", "expected"),
    [(0, "S"), (100, "S"), (101, "M"), (400, "M"), (1000, "L"), (1001, "XL")],
)
def test_size_bucket_boundaries(changed_lines: int, expected: str) -> None:
    assert size_bucket(changed_lines) == expected


def test_stage_snapshots_pool_prs_merged_in_the_period() -> None:
    week_end = MONDAY + timedelta(days=7)
    pull_requests = [
        _pull_request(),
        _pull_request(
            id="pr-43",
            first_review_at=MONDAY + timedelta(days=1, hours=15),
        ),
        # Merged without a review: no pickup or review, but time to merge still counts.
        _pull_request(id="pr-45", first_review_at=None),
        # Merged next week, so it belongs to another cohort.
        _pull_request(id="pr-44", merged_at=week_end + timedelta(hours=1)),
    ]

    snapshots = {
        snapshot.metric_type: snapshot
        for snapshot in calculate_cycle_time_stages(MONDAY, week_end, "week", pull_requests)
    }

    assert set(snapshots) == {
        "PR_CODING_TIME_HOURS",
        "PR_PICKUP_TIME_HOURS",
        "PR_REVIEW_TIME_HOURS",
        "PR_DEPLOY_TIME_HOURS",
    }
    pickup = snapshots["PR_PICKUP_TIME_HOURS"]
    assert pickup.granularity == "WEEK"
    assert pickup.calculation_version == CYCLE_TIME_CALCULATION_VERSION
    assert pickup.sample_size == 2
    assert pickup.value == 34.0  # median of 44h and 24h
    assert pickup.dimensions["observations"][0] == {
        "key": "pr-42",
        "at": (MONDAY + timedelta(days=2, hours=17)).isoformat(),
        "value": 44.0,
        "size": "M",
        "merge_hours": 50.0,
        "reviewed": True,
    }
    coding = snapshots["PR_CODING_TIME_HOURS"]
    assert coding.sample_size == 3
    assert coding.dimensions["observations"][2]["reviewed"] is False


def test_empty_period_produces_zero_sample_snapshots() -> None:
    snapshots = calculate_cycle_time_stages(MONDAY, MONDAY + timedelta(days=1), "DAY", [])
    assert len(snapshots) == 4
    assert all(snapshot.sample_size == 0 and snapshot.value == 0.0 for snapshot in snapshots)
    assert all(snapshot.dimensions == {"observations": []} for snapshot in snapshots)
