"""
Pure code-review cycle-time calculation.

A merged pull request's life is split into contiguous stages:

    first commit -> ready for review -> first review -> merge -> deploy
         coding            pickup              review      deploy

Every stage is measured per PR and stored as a pooled observation keyed by the
PR, timestamped at ``merged_at``. All stages therefore describe the same weekly
cohort (PRs merged that week), which keeps a stacked bar of medians coherent.
A stage is omitted for a PR when either boundary is unknown or out of order.
Every observation also carries per-PR facts (size, time to merge, whether it
was reviewed), so the API can break PRs down by size from any stage.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.metrics.calculator import MetricSnapshotResult, compute_percentiles

CYCLE_TIME_CALCULATION_VERSION = "cycle-time-v2"

STAGE_METRIC_TYPES = {
    "coding": "PR_CODING_TIME_HOURS",
    "pickup": "PR_PICKUP_TIME_HOURS",
    "review": "PR_REVIEW_TIME_HOURS",
    "deploy": "PR_DEPLOY_TIME_HOURS",
}

# Upper bounds of changed lines (additions + deletions) per size bucket.
_SIZE_BUCKETS = (("S", 100), ("M", 400), ("L", 1000))


def size_bucket(changed_lines: int) -> str:
    for label, upper in _SIZE_BUCKETS:
        if changed_lines <= upper:
            return label
    return "XL"


def pickup_start(pull_request: dict[str, Any]) -> datetime | None:
    """Review is awaited from the latest ready_for_review transition, else from opening.

    A transition after the first review describes a PR that was reviewed while
    still a draft, so opening remains the fairer start in that case.
    """
    opened_at: datetime | None = pull_request.get("opened_at")
    ready_at: datetime | None = pull_request.get("ready_for_review_at")
    first_review_at: datetime | None = pull_request.get("first_review_at")
    if (
        opened_at is not None
        and ready_at is not None
        and ready_at > opened_at
        and (first_review_at is None or ready_at <= first_review_at)
    ):
        return ready_at
    return opened_at


def pull_request_stage_hours(pull_request: dict[str, Any]) -> dict[str, float]:
    """Return the measurable stage durations in hours for one merged PR."""
    start = pickup_start(pull_request)
    boundaries = {
        "coding": (pull_request.get("first_commit_at"), start),
        "pickup": (start, pull_request.get("first_review_at")),
        "review": (pull_request.get("first_review_at"), pull_request.get("merged_at")),
        "deploy": (pull_request.get("merged_at"), pull_request.get("deployed_at")),
    }
    stages: dict[str, float] = {}
    for stage, (begin, end) in boundaries.items():
        hours = _hours_between(begin, end)
        if hours is not None:
            stages[stage] = hours
    return stages


def _hours_between(begin: datetime | None, end: datetime | None) -> float | None:
    if begin is None or end is None:
        return None
    seconds = (end - begin).total_seconds()
    return round(seconds / 3600.0, 4) if seconds >= 0 else None


def calculate_cycle_time_stages(
    period_start: datetime,
    period_end: datetime,
    granularity: str,
    pull_requests: list[dict[str, Any]],
) -> list[MetricSnapshotResult]:
    """Build one snapshot per stage for PRs merged in ``[period_start, period_end)``."""
    granularity_upper = granularity.upper()
    observations: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGE_METRIC_TYPES}

    for pull_request in pull_requests:
        merged_at: datetime | None = pull_request.get("merged_at")
        if merged_at is None or not period_start <= merged_at < period_end:
            continue
        size = size_bucket(
            int(pull_request.get("additions") or 0) + int(pull_request.get("deletions") or 0)
        )
        # Open until merged, whether or not anyone reviewed it.
        merge_hours = _hours_between(pickup_start(pull_request), merged_at)
        reviewed = pull_request.get("first_review_at") is not None
        for stage, hours in pull_request_stage_hours(pull_request).items():
            observations[stage].append(
                {
                    "key": str(pull_request["id"]),
                    "at": merged_at.isoformat(),
                    "value": hours,
                    "size": size,
                    "merge_hours": merge_hours,
                    "reviewed": reviewed,
                }
            )

    snapshots: list[MetricSnapshotResult] = []
    for stage, metric_type in STAGE_METRIC_TYPES.items():
        stage_observations = observations[stage]
        percentiles = compute_percentiles([float(item["value"]) for item in stage_observations])
        snapshots.append(
            MetricSnapshotResult(
                metric_type=metric_type,
                granularity=granularity_upper,
                period_start=period_start,
                period_end=period_end,
                value=round(percentiles.get("p50", 0.0), 2),
                unit="hours",
                sample_size=len(stage_observations),
                calculation_version=CYCLE_TIME_CALCULATION_VERSION,
                dimensions={**percentiles, "observations": stage_observations},
            )
        )
    return snapshots
