"""
Pure DORA metrics calculation routines and period bucketing.

Implements standard DORA formulas:
1. Deployment Frequency: count of successful production deployments.
2. Change Lead Time: median hours from first PR commit to deployment completion.
3. Failed Deployment Recovery Time: median hours from incident detection to recovery/resolution.
4. Change Failure Rate: percentage of production deployments that fail or cause incidents.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

CALCULATION_VERSION = "dora-v1"


@dataclass(frozen=True)
class MetricSnapshotResult:
    metric_type: str
    granularity: str
    period_start: datetime
    period_end: datetime
    value: float
    unit: str
    sample_size: int
    calculation_version: str = CALCULATION_VERSION
    dimensions: dict[str, Any] = field(default_factory=dict)


def compute_percentiles(values: list[float]) -> dict[str, float]:
    """
    Compute mean, p50 (median), p75, and p90 for a list of floats.
    Returns empty dict if values is empty.
    """
    if not values:
        return {}

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def _percentile(p: float) -> float:
        if n == 1:
            return sorted_vals[0]
        rank = p * (n - 1)
        lower_idx = int(rank)
        upper_idx = min(lower_idx + 1, n - 1)
        weight = rank - lower_idx
        return sorted_vals[lower_idx] * (1.0 - weight) + sorted_vals[upper_idx] * weight

    mean_val = sum(sorted_vals) / n
    p50_val = _percentile(0.50)
    p75_val = _percentile(0.75)
    p90_val = _percentile(0.90)

    return {
        "mean": round(mean_val, 2),
        "p50": round(p50_val, 2),
        "p75": round(p75_val, 2),
        "p90": round(p90_val, 2),
    }


def get_period_buckets(
    from_date: datetime,
    to_date: datetime,
    granularity: str,
) -> list[tuple[datetime, datetime]]:
    """
    Generate UTC period boundary tuples (start, end) covering [from_date, to_date).
    Supported granularities: 'DAY', 'WEEK', 'MONTH'.
    """
    granularity = granularity.upper()
    if from_date >= to_date:
        return []

    from_utc = from_date.astimezone(UTC)
    to_utc = to_date.astimezone(UTC)
    buckets: list[tuple[datetime, datetime]] = []

    if granularity == "DAY":
        current = datetime(from_utc.year, from_utc.month, from_utc.day, tzinfo=UTC)
        while current < to_utc:
            next_day = current + timedelta(days=1)
            buckets.append((current, next_day))
            current = next_day

    elif granularity == "WEEK":
        # Align to Monday 00:00:00 UTC
        monday = from_utc - timedelta(days=from_utc.weekday())
        current = datetime(monday.year, monday.month, monday.day, tzinfo=UTC)
        while current < to_utc:
            next_monday = current + timedelta(days=7)
            buckets.append((current, next_monday))
            current = next_monday

    elif granularity == "MONTH":
        # Align to 1st of month 00:00:00 UTC
        current = datetime(from_utc.year, from_utc.month, 1, tzinfo=UTC)
        while current < to_utc:
            _, days_in_month = calendar.monthrange(current.year, current.month)
            next_month = current + timedelta(days=days_in_month)
            buckets.append((current, next_month))
            current = next_month

    else:
        raise ValueError(f"Unsupported granularity: {granularity}")

    return buckets


def calculate_deployment_frequency(
    period_start: datetime,
    period_end: datetime,
    granularity: str,
    deployments: list[dict[str, Any]],
) -> MetricSnapshotResult:
    """
    Calculate Deployment Frequency.
    Counts production deployments with status='SUCCESS' in [period_start, period_end).
    """
    granularity_upper = granularity.upper()
    unit_map = {
        "DAY": "deployments/day",
        "WEEK": "deployments/week",
        "MONTH": "deployments/month",
    }
    unit = unit_map.get(granularity_upper, "deployments")

    successful_count = 0
    for dep in deployments:
        finished_at = dep.get("finished_at")
        is_prod = dep.get("is_production", False)
        status = dep.get("status")

        if (
            is_prod
            and status == "SUCCESS"
            and finished_at
            and period_start <= finished_at < period_end
        ):
            successful_count += 1

    return MetricSnapshotResult(
        metric_type="DEPLOYMENT_FREQUENCY",
        granularity=granularity_upper,
        period_start=period_start,
        period_end=period_end,
        value=float(successful_count),
        unit=unit,
        sample_size=successful_count,
        calculation_version=CALCULATION_VERSION,
        dimensions={},
    )


def calculate_change_lead_time(
    period_start: datetime,
    period_end: datetime,
    granularity: str,
    pr_deployments: list[dict[str, Any]],
) -> MetricSnapshotResult:
    """
    Calculate Change Lead Time in hours.
    For PRs linked to successful production deployments finishing in [period_start, period_end):
    lead_time_hours = (deployment.finished_at - pr.first_commit_at).total_seconds() / 3600.0
    """
    granularity_upper = granularity.upper()
    lead_time_hours_list: list[float] = []

    for item in pr_deployments:
        finished_at = item.get("deployment_finished_at")
        is_prod = item.get("is_production", False)
        status = item.get("deployment_status")
        commit_time = item.get("first_commit_at") or item.get("pr_opened_at")

        if (
            is_prod
            and status == "SUCCESS"
            and finished_at
            and commit_time
            and period_start <= finished_at < period_end
        ):
            lead_sec = (finished_at - commit_time).total_seconds()
            if lead_sec >= 0:
                lead_time_hours_list.append(lead_sec / 3600.0)

    sample_size = len(lead_time_hours_list)
    if sample_size == 0:
        return MetricSnapshotResult(
            metric_type="CHANGE_LEAD_TIME_HOURS",
            granularity=granularity_upper,
            period_start=period_start,
            period_end=period_end,
            value=0.0,
            unit="hours",
            sample_size=0,
            calculation_version=CALCULATION_VERSION,
            dimensions={},
        )

    percentiles = compute_percentiles(lead_time_hours_list)
    median_val = percentiles.get("p50", 0.0)

    return MetricSnapshotResult(
        metric_type="CHANGE_LEAD_TIME_HOURS",
        granularity=granularity_upper,
        period_start=period_start,
        period_end=period_end,
        value=round(median_val, 2),
        unit="hours",
        sample_size=sample_size,
        calculation_version=CALCULATION_VERSION,
        dimensions=percentiles,
    )


def calculate_recovery_time(
    period_start: datetime,
    period_end: datetime,
    granularity: str,
    incidents: list[dict[str, Any]],
) -> MetricSnapshotResult:
    """
    Calculate Failed Deployment Recovery Time (MTTR) in hours.
    For resolved incidents in [period_start, period_end):
    recovery_hours = (resolved_at - incident.detected_at).total_seconds() / 3600.0
    """
    granularity_upper = granularity.upper()
    recovery_hours_list: list[float] = []

    for inc in incidents:
        resolved_at = inc.get("recovery_finished_at") or inc.get("resolved_at")
        detected_at = inc.get("detected_at") or inc.get("created_at")

        if resolved_at and detected_at and period_start <= resolved_at < period_end:
            rec_sec = (resolved_at - detected_at).total_seconds()
            if rec_sec >= 0:
                recovery_hours_list.append(rec_sec / 3600.0)

    sample_size = len(recovery_hours_list)
    if sample_size == 0:
        return MetricSnapshotResult(
            metric_type="FAILED_DEPLOYMENT_RECOVERY_TIME_HOURS",
            granularity=granularity_upper,
            period_start=period_start,
            period_end=period_end,
            value=0.0,
            unit="hours",
            sample_size=0,
            calculation_version=CALCULATION_VERSION,
            dimensions={},
        )

    percentiles = compute_percentiles(recovery_hours_list)
    median_val = percentiles.get("p50", 0.0)

    return MetricSnapshotResult(
        metric_type="FAILED_DEPLOYMENT_RECOVERY_TIME_HOURS",
        granularity=granularity_upper,
        period_start=period_start,
        period_end=period_end,
        value=round(median_val, 2),
        unit="hours",
        sample_size=sample_size,
        calculation_version=CALCULATION_VERSION,
        dimensions=percentiles,
    )


def calculate_change_failure_rate(
    period_start: datetime,
    period_end: datetime,
    granularity: str,
    deployments: list[dict[str, Any]],
) -> MetricSnapshotResult:
    """
    Calculate Change Failure Rate as a percentage.
    change_failure_rate_percent = (failed production deployments / all production deployments) * 100
    """
    granularity_upper = granularity.upper()
    total_prod = 0
    failed_prod = 0

    for dep in deployments:
        finished_at = dep.get("finished_at")
        is_prod = dep.get("is_production", False)
        status = dep.get("status")

        if is_prod and finished_at and period_start <= finished_at < period_end:
            total_prod += 1
            if status == "FAILURE" or dep.get("has_incident", False):
                failed_prod += 1

    if total_prod == 0:
        return MetricSnapshotResult(
            metric_type="CHANGE_FAILURE_RATE_PERCENT",
            granularity=granularity_upper,
            period_start=period_start,
            period_end=period_end,
            value=0.0,
            unit="percent",
            sample_size=0,
            calculation_version=CALCULATION_VERSION,
            dimensions={"total_deployments": 0, "failed_deployments": 0},
        )

    rate = (failed_prod / total_prod) * 100.0
    return MetricSnapshotResult(
        metric_type="CHANGE_FAILURE_RATE_PERCENT",
        granularity=granularity_upper,
        period_start=period_start,
        period_end=period_end,
        value=round(rate, 2),
        unit="percent",
        sample_size=total_prod,
        calculation_version=CALCULATION_VERSION,
        dimensions={
            "total_deployments": total_prod,
            "failed_deployments": failed_prod,
        },
    )
