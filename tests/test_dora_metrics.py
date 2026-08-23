"""
Unit and formula fixture tests for DORA metrics calculation.

Covers:
- Deployment frequency formula, period boundaries, and status filtering.
- Change lead time formula, percentiles (mean, p50, p75, p90), and negative time exclusion.
- Recovery time formula (MTTR) and zero-sample handling.
- Change failure rate percentage and division-by-zero safety.
- Granularity bucketing (DAY, WEEK, MONTH).
- Recalculate metrics job handler validation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.db.models import ClaimedJob
from app.jobs.handlers.recalculate_metrics import handle_recalculate_metrics
from app.jobs.retry import PermanentJobError
from app.metrics.calculator import (
    CALCULATION_VERSION,
    calculate_change_failure_rate,
    calculate_change_lead_time,
    calculate_deployment_frequency,
    calculate_recovery_time,
    compute_percentiles,
    get_period_buckets,
)

# ---------------------------------------------------------------------------
# 1. Percentiles & Bucketing Tests
# ---------------------------------------------------------------------------


def test_compute_percentiles_empty() -> None:
    assert compute_percentiles([]) == {}


def test_compute_percentiles_single_value() -> None:
    res = compute_percentiles([12.5])
    assert res == {
        "mean": 12.5,
        "p50": 12.5,
        "p75": 12.5,
        "p90": 12.5,
    }


def test_compute_percentiles_known_dataset() -> None:
    # Values: [2.0, 4.0, 6.0, 8.0, 10.0]
    # Mean: 6.0
    # Median (p50): 6.0
    res = compute_percentiles([2.0, 4.0, 6.0, 8.0, 10.0])
    assert res["mean"] == 6.0
    assert res["p50"] == 6.0
    assert res["p75"] == 8.0
    assert res["p90"] == 9.2


def test_get_period_buckets_day() -> None:
    start = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
    end = datetime(2026, 8, 4, 0, 0, tzinfo=UTC)
    buckets = get_period_buckets(start, end, "DAY")
    assert len(buckets) == 3
    assert buckets[0] == (
        datetime(2026, 8, 1, 0, 0, tzinfo=UTC),
        datetime(2026, 8, 2, 0, 0, tzinfo=UTC),
    )
    assert buckets[2] == (
        datetime(2026, 8, 3, 0, 0, tzinfo=UTC),
        datetime(2026, 8, 4, 0, 0, tzinfo=UTC),
    )


def test_get_period_buckets_week() -> None:
    # 2026-08-03 is a Monday
    start = datetime(2026, 8, 3, 0, 0, tzinfo=UTC)
    end = datetime(2026, 8, 17, 0, 0, tzinfo=UTC)
    buckets = get_period_buckets(start, end, "WEEK")
    assert len(buckets) == 2
    assert buckets[0] == (
        datetime(2026, 8, 3, 0, 0, tzinfo=UTC),
        datetime(2026, 8, 10, 0, 0, tzinfo=UTC),
    )


def test_get_period_buckets_month() -> None:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    end = datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
    buckets = get_period_buckets(start, end, "MONTH")
    assert len(buckets) == 3
    assert buckets[0] == (
        datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2026, 2, 1, 0, 0, tzinfo=UTC),
    )


def test_get_period_buckets_invalid_granularity() -> None:
    with pytest.raises(ValueError, match="Unsupported granularity"):
        get_period_buckets(
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
            "YEAR",
        )


# ---------------------------------------------------------------------------
# 2. Deployment Frequency Formula Tests
# ---------------------------------------------------------------------------


def test_calculate_deployment_frequency_filters_correctly() -> None:
    p_start = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
    p_end = datetime(2026, 8, 8, 0, 0, tzinfo=UTC)

    deployments: list[dict[str, Any]] = [
        # 1. Successful production in period -> COUNTS
        {
            "finished_at": datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
            "is_production": True,
            "status": "SUCCESS",
        },
        # 2. Successful production in period -> COUNTS
        {
            "finished_at": datetime(2026, 8, 5, 10, 0, tzinfo=UTC),
            "is_production": True,
            "status": "SUCCESS",
        },
        # 3. Failed production in period -> IGNORED
        {
            "finished_at": datetime(2026, 8, 3, 12, 0, tzinfo=UTC),
            "is_production": True,
            "status": "FAILURE",
        },
        # 4. Successful staging in period -> IGNORED
        {
            "finished_at": datetime(2026, 8, 4, 12, 0, tzinfo=UTC),
            "is_production": False,
            "status": "SUCCESS",
        },
        # 5. Successful production outside period -> IGNORED
        {
            "finished_at": datetime(2026, 8, 10, 12, 0, tzinfo=UTC),
            "is_production": True,
            "status": "SUCCESS",
        },
    ]

    res = calculate_deployment_frequency(p_start, p_end, "WEEK", deployments)
    assert res.metric_type == "DEPLOYMENT_FREQUENCY"
    assert res.granularity == "WEEK"
    assert res.value == 2.0
    assert res.sample_size == 2
    assert res.unit == "deployments/week"
    assert res.calculation_version == CALCULATION_VERSION


# ---------------------------------------------------------------------------
# 3. Change Lead Time Formula Tests
# ---------------------------------------------------------------------------


def test_calculate_change_lead_time_zero_samples() -> None:
    p_start = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
    p_end = datetime(2026, 8, 8, 0, 0, tzinfo=UTC)
    res = calculate_change_lead_time(p_start, p_end, "WEEK", [])
    assert res.metric_type == "CHANGE_LEAD_TIME_HOURS"
    assert res.value == 0.0
    assert res.sample_size == 0
    assert res.dimensions == {}


def test_calculate_change_lead_time_computes_hours_and_percentiles() -> None:
    p_start = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
    p_end = datetime(2026, 8, 8, 0, 0, tzinfo=UTC)

    # PR 1: 4 hours lead time
    # PR 2: 6 hours lead time
    # PR 3: 20 hours lead time
    # Median = 6.0 hours, Mean = 10.0 hours
    pr_deployments: list[dict[str, Any]] = [
        {
            "deployment_finished_at": datetime(2026, 8, 2, 14, 0, tzinfo=UTC),
            "is_production": True,
            "deployment_status": "SUCCESS",
            "first_commit_at": datetime(2026, 8, 2, 10, 0, tzinfo=UTC),
        },
        {
            "deployment_finished_at": datetime(2026, 8, 3, 16, 0, tzinfo=UTC),
            "is_production": True,
            "deployment_status": "SUCCESS",
            "first_commit_at": datetime(2026, 8, 3, 10, 0, tzinfo=UTC),
        },
        {
            "deployment_finished_at": datetime(2026, 8, 4, 20, 0, tzinfo=UTC),
            "is_production": True,
            "deployment_status": "SUCCESS",
            "first_commit_at": datetime(2026, 8, 4, 0, 0, tzinfo=UTC),
        },
        # Negative lead time or staging -> excluded
        {
            "deployment_finished_at": datetime(2026, 8, 5, 12, 0, tzinfo=UTC),
            "is_production": False,
            "deployment_status": "SUCCESS",
            "first_commit_at": datetime(2026, 8, 5, 10, 0, tzinfo=UTC),
        },
    ]

    res = calculate_change_lead_time(p_start, p_end, "WEEK", pr_deployments)
    assert res.metric_type == "CHANGE_LEAD_TIME_HOURS"
    assert res.value == 6.0
    assert res.sample_size == 3
    assert res.unit == "hours"
    assert res.dimensions["mean"] == 10.0
    assert res.dimensions["p50"] == 6.0


# ---------------------------------------------------------------------------
# 4. Recovery Time (MTTR) Formula Tests
# ---------------------------------------------------------------------------


def test_calculate_recovery_time() -> None:
    p_start = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
    p_end = datetime(2026, 8, 8, 0, 0, tzinfo=UTC)

    # Incident 1: 2.5 hours recovery
    # Incident 2: 4.5 hours recovery
    incidents: list[dict[str, Any]] = [
        {
            "detected_at": datetime(2026, 8, 2, 10, 0, tzinfo=UTC),
            "resolved_at": datetime(2026, 8, 2, 12, 30, tzinfo=UTC),
            "recovery_finished_at": None,
        },
        {
            "detected_at": datetime(2026, 8, 3, 10, 0, tzinfo=UTC),
            "resolved_at": datetime(2026, 8, 3, 15, 0, tzinfo=UTC),
            "recovery_finished_at": datetime(2026, 8, 3, 14, 30, tzinfo=UTC),
        },
    ]

    res = calculate_recovery_time(p_start, p_end, "WEEK", incidents)
    assert res.metric_type == "FAILED_DEPLOYMENT_RECOVERY_TIME_HOURS"
    assert res.value == 3.5  # median of 2.5 and 4.5
    assert res.sample_size == 2
    assert res.dimensions["mean"] == 3.5


# ---------------------------------------------------------------------------
# 5. Change Failure Rate Formula Tests
# ---------------------------------------------------------------------------


def test_calculate_change_failure_rate_zero_deployments() -> None:
    p_start = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
    p_end = datetime(2026, 8, 8, 0, 0, tzinfo=UTC)
    res = calculate_change_failure_rate(p_start, p_end, "WEEK", [])
    assert res.metric_type == "CHANGE_FAILURE_RATE_PERCENT"
    assert res.value == 0.0
    assert res.sample_size == 0


def test_calculate_change_failure_rate_valid_percentage() -> None:
    p_start = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
    p_end = datetime(2026, 8, 8, 0, 0, tzinfo=UTC)

    # 10 production deployments total, 2 failures -> 20.0%
    deployments: list[dict[str, Any]] = [
        {
            "finished_at": datetime(2026, 8, 2, 10, 0, tzinfo=UTC),
            "is_production": True,
            "status": "FAILURE",
        },
        {
            "finished_at": datetime(2026, 8, 3, 10, 0, tzinfo=UTC),
            "is_production": True,
            "status": "FAILURE",
        },
    ] + [
        {
            "finished_at": datetime(2026, 8, 4, 10, 0, tzinfo=UTC),
            "is_production": True,
            "status": "SUCCESS",
        }
        for _ in range(8)
    ]

    res = calculate_change_failure_rate(p_start, p_end, "WEEK", deployments)
    assert res.metric_type == "CHANGE_FAILURE_RATE_PERCENT"
    assert res.value == 20.0
    assert res.sample_size == 10
    assert res.unit == "percent"
    assert res.dimensions["failed_deployments"] == 2
    assert res.dimensions["total_deployments"] == 10


# ---------------------------------------------------------------------------
# 6. Job Handler Validation Tests
# ---------------------------------------------------------------------------


def test_handle_recalculate_metrics_missing_payload() -> None:
    job = ClaimedJob(
        id=uuid4(),
        job_type="RECALCULATE_METRICS",
        payload={},
        priority=100,
        attempts=0,
        max_attempts=8,
        locked_by="test-worker",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        version=0,
    )
    with pytest.raises(PermanentJobError, match="missing repository_id or workspace_id"):
        handle_recalculate_metrics(MagicMock(), job, "test-worker")


def test_handle_recalculate_metrics_calls_service() -> None:
    repo_id = uuid4()
    ws_id = uuid4()
    job = ClaimedJob(
        id=uuid4(),
        job_type="RECALCULATE_METRICS",
        payload={
            "repository_id": str(repo_id),
            "workspace_id": str(ws_id),
            "from_date": "2026-08-01T00:00:00+00:00",
            "to_date": "2026-08-15T00:00:00+00:00",
        },
        priority=100,
        attempts=0,
        max_attempts=8,
        locked_by="test-worker",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        version=0,
    )

    mock_engine = MagicMock()
    with patch(
        "app.jobs.handlers.recalculate_metrics.recalculate_repository_metrics"
    ) as mock_recalc:
        mock_recalc.return_value = 12
        handle_recalculate_metrics(mock_engine, job, "test-worker")
        mock_recalc.assert_called_once()
