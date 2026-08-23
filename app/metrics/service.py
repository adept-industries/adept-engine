"""
Database-backed DORA metrics service and snapshot upsert pipeline.

Calculates repository snapshots from normalized tables and upserts them
into ``metric_snapshots``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import Engine, text

from app.metrics.calculator import (
    MetricSnapshotResult,
    calculate_change_failure_rate,
    calculate_change_lead_time,
    calculate_deployment_frequency,
    calculate_recovery_time,
    get_period_buckets,
)

logger = structlog.get_logger()


def link_deployments_to_pull_requests(
    database_engine: Engine,
    repository_id: UUID,
) -> int:
    """
    Auto-link any unlinked deployments to pull requests by matching commit SHAs.
    Matches deployment commit_sha with pull_request merge_commit_sha or head_sha.
    """
    with database_engine.begin() as connection:
        result = connection.execute(
            text(
                """
                INSERT INTO deployment_pull_requests (
                    deployment_id, pull_request_id, link_method, created_at
                )
                SELECT d.id, pr.id, 'MERGE_SHA', now()
                FROM deployments d
                JOIN pull_requests pr ON pr.repository_id = d.repository_id
                WHERE d.repository_id = :repository_id
                  AND (d.commit_sha = pr.merge_commit_sha OR d.commit_sha = pr.head_sha)
                  AND d.commit_sha IS NOT NULL
                  AND d.commit_sha <> ''
                ON CONFLICT (deployment_id, pull_request_id) DO NOTHING
                """
            ),
            {"repository_id": str(repository_id)},
        )
        return int(result.rowcount or 0)


def recalculate_repository_metrics(
    database_engine: Engine,
    workspace_id: UUID,
    repository_id: UUID,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
) -> int:
    """
    Recalculate and upsert DORA metric snapshots for a repository across DAY, WEEK, and MONTH.
    Defaults to the past 90 days up to tomorrow UTC.
    """
    # 1. Link any new deployments to PRs
    link_deployments_to_pull_requests(database_engine, repository_id)

    now_utc = datetime.now(UTC)
    if to_date is None:
        to_date = now_utc + timedelta(days=1)
    if from_date is None:
        from_date = now_utc - timedelta(days=90)

    # 2. Fetch raw normalized data for the repository
    with database_engine.connect() as connection:
        deployments_rows = (
            connection.execute(
                text(
                    """
                    SELECT id, is_production, status, finished_at, started_at, commit_sha
                    FROM deployments
                    WHERE repository_id = :repository_id
                      AND finished_at IS NOT NULL
                    ORDER BY finished_at ASC
                    """
                ),
                {"repository_id": str(repository_id)},
            )
            .mappings()
            .all()
        )

        deployments: list[dict[str, Any]] = [
            {
                "id": row["id"],
                "is_production": row["is_production"],
                "status": row["status"],
                "finished_at": row["finished_at"],
                "started_at": row["started_at"],
                "commit_sha": row["commit_sha"],
            }
            for row in deployments_rows
        ]

        pr_dep_rows = (
            connection.execute(
                text(
                    """
                    SELECT d.id AS deployment_id, d.is_production, d.status AS deployment_status,
                           d.finished_at AS deployment_finished_at,
                           pr.id AS pr_id, pr.first_commit_at, pr.opened_at AS pr_opened_at
                    FROM deployments d
                    JOIN deployment_pull_requests dpr ON dpr.deployment_id = d.id
                    JOIN pull_requests pr ON pr.id = dpr.pull_request_id
                    WHERE d.repository_id = :repository_id
                      AND d.finished_at IS NOT NULL
                    """
                ),
                {"repository_id": str(repository_id)},
            )
            .mappings()
            .all()
        )

        pr_deployments: list[dict[str, Any]] = [
            {
                "deployment_id": row["deployment_id"],
                "is_production": row["is_production"],
                "deployment_status": row["deployment_status"],
                "deployment_finished_at": row["deployment_finished_at"],
                "pr_id": row["pr_id"],
                "first_commit_at": row["first_commit_at"],
                "pr_opened_at": row["pr_opened_at"],
            }
            for row in pr_dep_rows
        ]

        incident_rows = (
            connection.execute(
                text(
                    """
                    SELECT ji.id, ji.is_incident, ji.jira_created_at AS created_at,
                           ji.jira_created_at AS detected_at, ji.resolved_at,
                           NULL AS recovery_finished_at
                    FROM jira_issues ji
                    JOIN repository_jira_projects rjp ON rjp.jira_project_id = ji.jira_project_id
                    WHERE rjp.repository_id = :repository_id
                      AND ji.is_incident = true
                      AND ji.resolved_at IS NOT NULL
                    """
                ),
                {"repository_id": str(repository_id)},
            )
            .mappings()
            .all()
        )

        incidents: list[dict[str, Any]] = [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "detected_at": row["detected_at"],
                "resolved_at": row["resolved_at"],
                "recovery_finished_at": row["recovery_finished_at"],
            }
            for row in incident_rows
        ]

    # 3. Calculate snapshots for DAY, WEEK, MONTH
    snapshots_to_upsert: list[MetricSnapshotResult] = []

    for granularity in ("DAY", "WEEK", "MONTH"):
        buckets = get_period_buckets(from_date, to_date, granularity)
        for p_start, p_end in buckets:
            # 1. Deployment Frequency
            snapshots_to_upsert.append(
                calculate_deployment_frequency(p_start, p_end, granularity, deployments)
            )
            # 2. Change Lead Time
            snapshots_to_upsert.append(
                calculate_change_lead_time(p_start, p_end, granularity, pr_deployments)
            )
            # 3. Recovery Time
            snapshots_to_upsert.append(
                calculate_recovery_time(p_start, p_end, granularity, incidents)
            )
            # 4. Change Failure Rate
            snapshots_to_upsert.append(
                calculate_change_failure_rate(p_start, p_end, granularity, deployments)
            )

    # 4. Upsert into metric_snapshots
    upserted_count = _upsert_snapshots(
        database_engine, workspace_id, repository_id, snapshots_to_upsert
    )
    logger.info(
        "metrics_recalculated",
        workspace_id=str(workspace_id),
        repository_id=str(repository_id),
        snapshot_count=upserted_count,
    )
    return upserted_count


def _upsert_snapshots(
    database_engine: Engine,
    workspace_id: UUID,
    repository_id: UUID,
    snapshots: list[MetricSnapshotResult],
) -> int:
    if not snapshots:
        return 0

    sql = text(
        """
        INSERT INTO metric_snapshots (
            workspace_id, repository_id, metric_type, granularity,
            period_start, period_end, value, unit, sample_size,
            calculation_version, dimensions, calculated_at, updated_at, version
        ) VALUES (
            :workspace_id, :repository_id, :metric_type, :granularity,
            :period_start, :period_end, :value, :unit, :sample_size,
            :calculation_version, CAST(:dimensions AS jsonb), now(), now(), 0
        )
        ON CONFLICT (
            repository_id, metric_type, granularity,
            period_start, period_end, calculation_version
        )
        DO UPDATE SET
            value = EXCLUDED.value,
            unit = EXCLUDED.unit,
            sample_size = EXCLUDED.sample_size,
            dimensions = EXCLUDED.dimensions,
            calculated_at = now(),
            updated_at = now(),
            version = metric_snapshots.version + 1
        """
    )

    count = 0
    with database_engine.begin() as connection:
        for snap in snapshots:
            params = {
                "workspace_id": str(workspace_id),
                "repository_id": str(repository_id),
                "metric_type": snap.metric_type,
                "granularity": snap.granularity,
                "period_start": snap.period_start,
                "period_end": snap.period_end,
                "value": snap.value,
                "unit": snap.unit,
                "sample_size": snap.sample_size,
                "calculation_version": snap.calculation_version,
                "dimensions": json.dumps(snap.dimensions),
            }
            connection.execute(sql, params)
            count += 1

    return count


def enqueue_recalculate_metrics_job(
    connection: Any,
    workspace_id: UUID,
    repository_id: UUID,
) -> UUID:
    """
    Deduplicated job enqueue for recalculating metrics on a repository.
    """
    sql = text(
        """
        INSERT INTO processing_jobs (
            workspace_id, repository_id, job_type, payload, status, priority, available_at
        ) VALUES (
            :workspace_id, :repository_id, 'RECALCULATE_METRICS',
            CAST(:payload AS jsonb), 'PENDING', 100, now()
        )
        RETURNING id
        """
    )
    payload = {
        "workspace_id": str(workspace_id),
        "repository_id": str(repository_id),
    }
    job_id = connection.execute(
        sql,
        {
            "workspace_id": str(workspace_id),
            "repository_id": str(repository_id),
            "payload": json.dumps(payload),
        },
    ).scalar_one()
    return UUID(str(job_id))
