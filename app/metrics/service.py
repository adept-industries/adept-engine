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
    get_recalculation_buckets,
)

logger = structlog.get_logger()


def link_deployments_to_pull_requests(
    database_engine: Engine,
    repository_id: UUID,
) -> int:
    """
    Rebuild automatic deployment links by merge/head SHA, normalized PR commit
    membership, then a bounded merge-window fallback for squash/rebase histories.

    Manual links are preserved.  The first recorded successful production
    deployment has no trustworthy lower boundary, so it receives exact links
    only instead of absorbing every older merged pull request.
    """
    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                DELETE FROM deployment_pull_requests dpr
                USING deployments d
                WHERE dpr.deployment_id = d.id
                  AND d.repository_id = :repository_id
                  AND dpr.link_method <> 'MANUAL'
                """
            ),
            {"repository_id": str(repository_id)},
        )
        exact_result = connection.execute(
            text(
                """
                INSERT INTO deployment_pull_requests (
                    deployment_id, pull_request_id, link_method, created_at
                )
                SELECT DISTINCT d.id, pr.id,
                       CASE
                           WHEN d.commit_sha = pr.merge_commit_sha
                             OR d.commit_sha = pr.head_sha
                           THEN 'MERGE_SHA'
                           ELSE 'COMMIT_GRAPH'
                       END,
                       now()
                FROM deployments d
                JOIN pull_requests pr ON pr.repository_id = d.repository_id
                LEFT JOIN pull_request_commits prc
                  ON prc.pull_request_id = pr.id
                 AND prc.commit_sha = d.commit_sha
                WHERE d.repository_id = :repository_id
                  AND (
                      d.commit_sha = pr.merge_commit_sha
                      OR d.commit_sha = pr.head_sha
                      OR prc.commit_sha IS NOT NULL
                  )
                  AND d.commit_sha IS NOT NULL
                  AND d.commit_sha <> ''
                ON CONFLICT (deployment_id, pull_request_id) DO NOTHING
                """
            ),
            {"repository_id": str(repository_id)},
        )
        fallback_result = connection.execute(
            text(
                """
                INSERT INTO deployment_pull_requests (
                    deployment_id, pull_request_id, link_method, created_at
                )
                SELECT d.id, pr.id, 'MERGE_WINDOW', now()
                FROM deployments d
                JOIN repositories r ON r.id = d.repository_id
                JOIN LATERAL (
                    SELECT previous.finished_at
                    FROM deployments previous
                    WHERE previous.repository_id = d.repository_id
                      AND previous.is_production = true
                      AND previous.status = 'SUCCESS'
                      AND previous.finished_at < d.finished_at
                    ORDER BY previous.finished_at DESC, previous.id DESC
                    LIMIT 1
                ) previous_boundary ON true
                JOIN pull_requests pr
                  ON pr.repository_id = d.repository_id
                 AND pr.state = 'MERGED'
                 AND pr.base_ref = COALESCE(
                     NULLIF(d.raw_data->'workflow_run'->>'head_branch', ''),
                     NULLIF(d.raw_data->'deployment'->>'ref', ''),
                     r.default_branch
                 )
                 AND pr.merged_at <= d.finished_at
                 AND pr.merged_at > previous_boundary.finished_at
                WHERE d.repository_id = :repository_id
                  AND d.is_production = true
                  AND d.status = 'SUCCESS'
                  AND d.finished_at IS NOT NULL
                ON CONFLICT (deployment_id, pull_request_id) DO NOTHING
                """
            ),
            {"repository_id": str(repository_id)},
        )
        return int(exact_result.rowcount or 0) + int(fallback_result.rowcount or 0)


def earliest_linked_production_deployment(
    database_engine: Engine,
    pull_request_id: UUID,
) -> datetime | None:
    """Return the earliest successful production deployment linked to one PR."""
    with database_engine.connect() as connection:
        return connection.execute(
            text(
                """
                SELECT min(d.finished_at)
                FROM deployment_pull_requests dpr
                JOIN deployments d ON d.id = dpr.deployment_id
                WHERE dpr.pull_request_id = :pull_request_id
                  AND d.is_production = true
                  AND d.status = 'SUCCESS'
                  AND d.finished_at IS NOT NULL
                """
            ),
            {"pull_request_id": pull_request_id},
        ).scalar_one()


def reconcile_github_incidents(database_engine: Engine, repository_id: UUID) -> None:
    """Derive GitHub failure incidents chronologically from normalized deployments."""
    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                DELETE FROM incidents i
                WHERE i.repository_id = :repository_id
                  AND i.source = 'GITHUB'
                  AND (
                      i.failed_deployment_id IS NULL
                      OR NOT EXISTS (
                          SELECT 1
                          FROM deployments failed
                          WHERE failed.id = i.failed_deployment_id
                            AND failed.repository_id = :repository_id
                            AND failed.is_production = true
                            AND failed.status = 'FAILURE'
                            AND failed.finished_at IS NOT NULL
                      )
                  )
                """
            ),
            {"repository_id": repository_id},
        )
        connection.execute(
            text(
                """
                INSERT INTO incidents (
                    workspace_id, repository_id, source, title, status,
                    failed_deployment_id, recovery_deployment_id,
                    detected_at, resolved_at, updated_at, version
                )
                SELECT failed.workspace_id,
                       failed.repository_id,
                       'GITHUB',
                       'Failed production deployment to ' || failed.environment,
                       CASE WHEN recovery.id IS NULL THEN 'OPEN' ELSE 'RESOLVED' END,
                       failed.id,
                       recovery.id,
                       failed.finished_at,
                       recovery.finished_at,
                       now(),
                       0
                FROM deployments failed
                LEFT JOIN LATERAL (
                    SELECT successful.id, successful.finished_at
                    FROM deployments successful
                    WHERE successful.repository_id = failed.repository_id
                      AND successful.is_production = true
                      AND successful.status = 'SUCCESS'
                      AND successful.finished_at > failed.finished_at
                    ORDER BY successful.finished_at ASC, successful.id ASC
                    LIMIT 1
                ) recovery ON true
                WHERE failed.repository_id = :repository_id
                  AND failed.is_production = true
                  AND failed.status = 'FAILURE'
                  AND failed.finished_at IS NOT NULL
                ON CONFLICT (failed_deployment_id)
                    WHERE failed_deployment_id IS NOT NULL
                DO UPDATE SET
                    workspace_id = EXCLUDED.workspace_id,
                    repository_id = EXCLUDED.repository_id,
                    source = 'GITHUB',
                    title = EXCLUDED.title,
                    status = EXCLUDED.status,
                    recovery_deployment_id = EXCLUDED.recovery_deployment_id,
                    detected_at = EXCLUDED.detected_at,
                    resolved_at = EXCLUDED.resolved_at,
                    updated_at = now(),
                    version = incidents.version + 1
                """
            ),
            {"repository_id": repository_id},
        )


def update_github_incident_lifecycle(database_engine: Engine, deployment_id: UUID) -> None:
    """Reconcile the deployment's repository without relying on delivery order."""
    with database_engine.connect() as connection:
        repository_id = connection.execute(
            text("SELECT repository_id FROM deployments WHERE id = :deployment_id"),
            {"deployment_id": deployment_id},
        ).scalar_one()
    reconcile_github_incidents(database_engine, UUID(str(repository_id)))


def recalculate_repository_metrics(
    database_engine: Engine,
    workspace_id: UUID,
    repository_id: UUID,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    affected_from: datetime | None = None,
    affected_to: datetime | None = None,
) -> int:
    """
    Recalculate and upsert DORA metric snapshots for a repository across DAY, WEEK, and MONTH.
    Defaults to the past 90 days up to tomorrow UTC.
    """
    # 1. Rebuild derived links and incident history before calculating.
    link_deployments_to_pull_requests(database_engine, repository_id)
    reconcile_github_incidents(database_engine, repository_id)

    now_utc = datetime.now(UTC)
    if to_date is None:
        to_date = now_utc + timedelta(days=1)
    if from_date is None:
        from_date = now_utc - timedelta(days=90)

    # 2. Fetch raw normalized data and the workspace calendar configuration.
    with database_engine.connect() as connection:
        timezone_name = str(
            connection.execute(
                text("SELECT timezone FROM workspaces WHERE id = :workspace_id"),
                {"workspace_id": str(workspace_id)},
            ).scalar_one()
        )
        deployments_rows = (
            connection.execute(
                text(
                    """
                    SELECT d.id, d.is_production, d.status, d.finished_at,
                           d.started_at, d.commit_sha,
                           EXISTS (
                               SELECT 1 FROM incidents i
                               WHERE i.failed_deployment_id = d.id
                           ) AS has_incident
                    FROM deployments d
                    WHERE d.repository_id = :repository_id
                      AND d.finished_at IS NOT NULL
                    ORDER BY d.finished_at ASC
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
                "has_incident": row["has_incident"],
            }
            for row in deployments_rows
        ]

        pr_dep_rows = (
            connection.execute(
                text(
                    """
                    SELECT DISTINCT ON (pr.id)
                           d.id AS deployment_id, d.is_production, d.status AS deployment_status,
                           d.finished_at AS deployment_finished_at,
                           pr.id AS pr_id, pr.first_commit_at
                    FROM deployments d
                    JOIN deployment_pull_requests dpr ON dpr.deployment_id = d.id
                    JOIN pull_requests pr ON pr.id = dpr.pull_request_id
                    WHERE d.repository_id = :repository_id
                      AND d.finished_at IS NOT NULL
                      AND d.is_production = true
                      AND d.status = 'SUCCESS'
                      AND pr.first_commit_at IS NOT NULL
                    ORDER BY pr.id, d.finished_at ASC
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
            }
            for row in pr_dep_rows
        ]

        incident_rows = (
            connection.execute(
                text(
                    """
                    SELECT i.id, i.detected_at, i.resolved_at,
                           recovery.finished_at AS recovery_finished_at
                    FROM incidents i
                    LEFT JOIN deployments recovery ON recovery.id = i.recovery_deployment_id
                    WHERE i.repository_id = :repository_id
                      AND i.status = 'RESOLVED'
                      AND i.resolved_at IS NOT NULL
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
                "detected_at": row["detected_at"],
                "resolved_at": row["resolved_at"],
                "recovery_finished_at": row["recovery_finished_at"],
            }
            for row in incident_rows
        ]

    # 3. Calculate snapshots for DAY, WEEK, MONTH
    snapshots_to_upsert: list[MetricSnapshotResult] = []

    for granularity in ("DAY", "WEEK", "MONTH"):
        if affected_from is not None:
            buckets = get_recalculation_buckets(
                affected_from,
                affected_to or affected_from,
                granularity,
                timezone_name,
            )
        else:
            buckets = get_period_buckets(from_date, to_date, granularity, timezone_name)
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
    affected_at: datetime | None = None,
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
        ON CONFLICT (repository_id, job_type)
        WHERE job_type = 'RECALCULATE_METRICS'
          AND status IN ('PENDING', 'FAILED')
        DO UPDATE SET
            payload = EXCLUDED.payload || jsonb_build_object(
                'affected_from', LEAST(
                    processing_jobs.payload->>'affected_from',
                    EXCLUDED.payload->>'affected_from'
                ),
                'affected_to', GREATEST(
                    processing_jobs.payload->>'affected_to',
                    EXCLUDED.payload->>'affected_to'
                )
            ),
            status = 'PENDING',
            attempts = 0,
            available_at = now(),
            last_error = NULL,
            updated_at = now(),
            version = processing_jobs.version + 1
        RETURNING id
        """
    )
    payload = {
        "workspace_id": str(workspace_id),
        "repository_id": str(repository_id),
    }
    if affected_at is not None:
        payload["affected_from"] = affected_at.isoformat()
        payload["affected_to"] = affected_at.isoformat()
    job_id = connection.execute(
        sql,
        {
            "workspace_id": str(workspace_id),
            "repository_id": str(repository_id),
            "payload": json.dumps(payload),
        },
    ).scalar_one()
    return UUID(str(job_id))
