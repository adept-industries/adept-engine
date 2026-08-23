"""
Job handler for RECALCULATE_METRICS.

Processes a claimed RECALCULATE_METRICS job and updates metric_snapshots.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import structlog
from sqlalchemy import Engine

from app.db.models import ClaimedJob
from app.jobs.retry import PermanentJobError
from app.metrics.service import recalculate_repository_metrics

logger = structlog.get_logger()


def handle_recalculate_metrics(
    database_engine: Engine,
    job: ClaimedJob,
    worker_id: str,
) -> None:
    """
    Recalculate DORA metrics for the target repository specified in the job payload.
    """
    payload = job.payload or {}
    repo_id_str = payload.get("repository_id") or payload.get("repositoryId")
    workspace_id_str = payload.get("workspace_id") or payload.get("workspaceId")

    if not repo_id_str or not workspace_id_str:
        raise PermanentJobError(
            f"RECALCULATE_METRICS job {job.id} missing repository_id or workspace_id"
        )

    try:
        repository_id = UUID(str(repo_id_str))
        workspace_id = UUID(str(workspace_id_str))
    except ValueError as exc:
        raise PermanentJobError(f"Invalid UUID in RECALCULATE_METRICS payload: {exc}") from exc

    from_date_str = payload.get("from_date")
    to_date_str = payload.get("to_date")

    from_date = datetime.fromisoformat(from_date_str) if from_date_str else None
    to_date = datetime.fromisoformat(to_date_str) if to_date_str else None

    logger.info(
        "recalculating_metrics_job_started",
        job_id=str(job.id),
        repository_id=str(repository_id),
        workspace_id=str(workspace_id),
        worker_id=worker_id,
    )

    count = recalculate_repository_metrics(
        database_engine,
        workspace_id=workspace_id,
        repository_id=repository_id,
        from_date=from_date,
        to_date=to_date,
    )

    logger.info(
        "recalculating_metrics_job_completed",
        job_id=str(job.id),
        repository_id=str(repository_id),
        snapshot_count=count,
    )
