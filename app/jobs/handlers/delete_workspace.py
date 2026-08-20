from uuid import UUID

import structlog
from sqlalchemy import Engine, text

from app.db.models import ClaimedJob
from app.jobs.retry import PermanentJobError

logger = structlog.get_logger()


def _workspace_id_from_payload(job: ClaimedJob) -> UUID:
    raw_workspace_id = job.payload.get("workspaceId")
    if not isinstance(raw_workspace_id, str) or not raw_workspace_id.strip():
        raise PermanentJobError("Missing workspaceId")

    try:
        return UUID(raw_workspace_id)
    except ValueError as exc:
        raise PermanentJobError("Invalid workspaceId") from exc


def handle_delete_workspace(
    database_engine: Engine,
    job: ClaimedJob,
    worker_id: str,
) -> None:
    """Hard-delete a workspace that the API has already marked DELETING."""
    workspace_id = _workspace_id_from_payload(job)

    logger.info(
        "delete_workspace_start",
        job_id=str(job.id),
        workspace_id=str(workspace_id),
    )

    with database_engine.begin() as connection:
        job_scope = (
            connection.execute(
                text(
                    """
                    SELECT workspace_id, repository_id, raw_event_id
                    FROM processing_jobs
                    WHERE id = :job_id
                      AND status = 'RUNNING'
                      AND locked_by = :worker_id
                    FOR UPDATE
                    """
                ),
                {"job_id": job.id, "worker_id": worker_id},
            )
            .mappings()
            .one_or_none()
        )
        if job_scope is None:
            raise RuntimeError("DELETE_WORKSPACE job is not owned by this worker")

        if any(
            job_scope[column] is not None
            for column in ("workspace_id", "repository_id", "raw_event_id")
        ):
            raise PermanentJobError("DELETE_WORKSPACE job must not reference tenant data")

        workspace_status = connection.execute(
            text(
                """
                SELECT status
                FROM workspaces
                WHERE id = :workspace_id
                FOR UPDATE
                """
            ),
            {"workspace_id": workspace_id},
        ).scalar_one_or_none()

        if workspace_status is None:
            logger.info(
                "delete_workspace_already_absent",
                job_id=str(job.id),
                workspace_id=str(workspace_id),
            )
            return

        if workspace_status != "DELETING":
            raise PermanentJobError(
                f"Workspace must be DELETING before removal; found {workspace_status}"
            )

        result = connection.execute(
            text(
                """
                DELETE FROM workspaces
                WHERE id = :workspace_id
                  AND status = 'DELETING'
                """
            ),
            {"workspace_id": workspace_id},
        )
        if result.rowcount != 1:
            raise RuntimeError("Workspace deletion did not remove exactly one row")

    logger.info(
        "delete_workspace_done",
        job_id=str(job.id),
        workspace_id=str(workspace_id),
    )
