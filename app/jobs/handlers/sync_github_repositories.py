import structlog
from sqlalchemy import Engine

from app.db.models import ClaimedJob
from app.jobs.retry import mark_failed, mark_succeeded, requeue_with_payload

logger = structlog.get_logger()


def handle_sync_github_repositories(
    database_engine: Engine, job: ClaimedJob, worker_id: str
) -> None:
    """
    Stub for paginated github repositories sync.
    Demonstrates cursor pattern for paginated jobs.
    """
    cursor = job.payload.get("cursor")
    workspace_id = job.payload.get("workspaceId")

    if not workspace_id:
        mark_failed(database_engine, job.id, worker_id, "Missing workspaceId", permanent=True)
        return

    logger.info("sync_github_repositories_start", job_id=str(job.id), cursor=cursor)

    # In a real implementation:
    # 1. client.list_repositories(cursor=cursor)
    # 2. upsert batch
    # 3. next_cursor = result.next_cursor

    next_cursor = None
    if cursor is None:
        next_cursor = "page_2"
    elif cursor == "page_2":
        next_cursor = None  # Done

    if next_cursor:
        job.payload["cursor"] = next_cursor
        logger.info("sync_github_repositories_requeue", job_id=str(job.id), next_cursor=next_cursor)
        requeue_with_payload(database_engine, job.id, worker_id, job.payload, delay_seconds=2.0)
    else:
        logger.info("sync_github_repositories_done", job_id=str(job.id))
        mark_succeeded(database_engine, job.id, worker_id)
