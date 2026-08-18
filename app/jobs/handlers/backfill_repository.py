import structlog
from sqlalchemy import Engine
from app.db.models import ClaimedJob
from app.jobs.retry import mark_failed, mark_succeeded, requeue_with_payload

logger = structlog.get_logger()


def handle_backfill_repository(database_engine: Engine, job: ClaimedJob, worker_id: str) -> None:
    """
    Stub for paginated repository backfill.
    """
    cursor = job.payload.get("cursor")
    repository_id = job.payload.get("repositoryId")

    if not repository_id:
        mark_failed(database_engine, job.id, worker_id, "Missing repositoryId", permanent=True)
        return

    logger.info("backfill_repository_start", job_id=str(job.id), cursor=cursor)

    # In a real implementation:
    # 1. client.list_pull_requests(cursor=cursor)
    # 2. upsert batch
    # 3. next_cursor = result.next_cursor

    next_cursor = None
    if cursor is None:
        next_cursor = "page_2"
    elif cursor == "page_2":
        next_cursor = None  # Done

    if next_cursor:
        job.payload["cursor"] = next_cursor
        logger.info("backfill_repository_requeue", job_id=str(job.id), next_cursor=next_cursor)
        requeue_with_payload(database_engine, job.id, worker_id, job.payload, delay_seconds=1.0)
    else:
        logger.info("backfill_repository_done", job_id=str(job.id))
        mark_succeeded(database_engine, job.id, worker_id)
