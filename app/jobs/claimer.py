from sqlalchemy import Engine, text

from app.db.models import ClaimedJob

CLAIM_SQL = text(
    """
    WITH claim AS (
        SELECT id
        FROM processing_jobs
        WHERE status IN ('PENDING', 'FAILED')
          AND available_at <= now()
          AND attempts < max_attempts
        ORDER BY priority ASC, created_at ASC, id ASC
        FOR UPDATE SKIP LOCKED
        LIMIT :limit
    )
    UPDATE processing_jobs AS job
    SET status = 'RUNNING',
        locked_at = now(),
        locked_by = :worker_id,
        attempts = job.attempts + 1,
        updated_at = now(),
        version = job.version + 1
    FROM claim
    WHERE job.id = claim.id
    RETURNING job.*
    """
)


def claim_jobs(database_engine: Engine, worker_id: str, limit: int = 10) -> list[ClaimedJob]:
    if not worker_id or len(worker_id) > 128:
        raise ValueError("worker_id must contain 1 to 128 characters")
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")

    with database_engine.begin() as connection:
        rows = (
            connection.execute(
                CLAIM_SQL,
                {"worker_id": worker_id, "limit": limit},
            )
            .mappings()
            .all()
        )

    return [ClaimedJob.from_row(row) for row in rows]
