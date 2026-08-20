from dataclasses import dataclass

import structlog
from sqlalchemy import Connection, Engine, text

from app.db.models import ClaimedJob

logger = structlog.get_logger()

STALE_LOCK_ERROR = "STALE_LOCK_RECOVERED: previous worker lock expired before completion"
STALE_LOCK_EXHAUSTED_ERROR = (
    "STALE_LOCK_EXHAUSTED: previous worker lock expired after the final attempt"
)


@dataclass(frozen=True, slots=True)
class StaleJobRecovery:
    retryable: int
    dead: int

    @property
    def total(self) -> int:
        return self.retryable + self.dead


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

MARK_EXHAUSTED_STALE_JOBS_DEAD_SQL = text(
    """
    UPDATE processing_jobs
    SET status = 'DEAD',
        locked_at = NULL,
        locked_by = NULL,
        last_error = :last_error,
        finished_at = now(),
        updated_at = now(),
        version = version + 1
    WHERE status = 'RUNNING'
      AND (locked_at IS NULL
           OR locked_at <= now() - make_interval(secs => :stale_after_seconds))
      AND attempts >= max_attempts
    """
)

REQUEUE_RETRYABLE_STALE_JOBS_SQL = text(
    """
    UPDATE processing_jobs
    SET status = 'FAILED',
        available_at = now(),
        locked_at = NULL,
        locked_by = NULL,
        last_error = :last_error,
        finished_at = NULL,
        updated_at = now(),
        version = version + 1
    WHERE status = 'RUNNING'
      AND (locked_at IS NULL
           OR locked_at <= now() - make_interval(secs => :stale_after_seconds))
      AND attempts < max_attempts
    """
)


def _recover_stale_jobs(connection: Connection, stale_after_seconds: int) -> StaleJobRecovery:
    parameters = {"stale_after_seconds": stale_after_seconds}
    dead = connection.execute(
        MARK_EXHAUSTED_STALE_JOBS_DEAD_SQL,
        {**parameters, "last_error": STALE_LOCK_EXHAUSTED_ERROR},
    ).rowcount
    retryable = connection.execute(
        REQUEUE_RETRYABLE_STALE_JOBS_SQL,
        {**parameters, "last_error": STALE_LOCK_ERROR},
    ).rowcount
    return StaleJobRecovery(retryable=retryable, dead=dead)


def claim_jobs(
    database_engine: Engine,
    worker_id: str,
    limit: int = 10,
    *,
    stale_after_seconds: int = 900,
) -> list[ClaimedJob]:
    if not worker_id or len(worker_id) > 128:
        raise ValueError("worker_id must contain 1 to 128 characters")
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    if stale_after_seconds < 30 or stale_after_seconds > 86_400:
        raise ValueError("stale_after_seconds must be between 30 and 86400")

    with database_engine.begin() as connection:
        recovery = _recover_stale_jobs(connection, stale_after_seconds)
        rows = (
            connection.execute(
                CLAIM_SQL,
                {"worker_id": worker_id, "limit": limit},
            )
            .mappings()
            .all()
        )

    if recovery.total:
        logger.warning(
            "stale_jobs_recovered",
            retryable=recovery.retryable,
            dead=recovery.dead,
            stale_after_seconds=stale_after_seconds,
        )

    return [ClaimedJob.from_row(row) for row in rows]
