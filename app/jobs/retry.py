from random import uniform
from uuid import UUID

from sqlalchemy import Engine, text


class JobOwnershipError(RuntimeError):
    pass


def retry_delay_seconds(attempts: int, jitter_seconds: float | None = None) -> float:
    if attempts < 0:
        raise ValueError("attempts cannot be negative")
    jitter = uniform(0.0, 1.0) if jitter_seconds is None else jitter_seconds
    return min(900.0, float(2**attempts) + max(0.0, jitter))


def sanitize_error(error: str) -> str:
    return " ".join(error.split())[:2000]


def mark_succeeded(database_engine: Engine, job_id: UUID, worker_id: str) -> None:
    with database_engine.begin() as connection:
        result = connection.execute(
            text(
                """
                UPDATE processing_jobs
                SET status = 'SUCCEEDED',
                    locked_at = NULL,
                    locked_by = NULL,
                    last_error = NULL,
                    finished_at = now(),
                    updated_at = now(),
                    version = version + 1
                WHERE id = :job_id
                  AND status = 'RUNNING'
                  AND locked_by = :worker_id
                """
            ),
            {"job_id": job_id, "worker_id": worker_id},
        )
        if result.rowcount != 1:
            raise JobOwnershipError("job is not owned by this worker")


def mark_failed(
    database_engine: Engine,
    job_id: UUID,
    worker_id: str,
    error: str,
    *,
    permanent: bool = False,
    jitter_seconds: float | None = None,
) -> str:
    safe_error = sanitize_error(error)

    with database_engine.begin() as connection:
        row = (
            connection.execute(
                text(
                    """
                SELECT attempts, max_attempts
                FROM processing_jobs
                WHERE id = :job_id
                  AND status = 'RUNNING'
                  AND locked_by = :worker_id
                FOR UPDATE
                """
                ),
                {"job_id": job_id, "worker_id": worker_id},
            )
            .mappings()
            .one_or_none()
        )

        if row is None:
            raise JobOwnershipError("job is not owned by this worker")

        dead = permanent or int(row["attempts"]) >= int(row["max_attempts"])
        if dead:
            connection.execute(
                text(
                    """
                    UPDATE processing_jobs
                    SET status = 'DEAD',
                        locked_at = NULL,
                        locked_by = NULL,
                        last_error = :error,
                        finished_at = now(),
                        updated_at = now(),
                        version = version + 1
                    WHERE id = :job_id
                    """
                ),
                {"job_id": job_id, "error": safe_error},
            )
            return "DEAD"

        delay = retry_delay_seconds(int(row["attempts"]), jitter_seconds)
        connection.execute(
            text(
                """
                UPDATE processing_jobs
                SET status = 'FAILED',
                    available_at = now() + make_interval(secs => :delay),
                    locked_at = NULL,
                    locked_by = NULL,
                    last_error = :error,
                    finished_at = NULL,
                    updated_at = now(),
                    version = version + 1
                WHERE id = :job_id
                """
            ),
            {"job_id": job_id, "delay": delay, "error": safe_error},
        )
        return "FAILED"
