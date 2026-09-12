from dataclasses import dataclass
from typing import cast

from sqlalchemy import Engine, text


@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    ready: int
    oldest_ready_wait_seconds: float
    running: int
    dead_letter: int


QUEUE_SNAPSHOT_SQL = text(
    """
    WITH ready_jobs AS (
        SELECT count(*) AS ready,
               COALESCE(
                   EXTRACT(EPOCH FROM now() - min(available_at)),
                   0
               )::double precision AS oldest_ready_wait_seconds
        FROM processing_jobs
        WHERE status IN ('PENDING', 'FAILED')
          AND available_at <= now()
          AND attempts < max_attempts
    ), job_states AS (
        SELECT count(*) FILTER (WHERE status = 'RUNNING') AS running,
               count(*) FILTER (WHERE status = 'DEAD') AS dead_letter
        FROM processing_jobs
        WHERE status IN ('RUNNING', 'DEAD')
    )
    SELECT ready_jobs.ready,
           ready_jobs.oldest_ready_wait_seconds,
           job_states.running,
           job_states.dead_letter
    FROM ready_jobs
    CROSS JOIN job_states
    """
)


def collect_queue_snapshot(
    database_engine: Engine,
    statement_timeout_seconds: int = 5,
) -> QueueSnapshot:
    if not 1 <= statement_timeout_seconds <= 60:
        raise ValueError("statement_timeout_seconds must be between 1 and 60")

    statement_timeout_ms = statement_timeout_seconds * 1000
    with database_engine.begin() as connection:
        connection.execute(text("SET TRANSACTION READ ONLY"))
        connection.execute(text(f"SET LOCAL statement_timeout = '{statement_timeout_ms}ms'"))
        row = connection.execute(QUEUE_SNAPSHOT_SQL).mappings().one()

    return QueueSnapshot(
        ready=cast(int, row["ready"]),
        oldest_ready_wait_seconds=cast(float, row["oldest_ready_wait_seconds"]),
        running=cast(int, row["running"]),
        dead_letter=cast(int, row["dead_letter"]),
    )
