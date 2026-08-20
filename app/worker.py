import time

import structlog
from sqlalchemy import Engine

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.models import ClaimedJob
from app.db.session import current_schema_version, get_database_engine
from app.jobs.claimer import claim_jobs
from app.jobs.dispatcher import dispatch_job

logger = structlog.get_logger()


def dispatch_claimed_jobs(database_engine: Engine, jobs: list[ClaimedJob], worker_id: str) -> None:
    """Dispatch every claimed job even if one job loses ownership or crashes."""
    for job in jobs:
        try:
            dispatch_job(database_engine, job, worker_id)
        except Exception as exc:
            logger.exception(
                "engine_worker_job_dispatch_failed",
                worker_id=worker_id,
                job_id=str(job.id),
                job_type=job.job_type,
                error=str(exc),
            )


def run() -> None:
    configure_logging()
    settings = get_settings()
    database_engine = get_database_engine()

    logger.info(
        "engine_worker_starting",
        worker_id=settings.engine_worker_id,
        dispatch_enabled=True,
    )

    while True:
        try:
            jobs = claim_jobs(
                database_engine,
                settings.engine_worker_id,
                limit=5,
                stale_after_seconds=settings.engine_job_lock_timeout_seconds,
            )
            if jobs:
                logger.info(
                    "engine_worker_claimed_jobs",
                    worker_id=settings.engine_worker_id,
                    count=len(jobs),
                )
                dispatch_claimed_jobs(database_engine, jobs, settings.engine_worker_id)
            else:
                version = current_schema_version(database_engine)
                logger.debug(
                    "engine_worker_idle",
                    worker_id=settings.engine_worker_id,
                    schema_version=version,
                )
        except Exception as exc:
            logger.warning(
                "engine_worker_poll_failed",
                worker_id=settings.engine_worker_id,
                error=str(exc),
            )

        time.sleep(settings.engine_poll_interval_ms / 1000)


if __name__ == "__main__":
    run()
