import time

import structlog

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import current_schema_version, get_database_engine
from app.jobs.claimer import claim_jobs
from app.jobs.dispatcher import dispatch_job


def run() -> None:
    configure_logging()
    logger = structlog.get_logger()
    settings = get_settings()
    database_engine = get_database_engine()

    logger.info(
        "engine_worker_starting",
        worker_id=settings.engine_worker_id,
        dispatch_enabled=True,
    )

    while True:
        try:
            jobs = claim_jobs(database_engine, settings.engine_worker_id, limit=5)
            if jobs:
                logger.info(
                    "engine_worker_claimed_jobs",
                    worker_id=settings.engine_worker_id,
                    count=len(jobs),
                )
                for job in jobs:
                    dispatch_job(database_engine, job, settings.engine_worker_id)
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
