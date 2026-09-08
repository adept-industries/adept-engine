import os
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from threading import Event
from types import FrameType
from uuid import uuid4

import structlog
from sqlalchemy import Engine

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db.models import ClaimedJob
from app.db.session import current_schema_version, get_database_engine
from app.jobs.claimer import claim_jobs
from app.jobs.dispatcher import dispatch_job
from app.jobs.lease import JobLease, JobScopeBusy
from app.jobs.retry import RequeueWithPayloadError, requeue_with_payload

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


def consumer_id(prefix: str, boot_id: str, slot: int) -> str:
    """Unique across threads and process restarts, within the DB's 128 chars."""
    suffix = f":{boot_id}:{slot}"
    return f"{prefix[: 128 - len(suffix)]}{suffix}"


def _connection_lost() -> None:
    # A lost advisory-lock session cannot be repaired safely while a handler is
    # still running. Docker's existing restart policy restarts this process;
    # unfinished jobs use the existing stale-claim recovery path.
    os._exit(1)


def _defer(database_engine: Engine, job: ClaimedJob, worker_id: str) -> None:
    with suppress(RequeueWithPayloadError):
        requeue_with_payload(database_engine, job.id, worker_id, job.payload, delay_seconds=5)


def consume_jobs(database_engine: Engine, settings: Settings, worker_id: str, stop: Event) -> None:
    logger.info(
        "engine_worker_starting",
        worker_id=worker_id,
        dispatch_enabled=True,
    )

    while not stop.is_set():
        try:
            jobs = claim_jobs(
                database_engine,
                worker_id,
                limit=1,
                stale_after_seconds=settings.engine_job_lock_timeout_seconds,
            )
            if jobs:
                job = jobs[0]
                if stop.is_set():
                    _defer(database_engine, job, worker_id)
                    return
                logger.info(
                    "engine_worker_claimed_jobs",
                    worker_id=worker_id,
                    count=len(jobs),
                )
                try:
                    with JobLease(
                        database_engine,
                        job,
                        worker_id,
                        interval_seconds=min(30, settings.engine_job_lock_timeout_seconds / 3),
                        on_connection_lost=_connection_lost,
                    ):
                        dispatch_claimed_jobs(database_engine, jobs, worker_id)
                except JobScopeBusy:
                    # Do not spend an attempt or block a consumer waiting for
                    # another thread's repository/catalog operation to finish.
                    _defer(database_engine, job, worker_id)
                    # Look for other eligible work before the deferred job is
                    # available again, rather than repeatedly claiming it.
                    continue
            else:
                version = current_schema_version(database_engine)
                logger.debug(
                    "engine_worker_idle",
                    worker_id=worker_id,
                    schema_version=version,
                )
        except Exception as exc:
            logger.warning(
                "engine_worker_poll_failed",
                worker_id=worker_id,
                error=str(exc),
            )

        stop.wait(settings.engine_poll_interval_ms / 1000)


def run() -> None:
    configure_logging()
    settings = get_settings()
    database_engine = get_database_engine()
    current_schema_version(database_engine)
    stop = Event()
    boot_id = uuid4().hex

    def request_stop(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        stop.set()

    previous_handlers = {
        signum: signal.signal(signum, request_stop) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    logger.info("engine_worker_pool_starting", consumer_count=settings.engine_worker_threads)
    try:
        with ThreadPoolExecutor(
            max_workers=settings.engine_worker_threads, thread_name_prefix="engine-consumer"
        ) as executor:
            try:
                futures = [
                    executor.submit(
                        consume_jobs,
                        database_engine,
                        settings,
                        consumer_id(settings.engine_worker_id, boot_id, slot),
                        stop,
                    )
                    for slot in range(1, settings.engine_worker_threads + 1)
                ]
                for future in as_completed(futures):
                    future.result()
                    if not stop.is_set():
                        raise RuntimeError("consumer exited unexpectedly")
            finally:
                stop.set()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        database_engine.dispose()


if __name__ == "__main__":
    run()
