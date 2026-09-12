import os
import signal
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from functools import partial
from threading import Event
from time import monotonic
from types import FrameType
from uuid import uuid4

import structlog
from sqlalchemy import Engine

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db.models import ClaimedJob
from app.db.session import current_schema_version, get_database_engine
from app.jobs.claimer import StaleJobRecovery, claim_jobs
from app.jobs.dispatcher import HANDLERS, dispatch_job
from app.jobs.lease import JobLease, JobScopeBusy
from app.jobs.retry import RequeueWithPayloadError, requeue_with_payload
from app.monitoring.worker_metrics import MetricsEndpoint, QueueMetricsSampler, WorkerMetrics

logger = structlog.get_logger()


def _record_metric(operation: str, callback: Callable[[], None]) -> None:
    """Keep observability failures from changing durable worker behavior."""
    try:
        callback()
    except Exception:
        logger.exception("engine_worker_metrics_update_failed", operation=operation)


def dispatch_claimed_jobs(
    database_engine: Engine,
    jobs: list[ClaimedJob],
    worker_id: str,
    metrics: WorkerMetrics | None = None,
) -> None:
    """Dispatch every claimed job even if one job loses ownership or crashes."""
    for job in jobs:
        started = monotonic()
        try:
            outcome = dispatch_job(database_engine, job, worker_id)
        except Exception as exc:
            if metrics is not None:
                _record_metric(
                    "dispatch_error",
                    partial(
                        metrics.record_dispatch_error,
                        job.job_type,
                        monotonic() - started,
                    ),
                )
            logger.exception(
                "engine_worker_job_dispatch_failed",
                worker_id=worker_id,
                job_id=str(job.id),
                job_type=job.job_type,
                error=str(exc),
            )
        else:
            if metrics is not None:
                _record_metric(
                    "dispatch_outcome",
                    partial(
                        metrics.record_dispatch_outcome,
                        job.job_type,
                        outcome.value,
                        monotonic() - started,
                    ),
                )


def consumer_id(prefix: str, boot_id: str, slot: int) -> str:
    """Unique across threads and process restarts, within the DB's 128 chars."""
    suffix = f":{boot_id}:{slot}"
    return f"{prefix[: 128 - len(suffix)]}{suffix}"


def _connection_lost(metrics: WorkerMetrics, thread_slot: int) -> None:
    # A lost advisory-lock session cannot be repaired safely while a handler is
    # still running. Docker's existing restart policy restarts this process;
    # unfinished jobs use the existing stale-claim recovery path.
    try:
        _record_metric(
            "lease_connection_lost",
            partial(metrics.record_lease_failure, thread_slot, "connection_lost"),
        )
    finally:
        os._exit(1)


def _defer(database_engine: Engine, job: ClaimedJob, worker_id: str) -> None:
    with suppress(RequeueWithPayloadError):
        requeue_with_payload(database_engine, job.id, worker_id, job.payload, delay_seconds=5)


def _defer_with_metrics(
    database_engine: Engine,
    job: ClaimedJob,
    worker_id: str,
    metrics: WorkerMetrics,
    reason: str,
) -> None:
    try:
        _defer(database_engine, job, worker_id)
    except Exception as exc:
        _record_metric(
            "deferral_error",
            partial(metrics.record_job_operation_error, job.job_type, "deferral"),
        )
        logger.warning(
            "engine_worker_job_deferral_failed",
            worker_id=worker_id,
            job_id=str(job.id),
            job_type=job.job_type,
            reason=reason,
            error=str(exc),
        )
    else:
        _record_metric(
            "deferral",
            partial(metrics.record_deferral, job.job_type, reason),
        )


def consume_jobs(
    database_engine: Engine,
    settings: Settings,
    worker_id: str,
    stop: Event,
    thread_slot: int,
    metrics: WorkerMetrics,
) -> None:
    logger.info(
        "engine_worker_starting",
        worker_id=worker_id,
        thread_slot=thread_slot,
        dispatch_enabled=True,
    )

    def record_stale_recovery(recovery: StaleJobRecovery) -> None:
        _record_metric(
            "stale_recovery",
            partial(metrics.record_stale_recovery, recovery.retryable, recovery.dead),
        )

    _record_metric("thread_started", partial(metrics.mark_thread_started, thread_slot))
    try:
        while not stop.is_set():
            try:
                jobs = claim_jobs(
                    database_engine,
                    worker_id,
                    limit=1,
                    stale_after_seconds=settings.engine_job_lock_timeout_seconds,
                    on_stale_recovery=record_stale_recovery,
                )
            except Exception as exc:
                _record_metric("poll_error", partial(metrics.record_poll_error, thread_slot))
                logger.warning(
                    "engine_worker_poll_failed",
                    worker_id=worker_id,
                    error=str(exc),
                )
                stop.wait(settings.engine_poll_interval_ms / 1000)
                continue

            _record_metric("poll_success", partial(metrics.record_poll_success, thread_slot))
            if jobs:
                job = jobs[0]
                if stop.is_set():
                    _defer_with_metrics(
                        database_engine,
                        job,
                        worker_id,
                        metrics,
                        "shutdown",
                    )
                    return
                logger.info(
                    "engine_worker_claimed_jobs",
                    worker_id=worker_id,
                    thread_slot=thread_slot,
                    count=len(jobs),
                )
                try:
                    with JobLease(
                        database_engine,
                        job,
                        worker_id,
                        interval_seconds=min(30, settings.engine_job_lock_timeout_seconds / 3),
                        on_connection_lost=lambda: _connection_lost(metrics, thread_slot),
                    ):
                        _record_metric(
                            "job_started",
                            partial(metrics.mark_job_started, thread_slot),
                        )
                        try:
                            dispatch_claimed_jobs(database_engine, jobs, worker_id, metrics)
                        finally:
                            _record_metric(
                                "job_stopped",
                                partial(metrics.mark_job_stopped, thread_slot),
                            )
                except JobScopeBusy:
                    # Do not spend an attempt or block a consumer waiting for
                    # another thread's repository/catalog operation to finish.
                    _defer_with_metrics(
                        database_engine,
                        job,
                        worker_id,
                        metrics,
                        "scope_busy",
                    )
                    # Look for other eligible work before the deferred job is
                    # available again, rather than repeatedly claiming it.
                    continue
                except Exception as exc:
                    _record_metric(
                        "lease_acquire_failure",
                        partial(metrics.record_lease_failure, thread_slot, "acquire"),
                    )
                    logger.warning(
                        "engine_worker_lease_failed",
                        worker_id=worker_id,
                        thread_slot=thread_slot,
                        job_id=str(job.id),
                        job_type=job.job_type,
                        error=str(exc),
                    )
            else:
                try:
                    version = current_schema_version(database_engine)
                except Exception as exc:
                    _record_metric("poll_error", partial(metrics.record_poll_error, thread_slot))
                    logger.warning(
                        "engine_worker_poll_failed",
                        worker_id=worker_id,
                        error=str(exc),
                    )
                else:
                    logger.debug(
                        "engine_worker_idle",
                        worker_id=worker_id,
                        thread_slot=thread_slot,
                        schema_version=version,
                    )

            stop.wait(settings.engine_poll_interval_ms / 1000)
    finally:
        _record_metric("thread_stopped", partial(metrics.mark_thread_stopped, thread_slot))


def run() -> None:
    configure_logging()
    settings = get_settings()
    database_engine = get_database_engine()
    current_schema_version(database_engine)
    stop = Event()
    boot_id = uuid4().hex
    metrics = WorkerMetrics(settings.engine_worker_threads, HANDLERS)
    metrics_endpoint = MetricsEndpoint(
        metrics.registry,
        settings.engine_metrics_bind_address,
        settings.engine_metrics_port,
    )
    queue_metrics = QueueMetricsSampler(
        database_engine,
        metrics,
        settings.engine_queue_metrics_interval_seconds,
    )

    def request_stop(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        _record_metric("shutdown_requested", metrics.mark_shutdown_requested)
        stop.set()

    previous_handlers = {
        signum: signal.signal(signum, request_stop) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    logger.info("engine_worker_pool_starting", consumer_count=settings.engine_worker_threads)
    try:
        metrics_endpoint.start()
        queue_metrics.start()
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
                        slot,
                        metrics,
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
        if not queue_metrics.stop():
            logger.warning("engine_worker_queue_metrics_shutdown_timed_out")
        metrics_endpoint.stop()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        database_engine.dispose()


if __name__ == "__main__":
    run()
