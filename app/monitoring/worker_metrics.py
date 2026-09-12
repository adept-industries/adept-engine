from collections.abc import Collection
from functools import partial
from math import nan
from threading import Event, Lock, Thread
from time import monotonic, time
from wsgiref.simple_server import WSGIServer

import structlog
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server
from sqlalchemy import Engine

from app.jobs.queue_metrics import QueueSnapshot, collect_queue_snapshot

logger = structlog.get_logger()

ATTEMPT_OUTCOMES = frozenset({"succeeded", "retry_scheduled", "dead_lettered"})
PROCESSING_OUTCOMES = ATTEMPT_OUTCOMES | {"continuation_requeued", "operational_error"}
DEFERRAL_REASONS = frozenset({"continuation", "scope_busy", "shutdown"})
LEASE_FAILURE_PHASES = frozenset({"acquire", "connection_lost"})
JOB_ERROR_OPERATIONS = frozenset({"deferral", "dispatch"})

JOB_DURATION_BUCKETS_SECONDS = (
    0.1,
    0.5,
    1,
    2.5,
    5,
    10,
    30,
    60,
    120,
    300,
    600,
    900,
    1_800,
    3_600,
)


class WorkerMetrics:
    """Thread-safe, process-local worker metrics with bounded label values."""

    def __init__(self, configured_threads: int, known_job_types: Collection[str]) -> None:
        if configured_threads < 1:
            raise ValueError("configured_threads must be positive")
        if not known_job_types:
            raise ValueError("known_job_types must not be empty")

        self.registry = CollectorRegistry(auto_describe=True)
        self._known_job_types = frozenset(known_job_types)
        self._thread_slots = frozenset(str(slot) for slot in range(1, configured_threads + 1))
        self._active_started: dict[str, float | None] = {slot: None for slot in self._thread_slots}
        self._active_lock = Lock()

        self.configured_threads = Gauge(
            "adept_engine_worker_configured_threads",
            "Configured number of worker processing threads.",
            registry=self.registry,
        )
        self.process_start_time = Gauge(
            "adept_engine_worker_process_start_time_seconds",
            "Worker process start time as Unix epoch seconds.",
            registry=self.registry,
        )
        self.shutdown_requested = Gauge(
            "adept_engine_worker_shutdown_requested",
            "Whether graceful worker shutdown has been requested (1 or 0).",
            registry=self.registry,
        )
        self.thread_alive = Gauge(
            "adept_engine_worker_thread_alive",
            "Whether a configured processing thread is running (1 or 0).",
            ("thread_slot",),
            registry=self.registry,
        )
        self.thread_active = Gauge(
            "adept_engine_worker_thread_active",
            "Whether a processing thread is executing a leased job (1 or 0).",
            ("thread_slot",),
            registry=self.registry,
        )
        self.thread_last_successful_poll = Gauge(
            "adept_engine_worker_thread_last_successful_poll_timestamp_seconds",
            "Unix epoch time of the thread's last successful queue poll.",
            ("thread_slot",),
            registry=self.registry,
        )
        self.thread_active_job_duration = Gauge(
            "adept_engine_worker_thread_active_job_duration_seconds",
            "Current leased job processing duration; zero while idle.",
            ("thread_slot",),
            registry=self.registry,
        )
        self.job_attempts = Counter(
            "adept_engine_worker_job_attempts_total",
            "Durably finalized job attempts by bounded job type and outcome.",
            ("job_type", "outcome"),
            registry=self.registry,
        )
        self.job_processing_duration = Histogram(
            "adept_engine_worker_job_processing_duration_seconds",
            "Leased job processing duration by bounded job type and outcome.",
            ("job_type", "outcome"),
            buckets=JOB_DURATION_BUCKETS_SECONDS,
            registry=self.registry,
        )
        self.job_deferrals = Counter(
            "adept_engine_worker_job_deferrals_total",
            "Durable requeues that do not consume an attempt.",
            ("job_type", "reason"),
            registry=self.registry,
        )
        self.stale_jobs_recovered = Counter(
            "adept_engine_worker_stale_jobs_recovered_total",
            "Stale running jobs durably recovered by outcome.",
            ("outcome",),
            registry=self.registry,
        )
        self.poll_errors = Counter(
            "adept_engine_worker_poll_errors_total",
            "Queue polling or idle database-check errors by stable thread slot.",
            ("thread_slot",),
            registry=self.registry,
        )
        self.job_operation_errors = Counter(
            "adept_engine_worker_job_operation_errors_total",
            "Job operations without a confirmed durable outcome.",
            ("job_type", "operation"),
            registry=self.registry,
        )
        self.lease_failures = Counter(
            "adept_engine_worker_lease_failures_total",
            "Unexpected lease failures by stable thread slot and phase.",
            ("thread_slot", "phase"),
            registry=self.registry,
        )
        self.queue_ready_jobs = Gauge(
            "adept_engine_worker_queue_ready_jobs",
            "Jobs eligible to be claimed now.",
            registry=self.registry,
        )
        self.queue_oldest_ready_wait = Gauge(
            "adept_engine_worker_queue_oldest_ready_wait_seconds",
            "Seconds since the oldest ready job became eligible.",
            registry=self.registry,
        )
        self.queue_running_jobs = Gauge(
            "adept_engine_worker_queue_running_jobs",
            "Jobs currently in RUNNING state.",
            registry=self.registry,
        )
        self.queue_dead_letter_jobs = Gauge(
            "adept_engine_worker_queue_dead_letter_jobs",
            "Jobs currently in DEAD state.",
            registry=self.registry,
        )
        self.queue_collection_success = Gauge(
            "adept_engine_worker_queue_collection_success",
            "Whether the latest queue snapshot query succeeded (1 or 0).",
            registry=self.registry,
        )
        self.queue_last_success = Gauge(
            "adept_engine_worker_queue_last_success_timestamp_seconds",
            "Unix epoch time of the last successful queue snapshot query.",
            registry=self.registry,
        )
        self.queue_collection_errors = Counter(
            "adept_engine_worker_queue_collection_errors_total",
            "Failed queue snapshot queries.",
            registry=self.registry,
        )

        self.configured_threads.set(configured_threads)
        self.process_start_time.set(time())
        self.shutdown_requested.set(0)
        self.queue_ready_jobs.set(nan)
        self.queue_oldest_ready_wait.set(nan)
        self.queue_running_jobs.set(nan)
        self.queue_dead_letter_jobs.set(nan)
        self.queue_collection_success.set(0)
        self.queue_last_success.set(0)

        for slot in self._thread_slots:
            self.thread_alive.labels(thread_slot=slot).set(0)
            self.thread_active.labels(thread_slot=slot).set(0)
            self.thread_last_successful_poll.labels(thread_slot=slot).set(0)
            self.thread_active_job_duration.labels(thread_slot=slot).set_function(
                partial(self._active_duration, slot)
            )

    def _slot(self, slot: int | str) -> str:
        value = str(slot)
        if value not in self._thread_slots:
            raise ValueError(f"unknown thread slot: {value}")
        return value

    def _job_type(self, job_type: str) -> str:
        return job_type if job_type in self._known_job_types else "UNKNOWN"

    def _active_duration(self, slot: str) -> float:
        with self._active_lock:
            started = self._active_started[slot]
        return 0.0 if started is None else max(0.0, monotonic() - started)

    def mark_thread_started(self, slot: int | str) -> None:
        thread_slot = self._slot(slot)
        self.thread_alive.labels(thread_slot=thread_slot).set(1)

    def mark_thread_stopped(self, slot: int | str) -> None:
        thread_slot = self._slot(slot)
        self.mark_job_stopped(thread_slot)
        self.thread_alive.labels(thread_slot=thread_slot).set(0)

    def record_poll_success(self, slot: int | str) -> None:
        thread_slot = self._slot(slot)
        self.thread_last_successful_poll.labels(thread_slot=thread_slot).set(time())

    def record_poll_error(self, slot: int | str) -> None:
        thread_slot = self._slot(slot)
        self.poll_errors.labels(thread_slot=thread_slot).inc()

    def mark_job_started(self, slot: int | str) -> None:
        thread_slot = self._slot(slot)
        with self._active_lock:
            self._active_started[thread_slot] = monotonic()
        self.thread_active.labels(thread_slot=thread_slot).set(1)

    def mark_job_stopped(self, slot: int | str) -> None:
        thread_slot = self._slot(slot)
        with self._active_lock:
            self._active_started[thread_slot] = None
        self.thread_active.labels(thread_slot=thread_slot).set(0)

    def record_dispatch_outcome(self, job_type: str, outcome: str, duration: float) -> None:
        if outcome not in PROCESSING_OUTCOMES - {"operational_error"}:
            raise ValueError(f"unknown dispatch outcome: {outcome}")
        bounded_job_type = self._job_type(job_type)
        self.job_processing_duration.labels(job_type=bounded_job_type, outcome=outcome).observe(
            max(0.0, duration)
        )
        if outcome in ATTEMPT_OUTCOMES:
            self.job_attempts.labels(job_type=bounded_job_type, outcome=outcome).inc()
        if outcome == "continuation_requeued":
            self.record_deferral(job_type, "continuation")

    def record_dispatch_error(self, job_type: str, duration: float) -> None:
        bounded_job_type = self._job_type(job_type)
        self.job_operation_errors.labels(job_type=bounded_job_type, operation="dispatch").inc()
        self.job_processing_duration.labels(
            job_type=bounded_job_type, outcome="operational_error"
        ).observe(max(0.0, duration))

    def record_job_operation_error(self, job_type: str, operation: str) -> None:
        if operation not in JOB_ERROR_OPERATIONS:
            raise ValueError(f"unknown job error operation: {operation}")
        self.job_operation_errors.labels(
            job_type=self._job_type(job_type), operation=operation
        ).inc()

    def record_deferral(self, job_type: str, reason: str) -> None:
        if reason not in DEFERRAL_REASONS:
            raise ValueError(f"unknown deferral reason: {reason}")
        self.job_deferrals.labels(job_type=self._job_type(job_type), reason=reason).inc()

    def record_stale_recovery(self, retryable: int, dead: int) -> None:
        if retryable < 0 or dead < 0:
            raise ValueError("stale recovery counts cannot be negative")
        if retryable:
            self.stale_jobs_recovered.labels(outcome="retry_scheduled").inc(retryable)
        if dead:
            self.stale_jobs_recovered.labels(outcome="dead_lettered").inc(dead)

    def record_lease_failure(self, slot: int | str, phase: str) -> None:
        if phase not in LEASE_FAILURE_PHASES:
            raise ValueError(f"unknown lease failure phase: {phase}")
        self.lease_failures.labels(thread_slot=self._slot(slot), phase=phase).inc()

    def record_queue_snapshot(self, snapshot: QueueSnapshot) -> None:
        self.queue_ready_jobs.set(snapshot.ready)
        self.queue_oldest_ready_wait.set(snapshot.oldest_ready_wait_seconds)
        self.queue_running_jobs.set(snapshot.running)
        self.queue_dead_letter_jobs.set(snapshot.dead_letter)
        self.queue_collection_success.set(1)
        self.queue_last_success.set(time())

    def record_queue_collection_error(self) -> None:
        self.queue_collection_success.set(0)
        self.queue_collection_errors.inc()

    def mark_shutdown_requested(self) -> None:
        self.shutdown_requested.set(1)


class MetricsEndpoint:
    """Lifecycle wrapper for the private Prometheus WSGI endpoint."""

    def __init__(
        self,
        registry: CollectorRegistry,
        bind_address: str,
        port: int,
    ) -> None:
        self.registry = registry
        self.bind_address = bind_address
        self.port = port
        self._server: WSGIServer | None = None
        self._thread: Thread | None = None

    @property
    def bound_port(self) -> int:
        if self._server is None:
            raise RuntimeError("metrics endpoint is not running")
        return self._server.server_port

    def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("metrics endpoint is already running")
        self._server, self._thread = start_http_server(
            port=self.port,
            addr=self.bind_address,
            registry=self.registry,
        )

    def stop(self) -> None:
        server, thread = self._server, self._thread
        if server is None or thread is None:
            return
        try:
            server.shutdown()
        finally:
            try:
                server.server_close()
            finally:
                thread.join()
                self._server = None
                self._thread = None


class QueueMetricsSampler:
    """Refresh cached queue gauges independently from polling and scraping."""

    def __init__(
        self, database_engine: Engine, metrics: WorkerMetrics, interval_seconds: int
    ) -> None:
        self.database_engine = database_engine
        self.metrics = metrics
        self.interval_seconds = interval_seconds
        self._stop = Event()
        self._thread = Thread(target=self._run, name="queue-metrics", daemon=True)
        self._started = False

    def collect_once(self) -> bool:
        try:
            snapshot = collect_queue_snapshot(self.database_engine)
            self.metrics.record_queue_snapshot(snapshot)
        except Exception as exc:
            try:
                self.metrics.record_queue_collection_error()
            except Exception:
                logger.exception("engine_worker_queue_metrics_update_failed")
            logger.warning(
                "engine_worker_queue_metrics_collection_failed",
                error_type=type(exc).__name__,
            )
            return False
        return True

    def start(self) -> None:
        if self._started:
            raise RuntimeError("queue metrics sampler is already started")
        self._thread.start()
        self._started = True

    def stop(self, timeout_seconds: float = 10.0) -> bool:
        self._stop.set()
        if not self._started:
            return True
        self._thread.join(timeout_seconds)
        return not self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.collect_once()
            if self._stop.wait(self.interval_seconds):
                return
