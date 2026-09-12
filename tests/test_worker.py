from datetime import UTC, datetime
from threading import Barrier, Event
from unittest.mock import ANY, MagicMock
from uuid import uuid4

import pytest
from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families
from pydantic import ValidationError

from app import worker
from app.core.config import Settings
from app.db.models import ClaimedJob
from app.jobs.claimer import StaleJobRecovery
from app.jobs.dispatcher import HANDLERS, JobDispatchOutcome
from app.monitoring.worker_metrics import WorkerMetrics
from app.worker import dispatch_claimed_jobs

pytestmark = pytest.mark.usefixtures("isolated_worker_environment")


def _job() -> ClaimedJob:
    return ClaimedJob(
        id=uuid4(),
        job_type="PROCESS_GITHUB_EVENT",
        payload={},
        priority=50,
        attempts=1,
        max_attempts=5,
        locked_by="test-worker",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        version=1,
    )


def _metrics(thread_count: int = 2) -> WorkerMetrics:
    return WorkerMetrics(thread_count, HANDLERS)


def _metric_samples(
    metrics: WorkerMetrics,
    name: str,
) -> list[tuple[dict[str, str], float]]:
    exposition = generate_latest(metrics.registry).decode("utf-8")
    return [
        (sample.labels, float(sample.value))
        for family in text_string_to_metric_families(exposition)
        for sample in family.samples
        if sample.name == name
    ]


def _metric_value(
    metrics: WorkerMetrics,
    name: str,
    labels: dict[str, str] | None = None,
) -> float:
    expected_labels = labels or {}
    matches = [
        value
        for sample_labels, value in _metric_samples(metrics, name)
        if sample_labels == expected_labels
    ]
    assert len(matches) == 1, f"expected one {name}{expected_labels} sample, got {matches}"
    return matches[0]


def test_dispatch_continues_after_one_claimed_job_crashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _job()
    second = _job()
    dispatched_ids = []

    def fake_dispatch(_engine: MagicMock, job: ClaimedJob, _worker_id: str) -> None:
        dispatched_ids.append(job.id)
        if job.id == first.id:
            raise RuntimeError("lost ownership")

    monkeypatch.setattr("app.worker.dispatch_job", fake_dispatch)

    dispatch_claimed_jobs(MagicMock(), [first, second], "test-worker")

    assert dispatched_ids == [first.id, second.id]


@pytest.mark.parametrize(
    "outcome",
    [
        JobDispatchOutcome.SUCCEEDED,
        JobDispatchOutcome.RETRY_SCHEDULED,
        JobDispatchOutcome.DEAD_LETTERED,
        JobDispatchOutcome.CONTINUATION_REQUEUED,
    ],
)
def test_dispatch_records_confirmed_outcome_metrics(
    monkeypatch: pytest.MonkeyPatch,
    outcome: JobDispatchOutcome,
) -> None:
    job = _job()
    metrics = _metrics()
    clock = iter((10.0, 12.5))
    monkeypatch.setattr(worker, "monotonic", lambda: next(clock))
    monkeypatch.setattr(worker, "dispatch_job", MagicMock(return_value=outcome))

    dispatch_claimed_jobs(MagicMock(), [job], "test-worker", metrics)

    labels = {"job_type": job.job_type, "outcome": outcome.value}
    assert (
        _metric_value(
            metrics,
            "adept_engine_worker_job_processing_duration_seconds_count",
            labels,
        )
        == 1
    )
    assert (
        _metric_value(
            metrics,
            "adept_engine_worker_job_processing_duration_seconds_sum",
            labels,
        )
        == 2.5
    )
    if outcome is JobDispatchOutcome.CONTINUATION_REQUEUED:
        assert _metric_samples(metrics, "adept_engine_worker_job_attempts_total") == []
        assert (
            _metric_value(
                metrics,
                "adept_engine_worker_job_deferrals_total",
                {"job_type": job.job_type, "reason": "continuation"},
            )
            == 1
        )
    else:
        assert (
            _metric_value(
                metrics,
                "adept_engine_worker_job_attempts_total",
                labels,
            )
            == 1
        )


def test_dispatch_records_operational_error_without_attempt_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job()
    metrics = _metrics()
    clock = iter((20.0, 23.0))
    monkeypatch.setattr(worker, "monotonic", lambda: next(clock))
    monkeypatch.setattr(worker, "dispatch_job", MagicMock(side_effect=RuntimeError("lost owner")))

    dispatch_claimed_jobs(MagicMock(), [job], "test-worker", metrics)

    assert _metric_samples(metrics, "adept_engine_worker_job_attempts_total") == []
    assert (
        _metric_value(
            metrics,
            "adept_engine_worker_job_operation_errors_total",
            {"job_type": job.job_type, "operation": "dispatch"},
        )
        == 1
    )


def test_metrics_failure_does_not_change_successful_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job()
    metrics = MagicMock()
    metrics.record_dispatch_outcome.side_effect = RuntimeError("metrics unavailable")
    dispatch = MagicMock(return_value=JobDispatchOutcome.SUCCEEDED)
    monkeypatch.setattr(worker, "dispatch_job", dispatch)

    dispatch_claimed_jobs(MagicMock(), [job], "test-worker", metrics)

    dispatch.assert_called_once()
    metrics.record_dispatch_outcome.assert_called_once()


def test_thread_count_is_bounded_and_defaults_to_two(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENGINE_WORKER_THREADS", raising=False)
    assert Settings().engine_worker_threads == 2
    for value in (1, 2):
        monkeypatch.setenv("ENGINE_WORKER_THREADS", str(value))
        assert Settings().engine_worker_threads == value
    for value in (0, 3):
        monkeypatch.setenv("ENGINE_WORKER_THREADS", str(value))
        with pytest.raises(ValidationError):
            Settings()


def test_worker_metrics_settings_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings()
    assert settings.engine_metrics_bind_address == "0.0.0.0"
    assert settings.engine_metrics_port == 8001
    assert settings.engine_queue_metrics_interval_seconds == 60

    monkeypatch.setenv("ENGINE_METRICS_BIND_ADDRESS", "127.0.0.1")
    monkeypatch.setenv("ENGINE_METRICS_PORT", "9101")
    monkeypatch.setenv("ENGINE_QUEUE_METRICS_INTERVAL_SECONDS", "60")
    settings = Settings()
    assert settings.engine_metrics_bind_address == "127.0.0.1"
    assert settings.engine_metrics_port == 9101
    assert settings.engine_queue_metrics_interval_seconds == 60

    for name, value in (
        ("ENGINE_METRICS_BIND_ADDRESS", ""),
        ("ENGINE_METRICS_PORT", "0"),
        ("ENGINE_METRICS_PORT", "65536"),
        ("ENGINE_QUEUE_METRICS_INTERVAL_SECONDS", "4"),
        ("ENGINE_QUEUE_METRICS_INTERVAL_SECONDS", "3601"),
    ):
        monkeypatch.setenv(name, value)
        with pytest.raises(ValidationError):
            Settings()
        monkeypatch.delenv(name)


def test_consumer_owners_are_unique_across_slots_and_restarts() -> None:
    owners = {
        worker.consumer_id("w" * 128, boot, slot)
        for boot in (uuid4().hex, uuid4().hex)
        for slot in (1, 2)
    }
    assert len(owners) == 4
    assert all(len(owner) == 128 for owner in owners)


@pytest.mark.parametrize("busy", [False, True])
def test_consumer_claims_one_job_and_guards_dispatch(
    monkeypatch: pytest.MonkeyPatch, busy: bool
) -> None:
    engine = MagicMock()
    stop = Event()
    job = _job()
    claim = MagicMock(return_value=[job])
    lease = MagicMock()
    if busy:
        lease.return_value.__enter__.side_effect = worker.JobScopeBusy("repo")
    dispatch = MagicMock(side_effect=lambda *_: stop.set())
    defer = MagicMock(side_effect=lambda *_: stop.set())
    monkeypatch.setattr(worker, "claim_jobs", claim)
    monkeypatch.setattr(worker, "JobLease", lease)
    monkeypatch.setattr(worker, "dispatch_claimed_jobs", dispatch)
    monkeypatch.setattr(worker, "_defer", defer)
    metrics = _metrics()

    worker.consume_jobs(engine, Settings(), "owner", stop, 1, metrics)

    claim.assert_called_once_with(
        engine,
        "owner",
        limit=1,
        stale_after_seconds=900,
        on_stale_recovery=ANY,
    )
    lease.return_value.__enter__.assert_called_once()
    if busy:
        dispatch.assert_not_called()
        defer.assert_called_once_with(engine, job, "owner")
        assert (
            _metric_value(
                metrics,
                "adept_engine_worker_job_deferrals_total",
                {"job_type": job.job_type, "reason": "scope_busy"},
            )
            == 1
        )
        assert _metric_samples(metrics, "adept_engine_worker_job_attempts_total") == []
    else:
        dispatch.assert_called_once_with(engine, [job], "owner", metrics)
        lease.return_value.__exit__.assert_called_once()
        defer.assert_not_called()


def test_shutdown_after_claim_returns_the_job_without_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = MagicMock()
    stop = Event()
    job = _job()

    def claim(*_args: object, **_kwargs: object) -> list[ClaimedJob]:
        stop.set()
        return [job]

    defer = MagicMock()
    dispatch = MagicMock()
    monkeypatch.setattr(worker, "claim_jobs", claim)
    monkeypatch.setattr(worker, "_defer", defer)
    monkeypatch.setattr(worker, "dispatch_claimed_jobs", dispatch)
    metrics = _metrics()
    worker.consume_jobs(engine, Settings(), "owner", stop, 1, metrics)
    defer.assert_called_once_with(engine, job, "owner")
    dispatch.assert_not_called()
    assert (
        _metric_value(
            metrics,
            "adept_engine_worker_job_deferrals_total",
            {"job_type": job.job_type, "reason": "shutdown"},
        )
        == 1
    )
    assert _metric_samples(metrics, "adept_engine_worker_job_attempts_total") == []


def test_metrics_failure_does_not_change_durable_deferral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job()
    metrics = MagicMock()
    metrics.record_deferral.side_effect = RuntimeError("metrics unavailable")
    defer = MagicMock()
    monkeypatch.setattr(worker, "_defer", defer)

    worker._defer_with_metrics(
        MagicMock(),
        job,
        "test-worker",
        metrics,
        "scope_busy",
    )

    defer.assert_called_once()
    metrics.record_deferral.assert_called_once_with(job.job_type, "scope_busy")


def test_idle_poll_is_successful_while_consumer_is_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = MagicMock()
    stop = Event()
    metrics = _metrics()
    claim = MagicMock(return_value=[])

    def schema_version(_engine: object) -> str:
        assert (
            _metric_value(
                metrics,
                "adept_engine_worker_thread_alive",
                {"thread_slot": "1"},
            )
            == 1
        )
        stop.set()
        return "15"

    monkeypatch.setattr(worker, "claim_jobs", claim)
    monkeypatch.setattr(worker, "current_schema_version", schema_version)

    worker.consume_jobs(engine, Settings(), "owner", stop, 1, metrics)

    assert (
        _metric_value(
            metrics,
            "adept_engine_worker_thread_last_successful_poll_timestamp_seconds",
            {"thread_slot": "1"},
        )
        > 0
    )
    assert _metric_samples(metrics, "adept_engine_worker_poll_errors_total") == []
    assert (
        _metric_value(
            metrics,
            "adept_engine_worker_thread_alive",
            {"thread_slot": "1"},
        )
        == 0
    )


def test_claim_poll_error_is_recorded_without_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = MagicMock()
    stop = Event()
    metrics = _metrics()

    def claim(*_args: object, **_kwargs: object) -> list[ClaimedJob]:
        stop.set()
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(worker, "claim_jobs", claim)

    worker.consume_jobs(engine, Settings(), "owner", stop, 1, metrics)

    assert (
        _metric_value(
            metrics,
            "adept_engine_worker_poll_errors_total",
            {"thread_slot": "1"},
        )
        == 1
    )
    assert (
        _metric_value(
            metrics,
            "adept_engine_worker_thread_last_successful_poll_timestamp_seconds",
            {"thread_slot": "1"},
        )
        == 0
    )


def test_lease_acquisition_failure_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = MagicMock()
    stop = Event()
    metrics = _metrics()
    job = _job()
    lease = MagicMock()

    def fail_lease() -> None:
        stop.set()
        raise RuntimeError("lease database failure")

    lease.return_value.__enter__.side_effect = fail_lease
    monkeypatch.setattr(worker, "claim_jobs", MagicMock(return_value=[job]))
    monkeypatch.setattr(worker, "JobLease", lease)

    worker.consume_jobs(engine, Settings(), "owner", stop, 1, metrics)

    assert (
        _metric_value(
            metrics,
            "adept_engine_worker_lease_failures_total",
            {"thread_slot": "1", "phase": "acquire"},
        )
        == 1
    )
    assert _metric_samples(metrics, "adept_engine_worker_job_attempts_total") == []


def test_lease_connection_loss_is_recorded_before_fail_closed_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = _metrics()
    exit_process = MagicMock()
    monkeypatch.setattr(worker.os, "_exit", exit_process)

    worker._connection_lost(metrics, 1)

    assert (
        _metric_value(
            metrics,
            "adept_engine_worker_lease_failures_total",
            {"thread_slot": "1", "phase": "connection_lost"},
        )
        == 1
    )
    exit_process.assert_called_once_with(1)


def test_metrics_failure_cannot_prevent_fail_closed_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = MagicMock()
    metrics.record_lease_failure.side_effect = RuntimeError("metrics unavailable")
    exit_process = MagicMock()
    monkeypatch.setattr(worker.os, "_exit", exit_process)

    worker._connection_lost(metrics, 1)

    exit_process.assert_called_once_with(1)


def test_stale_recovery_callback_records_committed_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = MagicMock()
    stop = Event()
    metrics = _metrics()

    def claim(*_args: object, **kwargs: object) -> list[ClaimedJob]:
        callback = kwargs["on_stale_recovery"]
        assert callable(callback)
        callback(StaleJobRecovery(retryable=2, dead=1))
        stop.set()
        return []

    monkeypatch.setattr(worker, "claim_jobs", claim)
    monkeypatch.setattr(worker, "current_schema_version", MagicMock(return_value="15"))

    worker.consume_jobs(engine, Settings(), "owner", stop, 1, metrics)

    assert (
        _metric_value(
            metrics,
            "adept_engine_worker_stale_jobs_recovered_total",
            {"outcome": "retry_scheduled"},
        )
        == 2
    )
    assert (
        _metric_value(
            metrics,
            "adept_engine_worker_stale_jobs_recovered_total",
            {"outcome": "dead_lettered"},
        )
        == 1
    )


def test_pool_runs_two_distinct_consumers_and_waits_for_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = Barrier(2, timeout=5)
    owners: list[str] = []
    finished: list[str] = []
    engine = MagicMock()
    handlers: dict[int, object] = {}

    def register(signum: int, handler: object) -> object:
        previous = handlers.get(signum)
        handlers[signum] = handler
        return previous

    def consume(
        db: object,
        _settings: Settings,
        owner: str,
        stop: Event,
        slot: int,
        _metrics: WorkerMetrics,
    ) -> None:
        assert db is engine
        assert slot in (1, 2)
        owners.append(owner)
        barrier.wait()
        handler = handlers[worker.signal.SIGTERM]
        assert callable(handler)
        handler(worker.signal.SIGTERM, None)
        assert stop.is_set()
        assert endpoint.start.call_count == 1
        assert endpoint.stop.call_count == 0
        assert queue_metrics.start.call_count == 1
        assert queue_metrics.stop.call_count == 0
        finished.append(owner)

    monkeypatch.setattr(worker, "configure_logging", lambda: None)
    monkeypatch.setattr(worker, "get_settings", Settings)
    monkeypatch.setattr(worker, "get_database_engine", lambda: engine)
    monkeypatch.setattr(worker, "current_schema_version", lambda _: "15")
    monkeypatch.setattr(worker, "consume_jobs", consume)
    monkeypatch.setattr(worker.signal, "signal", register)
    endpoint = MagicMock()
    queue_metrics = MagicMock()

    def stop_queue_metrics() -> bool:
        assert sorted(finished) == sorted(owners)
        assert endpoint.stop.call_count == 0
        return True

    def stop_endpoint() -> None:
        assert sorted(finished) == sorted(owners)
        assert queue_metrics.stop.call_count == 1

    queue_metrics.stop.side_effect = stop_queue_metrics
    endpoint.stop.side_effect = stop_endpoint
    monkeypatch.setattr(worker, "MetricsEndpoint", MagicMock(return_value=endpoint))
    monkeypatch.setattr(worker, "QueueMetricsSampler", MagicMock(return_value=queue_metrics))
    worker.run()
    assert len(set(owners)) == 2
    assert sorted(finished) == sorted(owners)
    assert all(handler is None for handler in handlers.values())
    endpoint.start.assert_called_once_with()
    endpoint.stop.assert_called_once_with()
    queue_metrics.start.assert_called_once_with()
    queue_metrics.stop.assert_called_once_with()
    engine.dispose.assert_called_once()
