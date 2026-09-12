import math
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.parser import text_string_to_metric_families

from app.jobs.queue_metrics import QueueSnapshot
from app.monitoring import worker_metrics
from app.monitoring.worker_metrics import MetricsEndpoint, WorkerMetrics


def _samples(
    registry: CollectorRegistry,
    name: str,
) -> list[tuple[dict[str, str], float]]:
    exposition = generate_latest(registry).decode("utf-8")
    return [
        (sample.labels, float(sample.value))
        for family in text_string_to_metric_families(exposition)
        for sample in family.samples
        if sample.name == name
    ]


def _sample_value(
    registry: CollectorRegistry,
    name: str,
    labels: dict[str, str] | None = None,
) -> float:
    expected_labels = labels or {}
    matches = [
        value
        for sample_labels, value in _samples(registry, name)
        if sample_labels == expected_labels
    ]
    assert len(matches) == 1, f"expected one {name}{expected_labels} sample, got {matches}"
    return matches[0]


@pytest.mark.parametrize(
    ("configured_threads", "expected_slots"),
    [(1, {"1"}), (2, {"1", "2"})],
)
def test_thread_series_are_initialized_for_configured_slots_only(
    configured_threads: int,
    expected_slots: set[str],
) -> None:
    metrics = WorkerMetrics(configured_threads, {"KNOWN"})

    assert (
        _sample_value(metrics.registry, "adept_engine_worker_configured_threads")
        == configured_threads
    )
    for metric_name in (
        "adept_engine_worker_thread_alive",
        "adept_engine_worker_thread_active",
        "adept_engine_worker_thread_last_successful_poll_timestamp_seconds",
        "adept_engine_worker_thread_active_job_duration_seconds",
    ):
        samples = _samples(metrics.registry, metric_name)
        assert {labels["thread_slot"] for labels, _ in samples} == expected_slots
        assert all(value == 0 for _, value in samples)

    with pytest.raises(ValueError, match="unknown thread slot"):
        metrics.mark_thread_started(configured_threads + 1)


def test_active_job_duration_progresses_and_resets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    monkeypatch.setattr(worker_metrics, "monotonic", lambda: clock[0])
    metrics = WorkerMetrics(1, {"KNOWN"})

    metrics.mark_job_started(1)
    clock[0] = 103.25

    assert (
        _sample_value(
            metrics.registry,
            "adept_engine_worker_thread_active",
            {"thread_slot": "1"},
        )
        == 1
    )
    assert _sample_value(
        metrics.registry,
        "adept_engine_worker_thread_active_job_duration_seconds",
        {"thread_slot": "1"},
    ) == pytest.approx(3.25)

    clock[0] = 108.5
    assert _sample_value(
        metrics.registry,
        "adept_engine_worker_thread_active_job_duration_seconds",
        {"thread_slot": "1"},
    ) == pytest.approx(8.5)

    metrics.mark_job_stopped(1)
    assert (
        _sample_value(
            metrics.registry,
            "adept_engine_worker_thread_active",
            {"thread_slot": "1"},
        )
        == 0
    )
    assert (
        _sample_value(
            metrics.registry,
            "adept_engine_worker_thread_active_job_duration_seconds",
            {"thread_slot": "1"},
        )
        == 0
    )


def test_unknown_job_types_are_bounded_and_outcomes_are_counted() -> None:
    metrics = WorkerMetrics(1, {"KNOWN"})

    metrics.record_dispatch_outcome("KNOWN", "succeeded", 1.0)
    metrics.record_dispatch_outcome("unbounded-a", "retry_scheduled", 2.0)
    metrics.record_dispatch_outcome("unbounded-b", "dead_lettered", 3.0)
    metrics.record_dispatch_outcome("unbounded-c", "continuation_requeued", 4.0)
    metrics.record_dispatch_error("unbounded-d", 5.0)
    metrics.record_deferral("unbounded-e", "scope_busy")

    assert (
        _sample_value(
            metrics.registry,
            "adept_engine_worker_job_attempts_total",
            {"job_type": "KNOWN", "outcome": "succeeded"},
        )
        == 1
    )
    assert (
        _sample_value(
            metrics.registry,
            "adept_engine_worker_job_attempts_total",
            {"job_type": "UNKNOWN", "outcome": "retry_scheduled"},
        )
        == 1
    )
    assert (
        _sample_value(
            metrics.registry,
            "adept_engine_worker_job_attempts_total",
            {"job_type": "UNKNOWN", "outcome": "dead_lettered"},
        )
        == 1
    )
    assert (
        _sample_value(
            metrics.registry,
            "adept_engine_worker_job_deferrals_total",
            {"job_type": "UNKNOWN", "reason": "continuation"},
        )
        == 1
    )
    assert (
        _sample_value(
            metrics.registry,
            "adept_engine_worker_job_deferrals_total",
            {"job_type": "UNKNOWN", "reason": "scope_busy"},
        )
        == 1
    )
    assert (
        _sample_value(
            metrics.registry,
            "adept_engine_worker_job_operation_errors_total",
            {"job_type": "UNKNOWN", "operation": "dispatch"},
        )
        == 1
    )

    attempt_labels = {
        tuple(sorted(labels.items()))
        for labels, _ in _samples(metrics.registry, "adept_engine_worker_job_attempts_total")
    }
    assert attempt_labels == {
        (("job_type", "KNOWN"), ("outcome", "succeeded")),
        (("job_type", "UNKNOWN"), ("outcome", "dead_lettered")),
        (("job_type", "UNKNOWN"), ("outcome", "retry_scheduled")),
    }
    exposition = generate_latest(metrics.registry).decode("utf-8")
    assert all(
        unbounded not in exposition
        for unbounded in (
            "unbounded-a",
            "unbounded-b",
            "unbounded-c",
            "unbounded-d",
            "unbounded-e",
        )
    )

    with pytest.raises(ValueError, match="unknown dispatch outcome"):
        metrics.record_dispatch_outcome("KNOWN", "arbitrary", 1.0)
    with pytest.raises(ValueError, match="unknown deferral reason"):
        metrics.record_deferral("KNOWN", "arbitrary")


def test_queue_error_retains_last_successful_values_and_freshness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [1_000.0]
    monkeypatch.setattr(worker_metrics, "time", lambda: clock[0])
    metrics = WorkerMetrics(1, {"KNOWN"})
    queue_names = (
        "adept_engine_worker_queue_ready_jobs",
        "adept_engine_worker_queue_oldest_ready_wait_seconds",
        "adept_engine_worker_queue_running_jobs",
        "adept_engine_worker_queue_dead_letter_jobs",
    )

    assert all(math.isnan(_sample_value(metrics.registry, name)) for name in queue_names)
    assert _sample_value(metrics.registry, "adept_engine_worker_queue_collection_success") == 0
    assert (
        _sample_value(
            metrics.registry,
            "adept_engine_worker_queue_last_success_timestamp_seconds",
        )
        == 0
    )

    metrics.record_queue_snapshot(
        QueueSnapshot(
            ready=7,
            oldest_ready_wait_seconds=42.5,
            running=2,
            dead_letter=3,
        )
    )
    assert _sample_value(metrics.registry, queue_names[0]) == 7
    assert _sample_value(metrics.registry, queue_names[1]) == 42.5
    assert _sample_value(metrics.registry, queue_names[2]) == 2
    assert _sample_value(metrics.registry, queue_names[3]) == 3
    assert _sample_value(metrics.registry, "adept_engine_worker_queue_collection_success") == 1
    assert (
        _sample_value(
            metrics.registry,
            "adept_engine_worker_queue_last_success_timestamp_seconds",
        )
        == 1_000
    )

    clock[0] = 1_120.0
    metrics.record_queue_collection_error()

    assert _sample_value(metrics.registry, queue_names[0]) == 7
    assert _sample_value(metrics.registry, queue_names[1]) == 42.5
    assert _sample_value(metrics.registry, queue_names[2]) == 2
    assert _sample_value(metrics.registry, queue_names[3]) == 3
    assert _sample_value(metrics.registry, "adept_engine_worker_queue_collection_success") == 0
    assert (
        _sample_value(
            metrics.registry,
            "adept_engine_worker_queue_last_success_timestamp_seconds",
        )
        == 1_000
    )
    assert (
        _sample_value(
            metrics.registry,
            "adept_engine_worker_queue_collection_errors_total",
        )
        == 1
    )


def test_two_slots_update_metrics_concurrently() -> None:
    metrics = WorkerMetrics(2, {"KNOWN"})
    both_active = Barrier(3, timeout=5)
    release = Event()

    def update(slot: int) -> None:
        metrics.mark_thread_started(slot)
        metrics.record_poll_success(slot)
        metrics.mark_job_started(slot)
        both_active.wait()
        assert release.wait(5)
        metrics.record_dispatch_outcome("KNOWN", "succeeded", float(slot))
        metrics.mark_job_stopped(slot)
        metrics.mark_thread_stopped(slot)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(update, slot) for slot in (1, 2)]
        both_active.wait()
        try:
            for slot in ("1", "2"):
                labels = {"thread_slot": slot}
                assert (
                    _sample_value(metrics.registry, "adept_engine_worker_thread_alive", labels) == 1
                )
                assert (
                    _sample_value(metrics.registry, "adept_engine_worker_thread_active", labels)
                    == 1
                )
                assert (
                    _sample_value(
                        metrics.registry,
                        "adept_engine_worker_thread_last_successful_poll_timestamp_seconds",
                        labels,
                    )
                    > 0
                )
        finally:
            release.set()
        for future in futures:
            future.result(timeout=5)

    for slot in ("1", "2"):
        labels = {"thread_slot": slot}
        assert _sample_value(metrics.registry, "adept_engine_worker_thread_alive", labels) == 0
        assert _sample_value(metrics.registry, "adept_engine_worker_thread_active", labels) == 0
    assert (
        _sample_value(
            metrics.registry,
            "adept_engine_worker_job_attempts_total",
            {"job_type": "KNOWN", "outcome": "succeeded"},
        )
        == 2
    )


def test_metrics_endpoint_scrape_stop_and_port_rebind() -> None:
    metrics = WorkerMetrics(1, {"KNOWN"})
    endpoint = MetricsEndpoint(metrics.registry, "127.0.0.1", 0)

    with pytest.raises(RuntimeError, match="not running"):
        _ = endpoint.bound_port

    endpoint.start()
    port = endpoint.bound_port
    try:
        with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as response:
            exposition = response.read().decode("utf-8")
        assert response.status == 200
        assert "adept_engine_worker_configured_threads 1.0" in exposition
        with pytest.raises(RuntimeError, match="already running"):
            endpoint.start()
    finally:
        endpoint.stop()

    with pytest.raises(RuntimeError, match="not running"):
        _ = endpoint.bound_port
    with pytest.raises(URLError):
        urlopen(f"http://127.0.0.1:{port}/metrics", timeout=0.5)
    endpoint.stop()

    rebound = MetricsEndpoint(metrics.registry, "127.0.0.1", port)
    rebound.start()
    try:
        assert rebound.bound_port == port
        with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as response:
            assert response.status == 200
    finally:
        rebound.stop()
