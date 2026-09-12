from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import Connection, Engine, text

from app import worker
from app.core.config import Settings
from app.db.models import ClaimedJob
from app.jobs import lease
from app.jobs.claimer import claim_jobs
from app.jobs.dispatcher import HANDLERS, JobDispatchOutcome
from app.jobs.retry import JobOwnershipError, mark_succeeded
from app.monitoring.worker_metrics import WorkerMetrics
from tests.conftest import JobFactory

pytestmark = pytest.mark.usefixtures("isolated_worker_environment")


def _guard(engine: Engine, job: ClaimedJob, owner: str) -> lease.JobLease:
    return lease.JobLease(engine, job, owner, interval_seconds=60, on_connection_lost=MagicMock())


@pytest.mark.integration
@pytest.mark.parametrize("max_attempts", [1, 8])
def test_live_job_is_not_recovered_even_if_heartbeat_is_overdue(
    database_engine: Engine, job_factory: JobFactory, max_attempts: int
) -> None:
    job_id = job_factory.insert(max_attempts=max_attempts)
    job = claim_jobs(database_engine, "first", limit=1)[0]
    with _guard(database_engine, job, "first"):
        with database_engine.begin() as conn:
            conn.execute(
                text("UPDATE processing_jobs SET locked_at=now()-interval '1 hour' WHERE id=:id"),
                {"id": job_id},
            )
        assert claim_jobs(database_engine, "second", stale_after_seconds=30) == []
        assert job_factory.row(job_id)["status"] == "RUNNING"
    # After the guard connection closes, existing retry/dead-letter rules apply.
    recovered = claim_jobs(database_engine, "second", stale_after_seconds=30)
    if max_attempts == 1:
        assert recovered == []
        assert job_factory.row(job_id)["status"] == "DEAD"
    else:
        assert [item.id for item in recovered] == [job_id]
        assert recovered[0].attempts == 2


@pytest.mark.integration
def test_same_repo_is_serialized_but_different_repos_can_run_together(
    database_engine: Engine, job_factory: JobFactory
) -> None:
    workspace_id, repo_id = uuid4(), uuid4()
    for repository in (repo_id, repo_id, uuid4()):
        job_factory.insert(
            payload={"workspaceId": str(workspace_id), "repositoryId": str(repository)}
        )
    first, same, different = claim_jobs(database_engine, "owner", limit=3)
    with _guard(database_engine, first, "owner"):
        with pytest.raises(lease.JobScopeBusy), _guard(database_engine, same, "owner"):
            pytest.fail("same-repository work must wait")
        with _guard(database_engine, different, "owner"):
            pass
    with _guard(database_engine, same, "owner"):
        pass


@pytest.mark.integration
@pytest.mark.parametrize("workspace_first", [False, True])
def test_workspace_jobs_exclude_repository_work_in_both_directions(
    database_engine: Engine, job_factory: JobFactory, workspace_first: bool
) -> None:
    workspace_id = uuid4()
    job_factory.insert(payload={"workspaceId": str(workspace_id), "repositoryId": str(uuid4())})
    # A deletion must ignore any extraneous repository ID in its payload.
    job_factory.insert(
        job_type="DELETE_WORKSPACE",
        payload={"workspaceId": str(workspace_id), "repositoryId": str(uuid4())},
    )
    first, second = claim_jobs(database_engine, "owner", limit=2)
    if workspace_first:
        first, second = second, first
    with (
        _guard(database_engine, first, "owner"),
        pytest.raises(lease.JobScopeBusy),
        _guard(database_engine, second, "owner"),
    ):
        pytest.fail("workspace-wide and repository work must not overlap")
    with _guard(database_engine, second, "owner"):
        pass


@pytest.mark.integration
def test_guard_checks_ownership_and_releases_locks_after_handler_exception(
    database_engine: Engine, job_factory: JobFactory
) -> None:
    job_factory.insert()
    job = claim_jobs(database_engine, "owner", limit=1)[0]
    with pytest.raises(JobOwnershipError), _guard(database_engine, job, "wrong-owner"):
        pytest.fail("stolen claims must not execute")
    with pytest.raises(ValueError, match="handler"), _guard(database_engine, job, "owner"):
        raise ValueError("handler")
    with _guard(database_engine, job, "owner"):
        pass


@pytest.mark.integration
def test_heartbeat_refreshes_and_skips_rows_locked_by_the_handler(
    database_engine: Engine, job_factory: JobFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_factory.insert()
    job = claim_jobs(database_engine, "owner", limit=1)[0]
    heartbeat_after_row_lock = Event()
    row_locked = Event()
    original = lease._heartbeat

    def heartbeat(conn: Connection, claimed: ClaimedJob, owner: str) -> None:
        original(conn, claimed, owner)
        if row_locked.is_set():
            heartbeat_after_row_lock.set()

    monkeypatch.setattr(lease, "_heartbeat", heartbeat)
    failed = MagicMock()
    before = job_factory.row(job.id)["locked_at"]
    with lease.JobLease(
        database_engine, job, "owner", interval_seconds=0.01, on_connection_lost=failed
    ):
        assert job_factory.row(job.id)["locked_at"] > before
        with database_engine.begin() as conn:
            conn.execute(
                text("SELECT id FROM processing_jobs WHERE id=:id FOR UPDATE"), {"id": job.id}
            )
            row_locked.set()
            assert heartbeat_after_row_lock.wait(3), "heartbeat blocked on the handler's row lock"
    failed.assert_not_called()


@pytest.mark.integration
def test_connection_loss_invokes_fail_closed_callback(
    database_engine: Engine, job_factory: JobFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_factory.insert()
    job = claim_jobs(database_engine, "owner", limit=1)[0]
    fail = Event()
    failed = Event()
    original = lease._heartbeat

    def heartbeat(conn: Connection, claimed: ClaimedJob, owner: str) -> None:
        if fail.is_set():
            raise ConnectionError("simulated lost database session")
        original(conn, claimed, owner)

    monkeypatch.setattr(lease, "_heartbeat", heartbeat)
    with lease.JobLease(
        database_engine, job, "owner", interval_seconds=0.01, on_connection_lost=failed.set
    ):
        fail.set()
        assert failed.wait(3)
    monkeypatch.setattr(lease, "_heartbeat", original)
    with _guard(database_engine, job, "owner"):
        pass


@pytest.mark.integration
def test_two_consumers_process_distinct_jobs_concurrently_and_drain_on_stop(
    database_engine: Engine, job_factory: JobFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace_id = uuid4()
    ids = [
        job_factory.insert(payload={"workspaceId": str(workspace_id), "repositoryId": str(uuid4())})
        for _ in range(2)
    ]
    both_active = Barrier(3, timeout=5)
    release = Event()
    stop = Event()
    observed: list[tuple[str, str]] = []
    record_lock = Lock()

    def dispatch(engine: Engine, job: ClaimedJob, owner: str) -> JobDispatchOutcome:
        with record_lock:
            observed.append((str(job.id), owner))
        both_active.wait()  # Fails if processing silently becomes sequential.
        assert release.wait(5)
        stop.set()  # In-flight handlers must finish even after shutdown starts.
        mark_succeeded(engine, job.id, owner)
        return JobDispatchOutcome.SUCCEEDED

    monkeypatch.setattr(worker, "dispatch_job", dispatch)
    settings = Settings(engine_poll_interval_ms=100)
    metrics = WorkerMetrics(2, HANDLERS)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                worker.consume_jobs,
                database_engine,
                settings,
                f"owner-{slot}",
                stop,
                slot,
                metrics,
            )
            for slot in (1, 2)
        ]
        try:
            both_active.wait()
            for slot in ("1", "2"):
                assert (
                    metrics.registry.get_sample_value(
                        "adept_engine_worker_thread_active",
                        {"thread_slot": slot},
                    )
                    == 1
                )
            release.set()
            for future in futures:
                future.result(timeout=10)
        finally:
            release.set()
            stop.set()
    assert {job for job, _ in observed} == {str(job_id) for job_id in ids}
    assert len({owner for _, owner in observed}) == 2
    assert all(job_factory.row(job_id)["status"] == "SUCCEEDED" for job_id in ids)
    assert (
        metrics.registry.get_sample_value(
            "adept_engine_worker_job_attempts_total",
            {"job_type": "RECALCULATE_METRICS", "outcome": "succeeded"},
        )
        == 2
    )


@pytest.mark.integration
def test_busy_scope_is_requeued_without_consuming_an_attempt(
    database_engine: Engine, job_factory: JobFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"workspaceId": str(uuid4()), "repositoryId": str(uuid4())}
    job_factory.insert(payload=payload)
    first = claim_jobs(database_engine, "first", limit=1)[0]
    queued_id = job_factory.insert(payload=payload)
    stop = Event()
    original = worker._defer
    dispatch = MagicMock()
    metrics = WorkerMetrics(2, HANDLERS)

    def defer(engine: Engine, job: ClaimedJob, owner: str) -> None:
        original(engine, job, owner)
        stop.set()

    monkeypatch.setattr(worker, "_defer", defer)
    monkeypatch.setattr(worker, "dispatch_job", dispatch)
    with _guard(database_engine, first, "first"):
        worker.consume_jobs(
            database_engine,
            Settings(),
            "second",
            stop,
            1,
            metrics,
        )
    dispatch.assert_not_called()
    row = job_factory.row(queued_id)
    assert row["status"] == "PENDING"
    assert row["attempts"] == 0
    assert row["locked_by"] is None
    assert row["payload"] == payload
    assert (
        metrics.registry.get_sample_value(
            "adept_engine_worker_job_deferrals_total",
            {"job_type": "RECALCULATE_METRICS", "reason": "scope_busy"},
        )
        == 1
    )
    for outcome in ("succeeded", "retry_scheduled", "dead_lettered"):
        assert metrics.registry.get_sample_value(
            "adept_engine_worker_job_attempts_total",
            {"job_type": "RECALCULATE_METRICS", "outcome": outcome},
        ) in (None, 0)


@pytest.mark.integration
def test_paginated_requeue_cannot_run_again_until_original_handler_exits(
    database_engine: Engine, job_factory: JobFactory
) -> None:
    from app.jobs.retry import RequeueWithPayloadError, requeue_with_payload

    job_factory.insert()
    job = claim_jobs(database_engine, "first", limit=1)[0]
    with _guard(database_engine, job, "first"):
        with pytest.raises(RequeueWithPayloadError):
            requeue_with_payload(database_engine, job.id, "first", {"page": 2})
        next_page = claim_jobs(database_engine, "second", limit=1)[0]
        with pytest.raises(lease.JobScopeBusy), _guard(database_engine, next_page, "second"):
            pytest.fail("previous handler still holds the job's execution lock")
    with _guard(database_engine, next_page, "second"):
        assert next_page.payload == {"page": 2}


@pytest.mark.integration
def test_busy_repository_does_not_prevent_claiming_other_ready_work(
    database_engine: Engine, job_factory: JobFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"workspaceId": str(uuid4()), "repositoryId": str(uuid4())}
    job_factory.insert(payload=payload)
    running = claim_jobs(database_engine, "first", limit=1)[0]
    waiting = job_factory.insert(payload=payload, priority=1)
    other = job_factory.insert(payload={"repositoryId": str(uuid4())}, priority=100)
    stop = Event()

    def dispatch(engine: Engine, job: ClaimedJob, owner: str) -> JobDispatchOutcome:
        assert job.id == other
        mark_succeeded(engine, job.id, owner)
        stop.set()
        return JobDispatchOutcome.SUCCEEDED

    monkeypatch.setattr(worker, "dispatch_job", dispatch)
    with _guard(database_engine, running, "first"), ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            worker.consume_jobs,
            database_engine,
            Settings(),
            "second",
            stop,
            1,
            WorkerMetrics(2, HANDLERS),
        )
        try:
            future.result(timeout=3)
        finally:
            stop.set()
    assert job_factory.row(waiting)["status"] == "PENDING"
    assert job_factory.row(waiting)["attempts"] == 0
    assert job_factory.row(other)["status"] == "SUCCEEDED"
