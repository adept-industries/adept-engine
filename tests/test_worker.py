from datetime import UTC, datetime
from threading import Barrier, Event
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app import worker
from app.core.config import Settings
from app.db.models import ClaimedJob
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

    worker.consume_jobs(engine, Settings(), "owner", stop)

    claim.assert_called_once_with(engine, "owner", limit=1, stale_after_seconds=900)
    lease.return_value.__enter__.assert_called_once()
    if busy:
        dispatch.assert_not_called()
        defer.assert_called_once_with(engine, job, "owner")
    else:
        dispatch.assert_called_once_with(engine, [job], "owner")
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
    worker.consume_jobs(engine, Settings(), "owner", stop)
    defer.assert_called_once_with(engine, job, "owner")
    dispatch.assert_not_called()


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

    def consume(db: object, _settings: Settings, owner: str, stop: Event) -> None:
        assert db is engine
        owners.append(owner)
        barrier.wait()
        handler = handlers[worker.signal.SIGTERM]
        assert callable(handler)
        handler(worker.signal.SIGTERM, None)
        assert stop.is_set()
        finished.append(owner)

    monkeypatch.setattr(worker, "configure_logging", lambda: None)
    monkeypatch.setattr(worker, "get_settings", Settings)
    monkeypatch.setattr(worker, "get_database_engine", lambda: engine)
    monkeypatch.setattr(worker, "current_schema_version", lambda _: "15")
    monkeypatch.setattr(worker, "consume_jobs", consume)
    monkeypatch.setattr(worker.signal, "signal", register)
    worker.run()
    assert len(set(owners)) == 2
    assert sorted(finished) == sorted(owners)
    assert all(handler is None for handler in handlers.values())
    engine.dispose.assert_called_once()
