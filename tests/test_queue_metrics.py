import pytest
from sqlalchemy import Engine

from app.jobs.queue_metrics import QueueSnapshot, collect_queue_snapshot
from tests.conftest import JobFactory

pytestmark = pytest.mark.integration


def test_empty_queue_snapshot(
    database_engine: Engine,
) -> None:
    assert collect_queue_snapshot(database_engine) == QueueSnapshot(
        ready=0,
        oldest_ready_wait_seconds=0.0,
        running=0,
        dead_letter=0,
    )


def test_pending_and_failed_jobs_are_ready(
    database_engine: Engine,
    job_factory: JobFactory,
) -> None:
    job_factory.insert(status="PENDING", available_offset_seconds=-120)
    job_factory.insert(status="FAILED", available_offset_seconds=-30)

    snapshot = collect_queue_snapshot(database_engine)

    assert snapshot.ready == 2
    assert snapshot.oldest_ready_wait_seconds == pytest.approx(120, abs=5)
    assert snapshot.running == 0
    assert snapshot.dead_letter == 0


def test_future_and_exhausted_jobs_are_not_ready(
    database_engine: Engine,
    job_factory: JobFactory,
) -> None:
    job_factory.insert(status="PENDING", available_offset_seconds=3600)
    job_factory.insert(
        status="FAILED",
        attempts=8,
        max_attempts=8,
        available_offset_seconds=-3600,
    )

    snapshot = collect_queue_snapshot(database_engine)

    assert snapshot.ready == 0
    assert snapshot.oldest_ready_wait_seconds == 0.0


def test_running_and_dead_letter_jobs_are_counted(
    database_engine: Engine,
    job_factory: JobFactory,
) -> None:
    job_factory.insert(status="RUNNING", locked_by="test-worker", locked_offset_seconds=0)
    job_factory.insert(status="DEAD")

    snapshot = collect_queue_snapshot(database_engine)

    assert snapshot.ready == 0
    assert snapshot.running == 1
    assert snapshot.dead_letter == 1


def test_old_creation_time_does_not_make_future_job_ready(
    database_engine: Engine,
    job_factory: JobFactory,
) -> None:
    job_factory.insert(
        status="PENDING",
        created_offset_seconds=-86_400,
        available_offset_seconds=3600,
    )

    snapshot = collect_queue_snapshot(database_engine)

    assert snapshot.ready == 0
    assert snapshot.oldest_ready_wait_seconds == 0.0
