from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import Engine

from app.jobs.claimer import claim_jobs
from tests.conftest import JobFactory

pytestmark = pytest.mark.integration


def test_two_workers_claim_distinct_jobs(
    database_engine: Engine,
    job_factory: JobFactory,
) -> None:
    first_id = job_factory.insert(priority=10)
    second_id = job_factory.insert(priority=10)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(claim_jobs, database_engine, "engine-test-worker-1", 1)
        second_future = executor.submit(claim_jobs, database_engine, "engine-test-worker-2", 1)
        first_claim = first_future.result(timeout=10)
        second_claim = second_future.result(timeout=10)

    claimed_ids = {first_claim[0].id, second_claim[0].id}
    assert claimed_ids == {first_id, second_id}
    assert first_claim[0].id != second_claim[0].id


def test_claim_filters_and_ordering(
    database_engine: Engine,
    job_factory: JobFactory,
) -> None:
    later_priority = job_factory.insert(priority=20, created_offset_seconds=-20)
    first_priority = job_factory.insert(priority=10, created_offset_seconds=-10)
    future = job_factory.insert(priority=1, available_offset_seconds=3600)
    running = job_factory.insert(status="RUNNING", priority=1)
    dead = job_factory.insert(status="DEAD", priority=1)
    exhausted = job_factory.insert(status="FAILED", attempts=8, max_attempts=8, priority=1)

    claimed = claim_jobs(database_engine, "engine-test-worker", 10)

    assert [job.id for job in claimed] == [first_priority, later_priority]
    assert future not in {job.id for job in claimed}
    assert running not in {job.id for job in claimed}
    assert dead not in {job.id for job in claimed}
    assert exhausted not in {job.id for job in claimed}


def test_claim_updates_owner_attempt_and_version(
    database_engine: Engine,
    job_factory: JobFactory,
) -> None:
    job_id = job_factory.insert(attempts=2)

    claimed = claim_jobs(database_engine, "engine-test-worker", 1)
    row = job_factory.row(job_id)

    assert claimed[0].id == job_id
    assert row["status"] == "RUNNING"
    assert row["locked_by"] == "engine-test-worker"
    assert row["locked_at"] is not None
    assert row["attempts"] == 3
    assert row["version"] == 1
