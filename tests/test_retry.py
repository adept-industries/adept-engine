import pytest
from sqlalchemy import Engine

from app.jobs.claimer import claim_jobs
from app.jobs.retry import (
    JobOwnershipError,
    mark_failed,
    mark_succeeded,
    retry_delay_seconds,
)
from tests.conftest import JobFactory


def test_retry_delay_is_exponential_and_capped() -> None:
    assert retry_delay_seconds(3, jitter_seconds=0) == 8
    assert retry_delay_seconds(20, jitter_seconds=10) == 900


@pytest.mark.integration
def test_only_owner_can_complete_job(
    database_engine: Engine,
    job_factory: JobFactory,
) -> None:
    job_id = job_factory.insert()
    claim_jobs(database_engine, "owner-worker", 1)

    with pytest.raises(JobOwnershipError):
        mark_succeeded(database_engine, job_id, "different-worker")

    mark_succeeded(database_engine, job_id, "owner-worker")
    row = job_factory.row(job_id)
    assert row["status"] == "SUCCEEDED"
    assert row["locked_by"] is None
    assert row["finished_at"] is not None


@pytest.mark.integration
def test_failure_retries_or_becomes_dead(
    database_engine: Engine,
    job_factory: JobFactory,
) -> None:
    retry_id = job_factory.insert(attempts=0, max_attempts=3)
    claim_jobs(database_engine, "retry-worker", 1)
    assert (
        mark_failed(
    database_engine,
    retry_id,
    "retry-worker",
    "temporary provider error\nwith details",
    jitter_seconds=0,
)
        == "FAILED"
    )
    retry_row = job_factory.row(retry_id)
    assert retry_row["status"] == "FAILED"
    assert retry_row["locked_by"] is None
    assert retry_row["last_error"] == "temporary provider error with details"

    dead_id = job_factory.insert(attempts=2, max_attempts=3)
    claim_jobs(database_engine, "dead-worker", 1)
    assert (
        mark_failed(
    database_engine,
    dead_id,
    "dead-worker",
    "final failure",
    jitter_seconds=0,
)
        == "DEAD"
    )
    dead_row = job_factory.row(dead_id)
    assert dead_row["status"] == "DEAD"
    assert dead_row["finished_at"] is not None
