from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.db.models import ClaimedJob
from app.worker import dispatch_claimed_jobs


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
