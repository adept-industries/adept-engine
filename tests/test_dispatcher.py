from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

from app.db.models import ClaimedJob
from app.jobs.dispatcher import HANDLERS, dispatch_job


def test_dispatcher_handles_backfill_repository():
    engine = MagicMock()
    job = ClaimedJob(
        id=uuid4(),
        job_type="BACKFILL_REPOSITORY",
        payload={"repositoryId": str(uuid4()), "backfillDays": 60},
        priority=50,
        attempts=1,
        max_attempts=5,
        locked_by="test-worker",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        version=1,
    )

    # Mock connection execute for mark_succeeded
    conn = MagicMock()
    conn.execute.return_value.rowcount = 1
    engine.begin.return_value.__enter__.return_value = conn

    dispatch_job(engine, job, "test-worker")
    assert "BACKFILL_REPOSITORY" in HANDLERS


def test_dispatcher_handles_sync_github_repositories():
    engine = MagicMock()
    job = ClaimedJob(
        id=uuid4(),
        job_type="SYNC_GITHUB_REPOSITORIES",
        payload={"integrationId": str(uuid4())},
        priority=10,
        attempts=1,
        max_attempts=3,
        locked_by="test-worker",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        version=1,
    )

    conn = MagicMock()
    conn.execute.return_value.rowcount = 1
    engine.begin.return_value.__enter__.return_value = conn

    dispatch_job(engine, job, "test-worker")
    assert "SYNC_GITHUB_REPOSITORIES" in HANDLERS


def test_dispatcher_handles_jira_jobs():
    engine = MagicMock()
    job = ClaimedJob(
        id=uuid4(),
        job_type="SYNC_JIRA_PROJECTS",
        payload={"integrationId": str(uuid4())},
        priority=20,
        attempts=1,
        max_attempts=3,
        locked_by="test-worker",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        version=1,
    )

    conn = MagicMock()
    conn.execute.return_value.rowcount = 1
    engine.begin.return_value.__enter__.return_value = conn

    dispatch_job(engine, job, "test-worker")
    assert "SYNC_JIRA_PROJECTS" in HANDLERS
