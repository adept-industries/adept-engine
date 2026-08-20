from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import Engine

from app.db.models import ClaimedJob
from app.jobs.claimer import claim_jobs
from app.jobs.dispatcher import HANDLERS, dispatch_job
from app.jobs.retry import JobOwnershipError, PermanentJobError, RequeueWithPayloadError
from app.providers import ProviderError
from tests.conftest import JobFactory


def _job(job_type: str) -> ClaimedJob:
    return ClaimedJob(
        id=uuid4(),
        job_type=job_type,
        payload={},
        priority=50,
        attempts=1,
        max_attempts=5,
        locked_by="test-worker",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        version=1,
    )


def test_expected_phase_five_handlers_are_registered() -> None:
    assert {
        "PROCESS_GITHUB_EVENT",
        "PROCESS_JIRA_EVENT",
        "BACKFILL_REPOSITORY",
        "SYNC_GITHUB_REPOSITORIES",
        "SYNC_JIRA_PROJECTS",
        "RENEW_JIRA_WEBHOOK",
        "DELETE_WORKSPACE",
    }.issubset(HANDLERS)


def test_dispatcher_is_the_only_success_finalizer(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = MagicMock()
    job = _job("TEST_SUCCESS")
    handler = MagicMock()
    monkeypatch.setitem(HANDLERS, job.job_type, handler)

    with (
        patch("app.jobs.dispatcher.mark_succeeded") as mark_succeeded,
        patch("app.jobs.dispatcher.mark_failed") as mark_failed,
    ):
        dispatch_job(engine, job, job.locked_by)

    handler.assert_called_once_with(engine, job, job.locked_by)
    mark_succeeded.assert_called_once_with(engine, job.id, job.locked_by)
    mark_failed.assert_not_called()


def test_dispatcher_marks_transient_handler_failure_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = MagicMock()
    job = _job("TEST_TRANSIENT_FAILURE")
    handler = MagicMock(side_effect=RuntimeError("temporary provider failure"))
    monkeypatch.setitem(HANDLERS, job.job_type, handler)

    with (
        patch("app.jobs.dispatcher.mark_succeeded") as mark_succeeded,
        patch("app.jobs.dispatcher.mark_failed") as mark_failed,
    ):
        dispatch_job(engine, job, job.locked_by)

    mark_succeeded.assert_not_called()
    mark_failed.assert_called_once_with(
        engine,
        job.id,
        job.locked_by,
        "temporary provider failure",
    )


def test_dispatcher_marks_permanent_handler_failure_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = MagicMock()
    job = _job("TEST_PERMANENT_FAILURE")
    handler = MagicMock(side_effect=PermanentJobError("invalid payload"))
    monkeypatch.setitem(HANDLERS, job.job_type, handler)

    with (
        patch("app.jobs.dispatcher.mark_succeeded") as mark_succeeded,
        patch("app.jobs.dispatcher.mark_failed") as mark_failed,
    ):
        dispatch_job(engine, job, job.locked_by)

    mark_succeeded.assert_not_called()
    mark_failed.assert_called_once_with(
        engine,
        job.id,
        job.locked_by,
        "invalid payload",
        permanent=True,
    )


def test_dispatcher_respects_provider_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = MagicMock()
    job = _job("TEST_RATE_LIMIT")
    handler = MagicMock(
        side_effect=ProviderError("GitHub returned HTTP 429", retry_after_seconds=75)
    )
    monkeypatch.setitem(HANDLERS, job.job_type, handler)

    with (
        patch("app.jobs.dispatcher.mark_succeeded") as mark_succeeded,
        patch("app.jobs.dispatcher.mark_failed") as mark_failed,
    ):
        dispatch_job(engine, job, job.locked_by)

    mark_succeeded.assert_not_called()
    mark_failed.assert_called_once_with(
        engine,
        job.id,
        job.locked_by,
        "GitHub returned HTTP 429",
        retry_after_seconds=75,
    )


def test_dispatcher_does_not_finalize_explicitly_requeued_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = MagicMock()
    job = _job("TEST_REQUEUE")
    handler = MagicMock(side_effect=RequeueWithPayloadError())
    monkeypatch.setitem(HANDLERS, job.job_type, handler)

    with (
        patch("app.jobs.dispatcher.mark_succeeded") as mark_succeeded,
        patch("app.jobs.dispatcher.mark_failed") as mark_failed,
    ):
        dispatch_job(engine, job, job.locked_by)

    mark_succeeded.assert_not_called()
    mark_failed.assert_not_called()


def test_success_transition_ownership_error_is_not_double_handled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = MagicMock()
    job = _job("TEST_LOST_OWNERSHIP")
    monkeypatch.setitem(HANDLERS, job.job_type, MagicMock())

    with (
        patch(
            "app.jobs.dispatcher.mark_succeeded",
            side_effect=JobOwnershipError("job is not owned by this worker"),
        ),
        patch("app.jobs.dispatcher.mark_failed") as mark_failed,
        pytest.raises(JobOwnershipError),
    ):
        dispatch_job(engine, job, job.locked_by)

    mark_failed.assert_not_called()


def test_unsupported_job_is_marked_dead() -> None:
    engine = MagicMock()
    job = _job("UNKNOWN_JOB")

    with patch("app.jobs.dispatcher.mark_failed") as mark_failed:
        dispatch_job(engine, job, job.locked_by)

    mark_failed.assert_called_once_with(
        engine,
        job.id,
        job.locked_by,
        "UNSUPPORTED_JOB_TYPE: UNKNOWN_JOB",
        permanent=True,
    )


@pytest.mark.integration
def test_real_handler_is_finalized_once(
    database_engine: Engine,
    job_factory: JobFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = job_factory.insert(
        job_type="DELETE_WORKSPACE",
        payload={"workspaceId": str(uuid4())},
    )
    job = claim_jobs(database_engine, "test-worker", 1)[0]
    handler = MagicMock()
    monkeypatch.setitem(HANDLERS, "DELETE_WORKSPACE", handler)

    dispatch_job(database_engine, job, "test-worker")

    row = job_factory.row(job_id)
    assert row["status"] == "SUCCEEDED"
    assert row["locked_by"] is None
    assert row["finished_at"] is not None
    handler.assert_called_once_with(database_engine, job, "test-worker")


@pytest.mark.integration
def test_requeue_with_payload_does_not_consume_attempt_budget(
    database_engine: Engine,
    job_factory: JobFactory,
) -> None:
    job_id = job_factory.insert(
        job_type="BACKFILL_REPOSITORY",
        payload={"repositoryId": str(uuid4())},
    )
    job = claim_jobs(database_engine, "test-worker", 1)[0]

    from app.jobs.retry import requeue_with_payload

    with pytest.raises(RequeueWithPayloadError):
        requeue_with_payload(
            database_engine,
            job.id,
            "test-worker",
            {"repositoryId": str(uuid4()), "cursor": {"stage": "pull_requests", "page": 2}},
        )

    row = job_factory.row(job_id)
    assert row["status"] == "PENDING"
    assert row["locked_by"] is None
    assert row["attempts"] == 0
    assert row["payload"]["cursor"] == {"stage": "pull_requests", "page": 2}
