"""Keep an executing job owned and serialize conflicting repository work.

The guard thread alone owns its database connection. Session advisory locks
survive heartbeat transactions, but disappear if the process/connection dies.
Handlers continue to use their own short-lived connections and transactions.
"""

from collections.abc import Callable
from threading import Event, Thread
from types import TracebackType
from uuid import UUID

import structlog
from sqlalchemy import Connection, Engine, text

from app.db.models import ClaimedJob
from app.jobs.retry import JobOwnershipError

logger = structlog.get_logger()


class JobScopeBusy(Exception):
    """Another consumer owns this job, repository or workspace operation."""


def _uuid(value: object) -> UUID | None:
    try:
        return UUID(str(value)) if value is not None else None
    except TypeError, ValueError:
        # The handler still validates and rejects malformed payloads.
        return None


def _lock(connection: Connection, key: str, *, shared: bool = False) -> None:
    function = "pg_try_advisory_lock_shared" if shared else "pg_try_advisory_lock"
    if not connection.execute(
        text(f"SELECT {function}(hashtextextended(:key, 0))"), {"key": key}
    ).scalar_one():
        raise JobScopeBusy(key)


def _acquire(connection: Connection, job: ClaimedJob, worker_id: str) -> None:
    _lock(connection, f"adept:job:{job.id}")
    row = (
        connection.execute(
            text(
                """
            SELECT COALESCE(j.workspace_id, e.workspace_id) AS workspace_id,
                   COALESCE(j.repository_id, e.repository_id) AS repository_id
            FROM processing_jobs j
            LEFT JOIN raw_webhook_events e
              ON e.id = COALESCE(j.raw_event_id, CAST(:raw_event_id AS uuid))
            WHERE j.id = :id AND j.status = 'RUNNING'
              AND j.locked_by = :owner AND j.attempts = :attempts
            """
            ),
            {
                "id": job.id,
                "owner": worker_id,
                "attempts": job.attempts,
                "raw_event_id": _uuid(
                    job.payload.get("rawEventId") or job.payload.get("raw_event_id")
                ),
            },
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        raise JobOwnershipError("job ownership changed before execution")

    repository_id = _uuid(row["repository_id"]) or _uuid(
        job.payload.get("repositoryId") or job.payload.get("repository_id")
    )
    if job.job_type in {
        "DELETE_WORKSPACE",
        "SYNC_GITHUB_REPOSITORIES",
        "SYNC_JIRA_PROJECTS",
        "RENEW_JIRA_WEBHOOK",
    }:
        repository_id = None
    workspace_id = _uuid(row["workspace_id"]) or _uuid(
        job.payload.get("workspaceId") or job.payload.get("workspace_id")
    )
    if workspace_id is None and job.job_type == "RENEW_JIRA_WEBHOOK":
        workspace_id = _uuid(
            connection.execute(
                text("SELECT workspace_id FROM jira_integrations WHERE id = :id"),
                {"id": _uuid(job.payload.get("jiraIntegrationId"))},
            ).scalar_one_or_none()
        )
    if repository_id is not None:
        # Resolve the authoritative workspace even for older payload-only jobs.
        actual_workspace = connection.execute(
            text("SELECT workspace_id FROM repositories WHERE id = :id"),
            {"id": repository_id},
        ).scalar_one_or_none()
        workspace_id = _uuid(actual_workspace) or workspace_id

    # Repository jobs may share a workspace; catalog/Jira/workspace-deletion
    # work without a repository gets exclusive access to that workspace.
    # Always acquire workspace before repository to avoid lock-order inversions.
    if workspace_id is not None:
        _lock(connection, f"adept:workspace:{workspace_id}", shared=repository_id is not None)
    if repository_id is not None:
        _lock(connection, f"adept:repository:{repository_id}")


def _heartbeat(connection: Connection, job: ClaimedJob, worker_id: str) -> None:
    connection.execute(
        text(
            """
            WITH owned AS (
                SELECT id FROM processing_jobs
                WHERE id = :id AND status = 'RUNNING'
                  AND locked_by = :owner AND attempts = :attempts
                FOR UPDATE SKIP LOCKED
            )
            UPDATE processing_jobs j SET locked_at = now()
            FROM owned WHERE j.id = owned.id
            """
        ),
        {"id": job.id, "owner": worker_id, "attempts": job.attempts},
    )
    # A handler can requeue/finalize/delete a job, or temporarily lock its row.
    # Zero updated rows is therefore normal. The advisory job lock remains held
    # until dispatch returns, including during explicit paginated requeues.
    connection.commit()


class JobLease:
    def __init__(
        self,
        database_engine: Engine,
        job: ClaimedJob,
        worker_id: str,
        *,
        interval_seconds: float,
        on_connection_lost: Callable[[], None],
    ) -> None:
        self.engine = database_engine
        self.job = job
        self.worker_id = worker_id
        self.interval = interval_seconds
        self.on_connection_lost = on_connection_lost
        self._ready = Event()
        self._stop = Event()
        self._error: Exception | None = None
        self._thread = Thread(target=self._maintain, name="job-lease", daemon=True)

    def __enter__(self) -> JobLease:
        self._thread.start()
        self._ready.wait()
        if self._error is not None:
            self._thread.join()
            raise self._error
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._thread.join()

    def _maintain(self) -> None:
        acquired = False
        try:
            with self.engine.connect() as connection:
                # Session locks MUST NOT be returned to the shared pool. Closing
                # this detached connection physically closes the DB session.
                connection.detach()
                connection.execute(text("SET statement_timeout = '5s'"))
                connection.execute(text("SET lock_timeout = '1s'"))
                _acquire(connection, self.job, self.worker_id)
                _heartbeat(connection, self.job, self.worker_id)
                acquired = True
                self._ready.set()
                while not self._stop.wait(self.interval):
                    _heartbeat(connection, self.job, self.worker_id)
        except Exception as exc:
            self._error = exc
            if acquired and not self._stop.is_set():
                logger.error(
                    "job_lease_connection_lost",
                    worker_id=self.worker_id,
                    job_id=str(self.job.id),
                    error_type=type(exc).__name__,
                )
                # Python cannot safely cancel a running handler thread. The
                # runtime must fail closed instead of allowing unowned writes.
                self.on_connection_lost()
        finally:
            self._ready.set()
