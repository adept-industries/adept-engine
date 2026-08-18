"""
Handler for PROCESS_GITHUB_EVENT jobs.

Loads the persisted raw webhook event, routes on event_type and action,
calls the appropriate normalizer, and logs structured fields (job_id,
delivery_id, workspace_id, repository_id) on every path so operators
can trace any delivery end-to-end.

Never-retry event types are marked DEAD immediately so they do not fill
the retry queue with known-permanent failures.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import Engine, text

from app.db.models import ClaimedJob
from app.jobs.retry import mark_failed
from app.normalization import deployments as deployment_normalizer
from app.normalization import pull_requests as pr_normalizer

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Event types the engine handles in Phase 5
# ---------------------------------------------------------------------------
# Events not in this set are acknowledged but produce no normalized data yet.
HANDLED_EVENTS = frozenset(
    {
        "pull_request",
        "workflow_run",
        "deployment_status",
        "push",
        "installation",
        "installation_repositories",
        "repository",
    }
)

# Actions that should never be retried because they indicate a permanent
# condition (unsupported action, unknown event, etc.).
PERMANENT_FAILURE_REASONS = frozenset(
    {
        "RAW_EVENT_NOT_FOUND",
        "WORKSPACE_OR_REPOSITORY_NOT_FOUND",
    }
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def handle_process_github_event(database_engine: Engine, job: ClaimedJob, worker_id: str) -> None:
    """
    Dispatch a PROCESS_GITHUB_EVENT job to the correct normalizer.

    Steps:
    1. Load the raw event row from the database.
    2. Resolve workspace_id and repository_id.
    3. Route on event_type + action.
    4. Call the normalizer (idempotent upsert).
    5. Mark the raw event PROCESSED.
    """
    bound_logger = logger.bind(
        job_id=str(job.id),
        job_type=job.job_type,
    )

    raw_event_id = job.payload.get("rawEventId") or job.payload.get("raw_event_id")
    delivery_id = job.payload.get("deliveryId") or job.payload.get("delivery_id", "unknown")
    event_type = job.payload.get("eventType") or job.payload.get("event_type", "unknown")

    bound_logger = bound_logger.bind(delivery_id=delivery_id, event_type=event_type)

    if not raw_event_id:
        bound_logger.error("process_github_event_missing_raw_event_id")
        mark_failed(
            database_engine,
            job.id,
            job.locked_by,
            "rawEventId missing from job payload",
            permanent=True,
        )
        return

    # 1. Load the raw event to get payload, workspace_id, and repository_id.
    raw_event = _load_raw_event(database_engine, UUID(str(raw_event_id)))
    if raw_event is None:
        bound_logger.error("process_github_event_raw_event_not_found", raw_event_id=raw_event_id)
        mark_failed(
            database_engine,
            job.id,
            job.locked_by,
            f"RAW_EVENT_NOT_FOUND: {raw_event_id}",
            permanent=True,
        )
        return

    workspace_id_raw = raw_event.get("workspace_id")
    repository_id_raw = raw_event.get("repository_id")
    actual_event_type = raw_event.get("event_type", event_type)
    action = raw_event.get("action")
    payload = raw_event.get("payload") or {}

    bound_logger = bound_logger.bind(
        workspace_id=str(workspace_id_raw),
        repository_id=str(repository_id_raw),
        action=action,
    )

    if not workspace_id_raw or not repository_id_raw:
        bound_logger.warning("process_github_event_no_workspace_or_repository")
        # The event was stored as IGNORED at ingestion time; mark the job complete.
        _mark_raw_event_processed(database_engine, UUID(str(raw_event_id)))
        return

    workspace_id = UUID(str(workspace_id_raw))
    repository_id = UUID(str(repository_id_raw))

    # Parse payload if it came back as a string (JSONB returned as text in some drivers).
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = {}

    # 2. Route to the correct normalizer.
    _dispatch(
        database_engine=database_engine,
        job=job,
        event_type=actual_event_type,
        action=action,
        payload=payload,
        workspace_id=workspace_id,
        repository_id=repository_id,
        bound_logger=bound_logger,
    )

    # 3. Mark the raw event PROCESSED regardless of normalizer outcome
    #    (the job status is what the worker uses for retry decisions).
    _mark_raw_event_processed(database_engine, UUID(str(raw_event_id)))


# ---------------------------------------------------------------------------
# Internal routing
# ---------------------------------------------------------------------------


def _dispatch(
    database_engine: Engine,
    job: ClaimedJob,
    event_type: str,
    action: str | None,
    payload: dict[str, Any],
    workspace_id: UUID,
    repository_id: UUID,
    bound_logger: Any,
) -> None:
    if event_type == "pull_request":
        _handle_pull_request(
            database_engine, payload, action, workspace_id, repository_id, bound_logger
        )
    elif event_type == "workflow_run":
        _handle_workflow_run(database_engine, payload, workspace_id, repository_id, bound_logger)
    elif event_type == "deployment_status":
        _handle_deployment_status(
            database_engine, payload, workspace_id, repository_id, bound_logger
        )
    elif event_type == "push":
        bound_logger.info("process_github_push_acknowledged_no_normalization_yet")
    elif event_type == "installation":
        _handle_installation(database_engine, payload, action, bound_logger)
    elif event_type == "installation_repositories":
        _handle_installation_repositories(database_engine, payload, action, bound_logger)
    elif event_type == "repository":
        _handle_repository(database_engine, payload, action, bound_logger)
    else:
        # Unknown event type — log and move on; not a retry-worthy failure.
        bound_logger.info(
            "process_github_event_unhandled_event_type",
            supported=sorted(HANDLED_EVENTS),
        )


def _handle_pull_request(
    database_engine: Engine,
    payload: dict[str, Any],
    action: str | None,
    workspace_id: UUID,
    repository_id: UUID,
    bound_logger: Any,
) -> None:
    # Supported actions that update the PR state.
    supported_actions = {
        "opened",
        "reopened",
        "synchronize",
        "closed",  # may be a merge
        "edited",
        "converted_to_draft",
        "ready_for_review",
    }
    if action not in supported_actions:
        bound_logger.info(
            "pull_request_action_skipped", action=action, supported=sorted(supported_actions)
        )
        return

    pr_data = payload.get("pull_request", {})
    if not pr_data:
        bound_logger.warning("pull_request_payload_missing_pull_request_key")
        return

    pr_id = pr_normalizer.upsert_pull_request(
        database_engine=database_engine,
        workspace_id=workspace_id,
        repository_id=repository_id,
        pr_data=pr_data,
        action=action,
    )
    bound_logger.info("pull_request_upserted", pr_db_id=str(pr_id), action=action)


def _handle_workflow_run(
    database_engine: Engine,
    payload: dict[str, Any],
    workspace_id: UUID,
    repository_id: UUID,
    bound_logger: Any,
) -> None:
    action = payload.get("action")
    if action != "completed":
        bound_logger.info("workflow_run_action_skipped", action=action)
        return

    deployment_id = deployment_normalizer.upsert_deployment_from_workflow_run(
        database_engine=database_engine,
        workspace_id=workspace_id,
        repository_id=repository_id,
        event_data=payload,
    )
    if deployment_id:
        bound_logger.info("workflow_run_deployment_upserted", deployment_db_id=str(deployment_id))


def _handle_deployment_status(
    database_engine: Engine,
    payload: dict[str, Any],
    workspace_id: UUID,
    repository_id: UUID,
    bound_logger: Any,
) -> None:
    deployment_id = deployment_normalizer.upsert_deployment_from_deployment_status(
        database_engine=database_engine,
        workspace_id=workspace_id,
        repository_id=repository_id,
        event_data=payload,
    )
    if deployment_id:
        bound_logger.info(
            "deployment_status_deployment_upserted", deployment_db_id=str(deployment_id)
        )


def _handle_installation(
    database_engine: Engine,
    payload: dict[str, Any],
    action: str | None,
    bound_logger: Any,
) -> None:
    # Handle installation suspended, unsuspended, deleted
    installation = payload.get("installation", {})
    installation_id = installation.get("id")
    if not installation_id:
        return

    if action == "deleted":
        status = "REVOKED"
    elif action == "suspend":
        status = "ERROR"  # Or a suspended state if we had one
    elif action == "unsuspend" or action == "created":
        status = "ACTIVE"
    else:
        bound_logger.info("installation_action_skipped", action=action)
        return

    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE github_integrations
                SET status = :status,
                    updated_at = now(),
                    version = version + 1
                WHERE installation_id = :installation_id
                """
            ),
            {"status": status, "installation_id": installation_id},
        )
    bound_logger.info("installation_status_updated", action=action, status=status)


def _handle_installation_repositories(
    database_engine: Engine,
    payload: dict[str, Any],
    action: str | None,
    bound_logger: Any,
) -> None:
    # Action is usually 'added' or 'removed'
    installation = payload.get("installation", {})
    installation_id = installation.get("id")
    if not installation_id:
        return

    repositories_removed = payload.get("repositories_removed", [])
    if repositories_removed:
        repo_ids = [repo["id"] for repo in repositories_removed]
        with database_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE repositories
                    SET tracking_enabled = false,
                        updated_at = now(),
                        version = version + 1
                    WHERE id IN (
                        SELECT r.id FROM repositories r
                        JOIN github_integrations gi ON r.github_integration_id = gi.id
                        WHERE gi.installation_id = :installation_id
                          AND r.github_repo_id = ANY(:repo_ids)
                    )
                    """
                ),
                {"installation_id": installation_id, "repo_ids": repo_ids},
            )
        bound_logger.info("installation_repositories_removed", count=len(repo_ids))


def _handle_repository(
    database_engine: Engine,
    payload: dict[str, Any],
    action: str | None,
    bound_logger: Any,
) -> None:
    # Handle repository renamed, archived, unarchived, deleted
    repository = payload.get("repository", {})
    repo_id = repository.get("id")
    if not repo_id:
        return

    if action in {"archived", "deleted"}:
        with database_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE repositories
                    SET archived = true,
                        tracking_enabled = false,
                        updated_at = now(),
                        version = version + 1
                    WHERE github_repo_id = :repo_id
                    """
                ),
                {"repo_id": repo_id},
            )
        bound_logger.info("repository_archived_or_deleted", action=action)
    elif action == "unarchived":
        with database_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE repositories
                    SET archived = false,
                        updated_at = now(),
                        version = version + 1
                    WHERE github_repo_id = :repo_id
                    """
                ),
                {"repo_id": repo_id},
            )
        bound_logger.info("repository_unarchived", action=action)
    elif action == "renamed":
        new_name = repository.get("full_name")
        if new_name:
            with database_engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        UPDATE repositories
                        SET owner_with_name = :new_name,
                            updated_at = now(),
                            version = version + 1
                        WHERE github_repo_id = :repo_id
                        """
                    ),
                    {"new_name": new_name, "repo_id": repo_id},
                )
            bound_logger.info("repository_renamed", action=action, new_name=new_name)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _load_raw_event(database_engine: Engine, raw_event_id: UUID) -> dict[str, Any] | None:
    with database_engine.connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                SELECT id, workspace_id, repository_id, event_type, action, payload
                FROM raw_webhook_events
                WHERE id = :raw_event_id
                """
                ),
                {"raw_event_id": str(raw_event_id)},
            )
            .mappings()
            .one_or_none()
        )

    if row is None:
        return None
    return dict(row)


def _mark_raw_event_processed(database_engine: Engine, raw_event_id: UUID) -> None:
    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE raw_webhook_events
                SET status = 'PROCESSED',
                    processed_at = now(),
                    updated_at = now(),
                    version = version + 1
                WHERE id = :raw_event_id
                """
            ),
            {"raw_event_id": str(raw_event_id)},
        )
