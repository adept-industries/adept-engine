"""Normalize persisted Jira webhook deliveries."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import Engine, text

from app.db.models import ClaimedJob
from app.jobs.handlers.provider_support import parse_uuid
from app.jobs.retry import PermanentJobError, sanitize_error
from app.normalization.jira_issues import upsert_jira_issue

logger = structlog.get_logger()


def handle_process_jira_event(database_engine: Engine, job: ClaimedJob, worker_id: str) -> None:
    """Process one authenticated Jira delivery with a durable raw-event lifecycle."""
    del worker_id
    raw_event_id = parse_uuid(job.payload.get("rawEventId"), "rawEventId")
    integration_id = parse_uuid(job.payload.get("jiraIntegrationId"), "jiraIntegrationId")
    raw_event = _load_raw_event(database_engine, raw_event_id)
    if raw_event is None:
        raise PermanentJobError(f"Raw event {raw_event_id} not found")
    workspace_id = parse_uuid(raw_event.get("workspace_id"), "workspaceId")
    event_type = str(raw_event.get("event_type") or "unknown")
    payload = _json_object(raw_event.get("payload"))

    _assert_integration_workspace(database_engine, integration_id, workspace_id)
    bound_logger = logger.bind(
        job_id=str(job.id),
        event_type=event_type,
        raw_event_id=str(raw_event_id),
        workspace_id=str(workspace_id),
        jira_integration_id=str(integration_id),
    )
    _mark_processing(database_engine, raw_event_id)
    try:
        _dispatch_issue_event(
            database_engine,
            workspace_id,
            integration_id,
            event_type,
            payload,
            bound_logger,
        )
    except Exception as exc:
        _mark_failed(database_engine, raw_event_id, str(exc))
        raise
    _mark_processed(database_engine, raw_event_id)


def _dispatch_issue_event(
    database_engine: Engine,
    workspace_id: UUID,
    integration_id: UUID,
    event_type: str,
    payload: dict[str, Any],
    bound_logger: Any,
) -> None:
    if not event_type.startswith("jira:issue_"):
        bound_logger.debug("unhandled_jira_event_type")
        return
    issue = payload.get("issue")
    if not isinstance(issue, dict):
        bound_logger.warning("jira_event_missing_issue")
        return
    fields_value = issue.get("fields")
    fields: dict[str, Any] = fields_value if isinstance(fields_value, dict) else {}
    project_value = fields.get("project")
    project = project_value if isinstance(project_value, dict) else {}
    remote_project_id = project.get("id")
    if remote_project_id is None:
        bound_logger.warning("jira_issue_missing_project", issue_id=issue.get("id"))
        return

    jira_project_id = _resolve_project(
        database_engine,
        workspace_id,
        integration_id,
        str(remote_project_id),
    )
    if jira_project_id is None:
        bound_logger.info("jira_project_not_catalogued", remote_project_id=remote_project_id)
        return

    if event_type == "jira:issue_deleted":
        remote_issue_id = issue.get("id")
        if remote_issue_id is not None:
            with database_engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        DELETE FROM jira_issues
                        WHERE workspace_id = :workspace_id
                          AND jira_project_id = :jira_project_id
                          AND jira_issue_id = :jira_issue_id
                        """
                    ),
                    {
                        "workspace_id": workspace_id,
                        "jira_project_id": jira_project_id,
                        "jira_issue_id": str(remote_issue_id),
                    },
                )
        return

    upsert_jira_issue(database_engine, workspace_id, jira_project_id, issue)


def _load_raw_event(database_engine: Engine, raw_event_id: UUID) -> dict[str, Any] | None:
    with database_engine.connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                    SELECT workspace_id, event_type, payload
                    FROM raw_webhook_events
                    WHERE id = :id AND source = 'JIRA'
                    """
                ),
                {"id": raw_event_id},
            )
            .mappings()
            .one_or_none()
        )
    return dict(row) if row is not None else None


def _assert_integration_workspace(
    database_engine: Engine, integration_id: UUID, workspace_id: UUID
) -> None:
    with database_engine.connect() as connection:
        exists = connection.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1 FROM jira_integrations
                    WHERE id = :integration_id AND workspace_id = :workspace_id
                )
                """
            ),
            {"integration_id": integration_id, "workspace_id": workspace_id},
        ).scalar_one()
    if not exists:
        raise PermanentJobError("Jira integration is not in the event workspace")


def _resolve_project(
    database_engine: Engine,
    workspace_id: UUID,
    integration_id: UUID,
    remote_project_id: str,
) -> UUID | None:
    with database_engine.connect() as connection:
        value = connection.execute(
            text(
                """
                SELECT id FROM jira_projects
                WHERE workspace_id = :workspace_id
                  AND jira_integration_id = :integration_id
                  AND jira_project_id = :remote_project_id
                  AND tracking_enabled = true
                """
            ),
            {
                "workspace_id": workspace_id,
                "integration_id": integration_id,
                "remote_project_id": remote_project_id,
            },
        ).scalar_one_or_none()
    return UUID(str(value)) if value is not None else None


def _mark_processing(database_engine: Engine, raw_event_id: UUID) -> None:
    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE raw_webhook_events
                SET status = 'PROCESSING', attempt_count = attempt_count + 1,
                    processing_started_at = now(), processed_at = NULL,
                    last_error = NULL, updated_at = now(), version = version + 1
                WHERE id = :id
                """
            ),
            {"id": raw_event_id},
        )


def _mark_processed(database_engine: Engine, raw_event_id: UUID) -> None:
    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE raw_webhook_events
                SET status = 'PROCESSED', processed_at = now(), last_error = NULL,
                    updated_at = now(), version = version + 1
                WHERE id = :id
                """
            ),
            {"id": raw_event_id},
        )


def _mark_failed(database_engine: Engine, raw_event_id: UUID, error: str) -> None:
    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE raw_webhook_events
                SET status = 'FAILED', processed_at = NULL, last_error = :error,
                    updated_at = now(), version = version + 1
                WHERE id = :id
                """
            ),
            {"id": raw_event_id, "error": sanitize_error(error)},
        )


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}
