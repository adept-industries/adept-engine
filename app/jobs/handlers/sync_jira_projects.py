from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import Engine, text

from app.core.config import get_settings
from app.db.models import ClaimedJob
from app.jobs.handlers.provider_support import (
    get_valid_jira_access_token,
    load_jira_integration,
    mark_jira_integration_error,
    parse_uuid,
    provider_exception_as_job_error,
)
from app.jobs.retry import PermanentJobError, requeue_with_payload
from app.providers.jira import JiraClient

logger = structlog.get_logger()


def handle_sync_jira_projects(database_engine: Engine, job: ClaimedJob, worker_id: str) -> None:
    workspace_id = parse_uuid(job.payload.get("workspaceId"), "workspaceId")
    integration_id = parse_uuid(job.payload.get("jiraIntegrationId"), "jiraIntegrationId")
    start_at = _non_negative_int(job.payload.get("cursor", 0), "cursor")
    sync_started_at = _sync_started_at(job.payload.get("syncStartedAt"))
    integration = load_jira_integration(database_engine, integration_id, workspace_id)
    settings = get_settings()

    bound_logger = logger.bind(
        job_id=str(job.id),
        workspace_id=str(workspace_id),
        jira_integration_id=str(integration_id),
        start_at=start_at,
    )
    bound_logger.info("sync_jira_projects_started")

    try:
        access_token, integration = get_valid_jira_access_token(
            database_engine, integration, settings
        )
        with JiraClient(settings, integration.cloud_id, access_token) as client:
            result = client.list_projects(start_at)
    except Exception as exc:
        converted = provider_exception_as_job_error(exc)
        if isinstance(converted, PermanentJobError) or job.attempts >= job.max_attempts:
            mark_jira_integration_error(database_engine, integration_id)
        if converted is exc:
            raise
        raise converted from exc

    with database_engine.begin() as connection:
        for project in result.items:
            _upsert_jira_project(
                connection,
                workspace_id,
                integration_id,
                project,
                sync_started_at,
            )

        if result.next_page is None:
            connection.execute(
                text(
                    """
                    UPDATE jira_projects
                    SET tracking_enabled = false,
                        updated_at = now(),
                        version = version + 1
                    WHERE jira_integration_id = :integration_id
                      AND (last_synced_at IS NULL OR last_synced_at < :sync_started_at)
                    """
                ),
                {
                    "integration_id": integration_id,
                    "sync_started_at": sync_started_at,
                },
            )
            connection.execute(
                text(
                    """
                    UPDATE jira_integrations
                    SET last_synced_at = :sync_started_at,
                        updated_at = now(),
                        version = version + 1
                    WHERE id = :integration_id AND status = 'ACTIVE'
                    """
                ),
                {
                    "integration_id": integration_id,
                    "sync_started_at": sync_started_at,
                },
            )

    if result.next_page is not None:
        bound_logger.info(
            "sync_jira_projects_page_completed",
            project_count=len(result.items),
            next_start_at=result.next_page,
        )
        requeue_with_payload(
            database_engine,
            job.id,
            worker_id,
            {
                **job.payload,
                "cursor": result.next_page,
                "syncStartedAt": sync_started_at.isoformat(),
            },
        )

    bound_logger.info("sync_jira_projects_completed", project_count=len(result.items))


def _upsert_jira_project(
    connection: Any,
    workspace_id: object,
    integration_id: object,
    project: dict[str, Any],
    sync_started_at: datetime,
) -> None:
    remote_id = project.get("id")
    key = project.get("key")
    name = project.get("name")
    if not all(isinstance(value, str) and value for value in (remote_id, key, name)):
        raise PermanentJobError("Jira returned an invalid project record")

    connection.execute(
        text(
            """
            INSERT INTO jira_projects (
                workspace_id, jira_integration_id, jira_project_id,
                project_key, project_name, project_type,
                tracking_enabled, last_synced_at
            ) VALUES (
                :workspace_id, :integration_id, :jira_project_id,
                :project_key, :project_name, :project_type,
                false, :sync_started_at
            )
            ON CONFLICT (jira_integration_id, jira_project_id)
            DO UPDATE SET
                project_key = EXCLUDED.project_key,
                project_name = EXCLUDED.project_name,
                project_type = EXCLUDED.project_type,
                last_synced_at = EXCLUDED.last_synced_at,
                updated_at = now(),
                version = jira_projects.version + 1
            """
        ),
        {
            "workspace_id": workspace_id,
            "integration_id": integration_id,
            "jira_project_id": remote_id,
            "project_key": key,
            "project_name": name,
            "project_type": project.get("projectTypeKey") or "software",
            "sync_started_at": sync_started_at,
        },
    )


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise PermanentJobError(f"Invalid {field_name}")
    if not isinstance(value, (int, str)):
        raise PermanentJobError(f"Invalid {field_name}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PermanentJobError(f"Invalid {field_name}") from exc
    if parsed < 0:
        raise PermanentJobError(f"Invalid {field_name}")
    return parsed


def _sync_started_at(value: object) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if not isinstance(value, str):
        raise PermanentJobError("Invalid syncStartedAt")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PermanentJobError("Invalid syncStartedAt") from exc
    if parsed.tzinfo is None:
        raise PermanentJobError("Invalid syncStartedAt")
    return parsed.astimezone(UTC)
