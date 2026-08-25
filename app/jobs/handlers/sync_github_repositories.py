from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import Engine, text

from app.core.config import get_settings
from app.db.models import ClaimedJob
from app.jobs.handlers.provider_support import (
    load_github_integration,
    parse_uuid,
    provider_exception_as_job_error,
)
from app.jobs.retry import PermanentJobError, requeue_with_payload
from app.providers.github import GithubClient

logger = structlog.get_logger()

DEFAULT_REPOSITORY_SETTINGS = {
    "deploymentSignal": "WORKFLOW_RUN",
    "productionBranchPatterns": ["main", "master", "release/*"],
    "productionEnvironmentPatterns": ["production", "prod", "live"],
    "deploymentWorkflowNamePatterns": ["*deploy*", "*production*", "*release*"],
    "incidentSource": "GITHUB",
    "doraExclusions": [],
    "defaultMetricGranularity": "WEEK",
    "backfillDays": 90,
}


def handle_sync_github_repositories(
    database_engine: Engine, job: ClaimedJob, worker_id: str
) -> None:
    workspace_id = parse_uuid(job.payload.get("workspaceId"), "workspaceId")
    integration_raw = job.payload.get("githubIntegrationId")
    integration_id = parse_uuid(integration_raw, "githubIntegrationId") if integration_raw else None
    page = _positive_int(job.payload.get("cursor", 1), "cursor")
    sync_started_at = _sync_started_at(job.payload.get("syncStartedAt"))
    integration = load_github_integration(database_engine, workspace_id, integration_id)

    bound_logger = logger.bind(
        job_id=str(job.id),
        workspace_id=str(workspace_id),
        github_integration_id=str(integration.id),
        page=page,
    )
    bound_logger.info("sync_github_repositories_started")

    try:
        with GithubClient(get_settings(), integration.installation_id) as client:
            result = client.list_installation_repositories(page)
    except Exception as exc:
        converted = provider_exception_as_job_error(exc)
        if converted is exc:
            raise
        raise converted from exc

    with database_engine.begin() as connection:
        for repository in result.items:
            _upsert_repository(
                connection,
                workspace_id,
                integration.id,
                repository,
                sync_started_at,
            )

        if result.next_page is None:
            connection.execute(
                text(
                    """
                    UPDATE repositories
                    SET tracking_enabled = false,
                        updated_at = now(),
                        version = version + 1
                    WHERE github_integration_id = :integration_id
                      AND (last_synced_at IS NULL OR last_synced_at < :sync_started_at)
                    """
                ),
                {
                    "integration_id": integration.id,
                    "sync_started_at": sync_started_at,
                },
            )
            connection.execute(
                text(
                    """
                    UPDATE github_integrations
                    SET last_synced_at = :sync_started_at,
                        updated_at = now(),
                        version = version + 1
                    WHERE id = :integration_id AND status = 'ACTIVE'
                    """
                ),
                {
                    "integration_id": integration.id,
                    "sync_started_at": sync_started_at,
                },
            )

    if result.next_page is not None:
        bound_logger.info(
            "sync_github_repositories_page_completed",
            repository_count=len(result.items),
            next_page=result.next_page,
        )
        requeue_with_payload(
            database_engine,
            job.id,
            worker_id,
            {
                **job.payload,
                "githubIntegrationId": str(integration.id),
                "cursor": result.next_page,
                "syncStartedAt": sync_started_at.isoformat(),
            },
        )

    bound_logger.info("sync_github_repositories_completed", repository_count=len(result.items))


def _upsert_repository(
    connection: Any,
    workspace_id: object,
    integration_id: object,
    repository: dict[str, Any],
    sync_started_at: datetime,
) -> None:
    repository_id = repository.get("id")
    owner_value = repository.get("owner")
    owner: dict[str, Any] = owner_value if isinstance(owner_value, dict) else {}
    owner_login = owner.get("login")
    name = repository.get("name")
    full_name = repository.get("full_name")
    if not isinstance(owner_login, str) and isinstance(full_name, str) and "/" in full_name:
        owner_login = full_name.split("/", 1)[0]
    if not isinstance(repository_id, int) or not all(
        isinstance(value, str) and value for value in (owner_login, name, full_name)
    ):
        raise PermanentJobError("GitHub returned an invalid repository record")

    visibility = str(repository.get("visibility") or "").upper()
    if visibility not in {"PUBLIC", "PRIVATE", "INTERNAL"}:
        visibility = "PRIVATE" if repository.get("private") else "PUBLIC"

    connection.execute(
        text(
            """
            INSERT INTO repositories (
                workspace_id, github_integration_id, github_repo_id,
                github_node_id, owner_login, name, full_name, default_branch,
                visibility, archived, tracking_enabled, settings, last_synced_at
            ) VALUES (
                :workspace_id, :integration_id, :github_repo_id,
                :github_node_id, :owner_login, :name, :full_name, :default_branch,
                :visibility, :archived, false, CAST(:settings AS jsonb), :sync_started_at
            )
            ON CONFLICT (workspace_id, github_repo_id)
            DO UPDATE SET
                github_integration_id = EXCLUDED.github_integration_id,
                github_node_id = EXCLUDED.github_node_id,
                owner_login = EXCLUDED.owner_login,
                name = EXCLUDED.name,
                full_name = EXCLUDED.full_name,
                default_branch = EXCLUDED.default_branch,
                visibility = EXCLUDED.visibility,
                archived = EXCLUDED.archived,
                tracking_enabled = CASE
                    WHEN EXCLUDED.archived THEN false
                    ELSE repositories.tracking_enabled
                END,
                last_synced_at = EXCLUDED.last_synced_at,
                updated_at = now(),
                version = repositories.version + 1
            """
        ),
        {
            "workspace_id": workspace_id,
            "integration_id": integration_id,
            "github_repo_id": repository_id,
            "github_node_id": repository.get("node_id"),
            "owner_login": owner_login,
            "name": name,
            "full_name": full_name,
            "default_branch": repository.get("default_branch") or "main",
            "visibility": visibility,
            "archived": bool(repository.get("archived", False)),
            "settings": json.dumps(DEFAULT_REPOSITORY_SETTINGS),
            "sync_started_at": sync_started_at,
        },
    )


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise PermanentJobError(f"Invalid {field_name}")
    if not isinstance(value, (int, str)):
        raise PermanentJobError(f"Invalid {field_name}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PermanentJobError(f"Invalid {field_name}") from exc
    if parsed < 1:
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
