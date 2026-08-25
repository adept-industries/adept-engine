"""Normalize persisted GitHub webhook deliveries.

Repository lifecycle deliveries are intentionally handled without a local
``repository_id``. GitHub sends installation and repository-catalog changes
before (or after) a local repository row exists, so those handlers scope all
writes through the workspace and installation instead.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import Engine, text

from app.core.config import get_settings
from app.db.models import ClaimedJob
from app.jobs.handlers.provider_support import load_github_repository, parse_uuid
from app.jobs.handlers.sync_github_repositories import _upsert_repository
from app.jobs.retry import PermanentJobError, sanitize_error
from app.normalization import deployments as deployment_normalizer
from app.normalization import pull_requests as pr_normalizer
from app.providers.github import GithubClient

logger = structlog.get_logger()

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
REPOSITORY_OPTIONAL_EVENTS = frozenset({"installation", "installation_repositories", "repository"})


def handle_process_github_event(database_engine: Engine, job: ClaimedJob, worker_id: str) -> None:
    """Load, route, and acknowledge one durable GitHub delivery."""
    del worker_id
    raw_event_id = parse_uuid(
        job.payload.get("rawEventId") or job.payload.get("raw_event_id"),
        "rawEventId",
    )
    raw_event = _load_raw_event(database_engine, raw_event_id)
    if raw_event is None:
        raise PermanentJobError(f"RAW_EVENT_NOT_FOUND: {raw_event_id}")

    event_type = str(raw_event.get("event_type") or job.payload.get("eventType") or "unknown")
    action_raw = raw_event.get("action")
    action = str(action_raw) if action_raw is not None else None
    payload = _json_object(raw_event.get("payload"))
    workspace_raw = raw_event.get("workspace_id")
    repository_raw = raw_event.get("repository_id")
    bound_logger = logger.bind(
        job_id=str(job.id),
        delivery_id=job.payload.get("deliveryId", "unknown"),
        event_type=event_type,
        action=action,
        workspace_id=str(workspace_raw),
        repository_id=str(repository_raw),
    )

    if workspace_raw is None:
        bound_logger.warning("process_github_event_missing_workspace")
        _mark_raw_event_ignored(database_engine, raw_event_id, "workspace not resolved")
        return

    workspace_id = parse_uuid(workspace_raw, "workspaceId")
    repository_id = (
        parse_uuid(repository_raw, "repositoryId") if repository_raw is not None else None
    )
    if repository_id is None and event_type not in REPOSITORY_OPTIONAL_EVENTS:
        bound_logger.warning("process_github_event_missing_repository")
        _mark_raw_event_ignored(database_engine, raw_event_id, "repository not resolved")
        return
    deployment_signal: str | None = None
    if event_type not in REPOSITORY_OPTIONAL_EVENTS and repository_id is not None:
        deployment_signal = _tracked_repository_deployment_signal(
            database_engine, workspace_id, repository_id
        )
        if deployment_signal is None:
            bound_logger.info("process_github_event_repository_not_tracked")
            _mark_raw_event_ignored(database_engine, raw_event_id, "repository tracking disabled")
            return

    _mark_raw_event_processing(database_engine, raw_event_id)
    try:
        _dispatch(
            database_engine,
            event_type,
            action,
            payload,
            workspace_id,
            repository_id,
            deployment_signal,
            bound_logger,
        )
    except Exception as exc:
        _mark_raw_event_failed(database_engine, raw_event_id, str(exc))
        raise
    _mark_raw_event_processed(database_engine, raw_event_id)


def _dispatch(
    database_engine: Engine,
    event_type: str,
    action: str | None,
    payload: dict[str, Any],
    workspace_id: UUID,
    repository_id: UUID | None,
    deployment_signal: str | None,
    bound_logger: Any,
) -> None:
    if event_type == "pull_request":
        _handle_pull_request(
            database_engine,
            payload,
            action,
            workspace_id,
            _required_repository(repository_id),
            bound_logger,
        )
    elif event_type == "workflow_run":
        if deployment_signal != "WORKFLOW_RUN":
            bound_logger.info(
                "workflow_run_skipped_for_deployment_signal",
                deployment_signal=deployment_signal,
            )
            return
        _handle_workflow_run(
            database_engine,
            payload,
            workspace_id,
            _required_repository(repository_id),
            bound_logger,
        )
    elif event_type == "deployment_status":
        if deployment_signal != "DEPLOYMENT":
            bound_logger.info(
                "deployment_status_skipped_for_deployment_signal",
                deployment_signal=deployment_signal,
            )
            return
        _handle_deployment_status(
            database_engine,
            payload,
            workspace_id,
            _required_repository(repository_id),
            bound_logger,
        )
    elif event_type == "push":
        bound_logger.info(
            "process_github_push_acknowledged_no_normalization",
            deployment_signal=deployment_signal,
        )
    elif event_type == "installation":
        _handle_installation(database_engine, payload, action, workspace_id, bound_logger)
    elif event_type == "installation_repositories":
        _handle_installation_repositories(database_engine, payload, workspace_id, bound_logger)
    elif event_type == "repository":
        _handle_repository(database_engine, payload, action, workspace_id, bound_logger)
    else:
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
    supported_actions = {
        "opened",
        "reopened",
        "synchronize",
        "closed",
        "edited",
        "converted_to_draft",
        "ready_for_review",
    }
    if action not in supported_actions:
        bound_logger.info(
            "pull_request_action_skipped", action=action, supported=sorted(supported_actions)
        )
        return
    pr_data = payload.get("pull_request")
    if not isinstance(pr_data, dict):
        bound_logger.warning("pull_request_payload_missing_pull_request_key")
        return
    number = pr_data.get("number")
    if not isinstance(number, int):
        raise PermanentJobError("GitHub returned a pull request without a number")
    repository = load_github_repository(database_engine, repository_id)
    with GithubClient(get_settings(), repository.installation_id) as client:
        commits = client.list_pull_request_commits(
            repository.owner_login,
            repository.name,
            number,
        )
    pr_id = pr_normalizer.upsert_pull_request(
        database_engine,
        workspace_id,
        repository_id,
        pr_data,
        action,
        commits,
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
        database_engine, workspace_id, repository_id, payload
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
        database_engine, workspace_id, repository_id, payload
    )
    if deployment_id:
        bound_logger.info(
            "deployment_status_deployment_upserted", deployment_db_id=str(deployment_id)
        )


def _handle_installation(
    database_engine: Engine,
    payload: dict[str, Any],
    action: str | None,
    workspace_id: UUID,
    bound_logger: Any,
) -> None:
    installation = payload.get("installation")
    installation_id = installation.get("id") if isinstance(installation, dict) else None
    if not isinstance(installation_id, int):
        raise PermanentJobError("GitHub installation event is missing installation.id")

    status_by_action = {
        "created": "ACTIVE",
        "new_permissions_accepted": "ACTIVE",
        "unsuspend": "ACTIVE",
        "suspend": "SUSPENDED",
        "deleted": "REVOKED",
    }
    status = status_by_action.get(action or "")
    if status is None:
        bound_logger.info("installation_action_skipped", action=action)
        return
    with database_engine.begin() as connection:
        result = connection.execute(
            text(
                """
                UPDATE github_integrations
                SET status = :status,
                    suspended_at = CASE WHEN :status = 'SUSPENDED' THEN now() ELSE NULL END,
                    updated_at = now(),
                    version = version + 1
                WHERE workspace_id = :workspace_id
                  AND installation_id = :installation_id
                  AND (status <> :status OR
                       (status = 'SUSPENDED' AND suspended_at IS NULL))
                """
            ),
            {
                "status": status,
                "workspace_id": workspace_id,
                "installation_id": installation_id,
            },
        )
        if action == "deleted":
            connection.execute(
                text(
                    """
                    UPDATE repositories r
                    SET tracking_enabled = false,
                        updated_at = now(),
                        version = r.version + 1
                    FROM github_integrations gi
                    WHERE r.github_integration_id = gi.id
                      AND gi.workspace_id = :workspace_id
                      AND gi.installation_id = :installation_id
                    """
                ),
                {
                    "workspace_id": workspace_id,
                    "installation_id": installation_id,
                },
            )
    bound_logger.info(
        "installation_status_updated", action=action, status=status, changed=result.rowcount
    )


def _handle_installation_repositories(
    database_engine: Engine,
    payload: dict[str, Any],
    workspace_id: UUID,
    bound_logger: Any,
) -> None:
    integration = _github_integration_for_event(database_engine, payload, workspace_id)
    if integration is None:
        bound_logger.warning("installation_repositories_integration_not_found")
        return
    integration_id, installation_id = integration
    added = _object_list(payload.get("repositories_added"))
    removed = _object_list(payload.get("repositories_removed"))
    sync_time = datetime.now(UTC)
    with database_engine.begin() as connection:
        for repository in added:
            _upsert_repository(connection, workspace_id, integration_id, repository, sync_time)
        for repository in removed:
            remote_id = repository.get("id")
            if isinstance(remote_id, int):
                connection.execute(
                    text(
                        """
                        UPDATE repositories
                        SET tracking_enabled = false,
                            updated_at = now(),
                            version = version + 1
                        WHERE workspace_id = :workspace_id
                          AND github_integration_id = :integration_id
                          AND github_repo_id = :github_repo_id
                        """
                    ),
                    {
                        "workspace_id": workspace_id,
                        "integration_id": integration_id,
                        "github_repo_id": remote_id,
                    },
                )
    bound_logger.info(
        "installation_repositories_updated",
        installation_id=installation_id,
        added=len(added),
        removed=len(removed),
    )


def _handle_repository(
    database_engine: Engine,
    payload: dict[str, Any],
    action: str | None,
    workspace_id: UUID,
    bound_logger: Any,
) -> None:
    repository = payload.get("repository")
    if not isinstance(repository, dict) or not isinstance(repository.get("id"), int):
        raise PermanentJobError("GitHub repository event is missing repository.id")
    remote_id = int(repository["id"])
    integration = _github_integration_for_event(database_engine, payload, workspace_id)

    if action == "created" and integration is not None:
        with database_engine.begin() as connection:
            _upsert_repository(
                connection, workspace_id, integration[0], repository, datetime.now(UTC)
            )
        bound_logger.info("repository_catalogued", github_repo_id=remote_id)
        return

    if action in {"archived", "deleted"}:
        with database_engine.begin() as connection:
            result = connection.execute(
                text(
                    """
                    UPDATE repositories
                    SET archived = true,
                        tracking_enabled = false,
                        updated_at = now(),
                        version = version + 1
                    WHERE workspace_id = :workspace_id
                      AND github_repo_id = :github_repo_id
                    """
                ),
                {"workspace_id": workspace_id, "github_repo_id": remote_id},
            )
        bound_logger.info("repository_archived_or_deleted", action=action, changed=result.rowcount)
        return

    if action not in {
        "unarchived",
        "renamed",
        "edited",
        "transferred",
        "publicized",
        "privatized",
    }:
        bound_logger.info("repository_action_skipped", action=action)
        return

    owner_value = repository.get("owner")
    owner: dict[str, Any] = owner_value if isinstance(owner_value, dict) else {}
    full_name = repository.get("full_name")
    owner_login = owner.get("login")
    if not isinstance(owner_login, str) and isinstance(full_name, str) and "/" in full_name:
        owner_login = full_name.split("/", 1)[0]
    visibility = _repository_visibility(repository)
    with database_engine.begin() as connection:
        result = connection.execute(
            text(
                """
                UPDATE repositories
                SET owner_login = COALESCE(:owner_login, owner_login),
                    name = COALESCE(:name, name),
                    full_name = COALESCE(:full_name, full_name),
                    default_branch = COALESCE(:default_branch, default_branch),
                    visibility = COALESCE(:visibility, visibility),
                    archived = COALESCE(:archived, archived),
                    last_synced_at = now(),
                    updated_at = now(),
                    version = version + 1
                WHERE workspace_id = :workspace_id
                  AND github_repo_id = :github_repo_id
                """
            ),
            {
                "workspace_id": workspace_id,
                "github_repo_id": remote_id,
                "owner_login": owner_login if isinstance(owner_login, str) else None,
                "name": repository.get("name") if isinstance(repository.get("name"), str) else None,
                "full_name": full_name if isinstance(full_name, str) else None,
                "default_branch": (
                    repository.get("default_branch")
                    if isinstance(repository.get("default_branch"), str)
                    else None
                ),
                "visibility": visibility,
                "archived": (bool(repository["archived"]) if "archived" in repository else None),
            },
        )
    bound_logger.info("repository_metadata_updated", action=action, changed=result.rowcount)


def _github_integration_for_event(
    database_engine: Engine, payload: dict[str, Any], workspace_id: UUID
) -> tuple[UUID, int] | None:
    installation = payload.get("installation")
    installation_id = installation.get("id") if isinstance(installation, dict) else None
    if not isinstance(installation_id, int):
        return None
    with database_engine.connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                    SELECT id, installation_id
                    FROM github_integrations
                    WHERE workspace_id = :workspace_id
                      AND installation_id = :installation_id
                    """
                ),
                {"workspace_id": workspace_id, "installation_id": installation_id},
            )
            .mappings()
            .one_or_none()
        )
    if row is None:
        return None
    return UUID(str(row["id"])), int(row["installation_id"])


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
                {"raw_event_id": raw_event_id},
            )
            .mappings()
            .one_or_none()
        )
    return dict(row) if row is not None else None


def _tracked_repository_deployment_signal(
    database_engine: Engine, workspace_id: UUID, repository_id: UUID
) -> str | None:
    with database_engine.connect() as connection:
        value = connection.execute(
            text(
                """
                SELECT COALESCE(NULLIF(settings->>'deploymentSignal', ''), 'WORKFLOW_RUN')
                FROM repositories
                WHERE id = :repository_id
                  AND workspace_id = :workspace_id
                  AND tracking_enabled = true
                  AND archived = false
                """
            ),
            {"repository_id": repository_id, "workspace_id": workspace_id},
        ).scalar_one_or_none()
    return str(value) if value is not None else None


def _mark_raw_event_processing(database_engine: Engine, raw_event_id: UUID) -> None:
    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE raw_webhook_events
                SET status = 'PROCESSING',
                    attempt_count = attempt_count + 1,
                    processing_started_at = now(),
                    processed_at = NULL,
                    last_error = NULL,
                    updated_at = now(),
                    version = version + 1
                WHERE id = :raw_event_id
                """
            ),
            {"raw_event_id": raw_event_id},
        )


def _mark_raw_event_processed(database_engine: Engine, raw_event_id: UUID) -> None:
    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE raw_webhook_events
                SET status = 'PROCESSED',
                    processed_at = now(),
                    last_error = NULL,
                    updated_at = now(),
                    version = version + 1
                WHERE id = :raw_event_id
                """
            ),
            {"raw_event_id": raw_event_id},
        )


def _mark_raw_event_ignored(database_engine: Engine, raw_event_id: UUID, reason: str) -> None:
    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE raw_webhook_events
                SET status = 'IGNORED',
                    processed_at = now(),
                    last_error = :reason,
                    updated_at = now(),
                    version = version + 1
                WHERE id = :raw_event_id
                """
            ),
            {"raw_event_id": raw_event_id, "reason": sanitize_error(reason)},
        )


def _mark_raw_event_failed(database_engine: Engine, raw_event_id: UUID, error: str) -> None:
    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE raw_webhook_events
                SET status = 'FAILED',
                    last_error = :error,
                    processed_at = NULL,
                    updated_at = now(),
                    version = version + 1
                WHERE id = :raw_event_id
                """
            ),
            {"raw_event_id": raw_event_id, "error": sanitize_error(error)},
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


def _object_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _repository_visibility(repository: dict[str, Any]) -> str:
    visibility = str(repository.get("visibility") or "").upper()
    if visibility in {"PUBLIC", "PRIVATE", "INTERNAL"}:
        return visibility
    return "PRIVATE" if repository.get("private") else "PUBLIC"


def _required_repository(repository_id: UUID | None) -> UUID:
    if repository_id is None:
        raise PermanentJobError("Repository is required for this GitHub event")
    return repository_id
