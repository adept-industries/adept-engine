"""
Deployment normalization.

Parses GitHub ``workflow_run`` (completed) and ``deployment_status``
(success / failure) events and upserts rows into ``deployments``.

Unique key: (repository_id, source, external_deployment_id)
"""

from __future__ import annotations

import fnmatch
import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import Engine, text

from app.metrics.service import (
    enqueue_recalculate_metrics_job,
    link_deployments_to_pull_requests,
    update_github_incident_lifecycle,
)

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _get_repository_settings(database_engine: Engine, repository_id: UUID) -> dict[str, Any]:
    with database_engine.connect() as connection:
        row = connection.execute(
            text("SELECT settings FROM repositories WHERE id = :repository_id"),
            {"repository_id": str(repository_id)},
        ).scalar_one_or_none()
        if isinstance(row, dict):
            return row
        if isinstance(row, str):
            try:
                parsed = json.loads(row)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def _patterns(value: object) -> list[str]:
    if isinstance(value, str):
        return [pattern.strip() for pattern in value.split(",") if pattern.strip()]
    if isinstance(value, list):
        return [str(pattern).strip() for pattern in value if str(pattern).strip()]
    return []


def _matches(value: str, configured_patterns: object, defaults: list[str]) -> bool:
    if not value:
        return False
    patterns = _patterns(configured_patterns) or defaults
    lowered = value.lower()
    return any(fnmatch.fnmatch(lowered, pattern.lower()) for pattern in patterns)


def _is_excluded(signal_name: str, settings: dict[str, Any]) -> bool:
    return (
        _matches(signal_name, settings.get("doraExclusions"), [])
        if _patterns(settings.get("doraExclusions"))
        else False
    )


def _is_production_workflow(
    workflow_name: str,
    head_branch: str,
    default_branch: str,
    settings: dict[str, Any],
) -> bool:
    return (
        _matches(
            head_branch,
            settings.get("productionBranchPatterns"),
            [default_branch],
        )
        and _matches(
            workflow_name,
            settings.get("deploymentWorkflowNamePatterns"),
            ["*deploy*", "*production*", "*release*"],
        )
        and not _is_excluded(workflow_name, settings)
    )


def reclassify_repository_deployments(
    database_engine: Engine,
    repository_id: UUID,
) -> int:
    """Reapply the repository's current signal and production patterns to stored rows."""
    with database_engine.connect() as connection:
        repository = (
            connection.execute(
                text(
                    """
                    SELECT default_branch, settings
                    FROM repositories
                    WHERE id = :repository_id
                    """
                ),
                {"repository_id": repository_id},
            )
            .mappings()
            .one()
        )
        rows = (
            connection.execute(
                text(
                    """
                    SELECT id, source, environment, raw_data, is_production
                    FROM deployments
                    WHERE repository_id = :repository_id
                    """
                ),
                {"repository_id": repository_id},
            )
            .mappings()
            .all()
        )

    settings = repository["settings"] if isinstance(repository["settings"], dict) else {}
    selected_signal = str(settings.get("deploymentSignal", "WORKFLOW_RUN")).upper()
    updates: list[dict[str, Any]] = []
    for row in rows:
        raw_data = row["raw_data"] if isinstance(row["raw_data"], dict) else {}
        is_production = False
        if selected_signal == "WORKFLOW_RUN" and row["source"] == "GITHUB_WORKFLOW":
            workflow = raw_data.get("workflow_run")
            workflow_data = workflow if isinstance(workflow, dict) else {}
            is_production = _is_production_workflow(
                str(workflow_data.get("name") or row["environment"] or ""),
                str(workflow_data.get("head_branch") or ""),
                str(repository["default_branch"]),
                settings,
            )
        elif selected_signal == "DEPLOYMENT" and row["source"] == "GITHUB_DEPLOYMENT":
            environment = str(row["environment"] or "")
            is_production = _matches(
                environment,
                settings.get("productionEnvironmentPatterns"),
                ["production", "prod", "live"],
            ) and not _is_excluded(environment, settings)
        elif row["source"] == "MANUAL":
            is_production = bool(row["is_production"])

        if is_production != bool(row["is_production"]):
            updates.append({"deployment_id": row["id"], "is_production": is_production})

    if updates:
        with database_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE deployments
                    SET is_production = :is_production,
                        updated_at = now(),
                        version = version + 1
                    WHERE id = :deployment_id
                    """
                ),
                updates,
            )
    return len(updates)


def upsert_deployment_from_workflow_run(
    database_engine: Engine,
    workspace_id: UUID,
    repository_id: UUID,
    event_data: dict[str, Any],
) -> UUID | None:
    """
    Normalize a ``workflow_run`` completed event into a ``deployments`` row.

    Only processes runs with conclusion ``success`` or ``failure``.
    Returns the DB UUID, or None if the run is not eligible.
    """
    workflow_run: dict[str, Any] = event_data.get("workflow_run", {})
    conclusion = (workflow_run.get("conclusion") or "").lower()

    if conclusion not in ("success", "failure"):
        logger.info(
            "workflow_run_skipped_non_terminal_conclusion",
            conclusion=conclusion,
            run_id=workflow_run.get("id"),
        )
        return None

    status = "SUCCESS" if conclusion == "success" else "FAILURE"
    environment = workflow_run.get("name", "unknown")

    settings = _get_repository_settings(database_engine, repository_id)
    repo = event_data.get("repository", {})
    default_branch = repo.get("default_branch", "main")
    head_branch = workflow_run.get("head_branch", "")
    is_production = _is_production_workflow(
        environment,
        head_branch,
        default_branch,
        settings,
    )

    row = {
        "workspace_id": str(workspace_id),
        "repository_id": str(repository_id),
        "source": "GITHUB_WORKFLOW",
        "external_deployment_id": str(workflow_run.get("id", "")),
        "environment": environment,
        "is_production": is_production,
        "status": status,
        "commit_sha": workflow_run.get("head_sha", ""),
        "started_at": _parse_ts(
            workflow_run.get("run_started_at") or workflow_run.get("created_at")
        ),
        "finished_at": _parse_ts(workflow_run.get("updated_at")),
        "raw_data": event_data,
    }

    deployment_id = _run_upsert(database_engine, row)
    if deployment_id and is_production:
        link_deployments_to_pull_requests(database_engine, repository_id)
        update_github_incident_lifecycle(database_engine, deployment_id)
        with database_engine.begin() as conn:
            enqueue_recalculate_metrics_job(
                conn,
                workspace_id,
                repository_id,
                affected_at=row["finished_at"],
            )

    return deployment_id


def upsert_deployment_from_deployment_status(
    database_engine: Engine,
    workspace_id: UUID,
    repository_id: UUID,
    event_data: dict[str, Any],
) -> UUID | None:
    """
    Normalize a ``deployment_status`` event into a ``deployments`` row.

    Only processes terminal states (success / failure).
    Returns the DB UUID, or None if not eligible.
    """
    deployment_status: dict[str, Any] = event_data.get("deployment_status", {})
    gh_state = (deployment_status.get("state") or "").lower()

    if gh_state not in ("success", "failure"):
        logger.info(
            "deployment_status_skipped_non_terminal_state",
            state=gh_state,
        )
        return None

    deployment: dict[str, Any] = event_data.get("deployment", {})
    status = "SUCCESS" if gh_state == "success" else "FAILURE"
    environment = deployment.get("environment", "unknown")

    settings = _get_repository_settings(database_engine, repository_id)
    is_production = _matches(
        environment,
        settings.get("productionEnvironmentPatterns"),
        ["production", "prod", "live"],
    ) and not _is_excluded(environment, settings)

    row = {
        "workspace_id": str(workspace_id),
        "repository_id": str(repository_id),
        "source": "GITHUB_DEPLOYMENT",
        "external_deployment_id": str(deployment.get("id", "")),
        "environment": environment,
        "is_production": is_production,
        "status": status,
        "commit_sha": deployment.get("sha", ""),
        "started_at": _parse_ts(deployment.get("created_at")),
        "finished_at": _parse_ts(deployment_status.get("updated_at")),
        "raw_data": event_data,
    }

    deployment_id = _run_upsert(database_engine, row)
    if deployment_id and is_production:
        link_deployments_to_pull_requests(database_engine, repository_id)
        update_github_incident_lifecycle(database_engine, deployment_id)
        with database_engine.begin() as conn:
            enqueue_recalculate_metrics_job(
                conn,
                workspace_id,
                repository_id,
                affected_at=row["finished_at"],
            )

    return deployment_id


# ---------------------------------------------------------------------------
# Shared upsert
# ---------------------------------------------------------------------------


def _run_upsert(database_engine: Engine, row: dict[str, Any]) -> UUID:
    """
    Insert or update on the unique (repository_id, source, external_deployment_id)
    conflict key.  Repeated delivery produces one stable row.
    """
    sql = text(
        """
        INSERT INTO deployments (
            workspace_id, repository_id, source, external_deployment_id,
            environment, is_production, status, commit_sha,
            started_at, finished_at, raw_data, updated_at, version
        ) VALUES (
            :workspace_id, :repository_id, :source, :external_deployment_id,
            :environment, :is_production, :status, :commit_sha,
            :started_at, :finished_at, :raw_data, now(), 0
        )
        ON CONFLICT (repository_id, source, external_deployment_id)
        DO UPDATE SET
            environment    = EXCLUDED.environment,
            is_production  = EXCLUDED.is_production,
            status         = EXCLUDED.status,
            commit_sha     = EXCLUDED.commit_sha,
            started_at     = COALESCE(deployments.started_at, EXCLUDED.started_at),
            finished_at    = EXCLUDED.finished_at,
            raw_data       = EXCLUDED.raw_data,
            updated_at     = now(),
            version        = deployments.version + 1
        WHERE deployments.finished_at IS NULL
           OR (
               EXCLUDED.finished_at IS NOT NULL
               AND EXCLUDED.finished_at >= deployments.finished_at
           )
        RETURNING id
        """
    )

    params = dict(row)
    params["raw_data"] = json.dumps(params["raw_data"])

    with database_engine.begin() as connection:
        deployment_id = connection.execute(sql, params).scalar_one_or_none()
        if deployment_id is None:
            # A delayed provider delivery can lose the monotonic update guard.
            # The canonical row still exists, so return its stable identifier.
            deployment_id = connection.execute(
                text(
                    """
                    SELECT id
                    FROM deployments
                    WHERE repository_id = :repository_id
                      AND source = :source
                      AND external_deployment_id = :external_deployment_id
                    """
                ),
                params,
            ).scalar_one()

    return UUID(str(deployment_id))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError, AttributeError:
        return None
