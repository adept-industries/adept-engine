"""
Deployment normalization.

Parses GitHub ``workflow_run`` (completed) and ``deployment_status``
(success / failure) events and upserts rows into ``deployments``.

Unique key: (repository_id, source, external_deployment_id)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import Engine, text

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


import fnmatch
from app.metrics.service import enqueue_recalculate_metrics_job, link_deployments_to_pull_requests


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


def _is_branch_production(head_branch: str, default_branch: str, patterns_str: str | None) -> bool:
    if not head_branch:
        return False
    if patterns_str and patterns_str.strip():
        patterns = [p.strip() for p in patterns_str.split(",") if p.strip()]
        return any(fnmatch.fnmatch(head_branch, p) for p in patterns)
    return head_branch == default_branch


def _is_env_production(environment: str, patterns_str: str | None) -> bool:
    if not environment:
        return False
    if patterns_str and patterns_str.strip():
        patterns = [p.strip() for p in patterns_str.split(",") if p.strip()]
        return any(fnmatch.fnmatch(environment.lower(), p.lower()) for p in patterns)
    return environment.lower() in ("production", "prod", "live")


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
    branch_patterns = settings.get("productionBranchPatterns")
    repo = event_data.get("repository", {})
    default_branch = repo.get("default_branch", "main")
    head_branch = workflow_run.get("head_branch", "")
    is_production = _is_branch_production(head_branch, default_branch, branch_patterns)

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
        try:
            link_deployments_to_pull_requests(database_engine, repository_id)
            with database_engine.begin() as conn:
                enqueue_recalculate_metrics_job(conn, workspace_id, repository_id)
        except Exception as exc:
            logger.warning("metrics_recalculate_enqueue_failed", error=str(exc))

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
    env_patterns = settings.get("productionEnvironmentPatterns")
    is_production = _is_env_production(environment, env_patterns)

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
        try:
            link_deployments_to_pull_requests(database_engine, repository_id)
            with database_engine.begin() as conn:
                enqueue_recalculate_metrics_job(conn, workspace_id, repository_id)
        except Exception as exc:
            logger.warning("metrics_recalculate_enqueue_failed", error=str(exc))

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
