from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import Engine, text

from app.jobs.retry import PermanentJobError

logger = structlog.get_logger()


def upsert_github_issue(
    database_engine: Engine,
    workspace_id: UUID,
    repository_id: UUID,
    issue_payload: dict[str, Any],
    *,
    observed_at: datetime | None = None,
) -> UUID | None:
    """Normalize one GitHub issue; pull-request-shaped issue records are ignored."""
    if isinstance(issue_payload.get("pull_request"), dict):
        return None

    provider_id = _positive_int(issue_payload.get("id"), "issue id")
    number = _positive_int(issue_payload.get("number"), "issue number")
    title = issue_payload.get("title")
    state = str(issue_payload.get("state", "")).upper()
    created_at = _timestamp(issue_payload.get("created_at"))
    updated_at = _timestamp(issue_payload.get("updated_at"))
    if not isinstance(title, str) or not title or state not in {"OPEN", "CLOSED"}:
        raise PermanentJobError("GitHub returned an invalid issue record")
    if created_at is None:
        raise PermanentJobError("GitHub returned an issue without created_at")

    user = issue_payload.get("user")
    author_login = _login(user)
    assignees_raw = issue_payload.get("assignees")
    assignees = (
        [login for item in assignees_raw if (login := _login(item)) is not None]
        if isinstance(assignees_raw, list)
        else []
    )
    labels_raw = issue_payload.get("labels")
    labels = (
        [name for item in labels_raw if (name := _label_name(item)) is not None]
        if isinstance(labels_raw, list)
        else []
    )
    comments = _non_negative_int(issue_payload.get("comments", 0), "comments")
    closed_at = _timestamp(issue_payload.get("closed_at"))
    synced_at = observed_at or datetime.now(UTC)

    with database_engine.begin() as connection:
        issue_id = connection.execute(
            text(
                """
                INSERT INTO github_issues (
                    workspace_id, repository_id, github_issue_id, github_node_id,
                    number, title, state, author_login, assignee_logins, labels,
                    comments_count, github_created_at, github_updated_at, closed_at,
                    last_synced_at, raw_data
                ) VALUES (
                    :workspace_id, :repository_id, :github_issue_id, :github_node_id,
                    :number, :title, :state, :author_login, :assignee_logins, :labels,
                    :comments_count, :github_created_at, :github_updated_at, :closed_at,
                    :last_synced_at, cast(:raw_data as jsonb)
                )
                ON CONFLICT (repository_id, github_issue_id) DO UPDATE SET
                    github_node_id = EXCLUDED.github_node_id,
                    number = EXCLUDED.number,
                    title = EXCLUDED.title,
                    state = EXCLUDED.state,
                    author_login = EXCLUDED.author_login,
                    assignee_logins = EXCLUDED.assignee_logins,
                    labels = EXCLUDED.labels,
                    comments_count = EXCLUDED.comments_count,
                    github_created_at = EXCLUDED.github_created_at,
                    github_updated_at = EXCLUDED.github_updated_at,
                    closed_at = EXCLUDED.closed_at,
                    last_synced_at = :last_synced_at,
                    raw_data = EXCLUDED.raw_data,
                    updated_at = now(),
                    version = github_issues.version + 1
                WHERE github_issues.github_updated_at IS NULL
                   OR EXCLUDED.github_updated_at IS NULL
                   OR EXCLUDED.github_updated_at >= github_issues.github_updated_at
                RETURNING id
                """
            ),
            {
                "workspace_id": workspace_id,
                "repository_id": repository_id,
                "github_issue_id": provider_id,
                "github_node_id": _optional_string(issue_payload.get("node_id")),
                "number": number,
                "title": title,
                "state": state,
                "author_login": author_login,
                "assignee_logins": assignees,
                "labels": labels,
                "comments_count": comments,
                "github_created_at": created_at,
                "github_updated_at": updated_at,
                "closed_at": closed_at,
                "last_synced_at": synced_at,
                "raw_data": json.dumps(issue_payload),
            },
        ).scalar_one_or_none()

    if issue_id is not None:
        logger.info(
            "github_issue_upserted",
            repository_id=str(repository_id),
            issue_number=number,
            state=state,
        )
        return UUID(str(issue_id))
    return None


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _positive_int(value: object, field_name: str) -> int:
    parsed = _non_negative_int(value, field_name)
    if parsed == 0:
        raise PermanentJobError(f"GitHub returned an invalid {field_name}")
    return parsed


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise PermanentJobError(f"GitHub returned an invalid {field_name}")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise PermanentJobError(f"GitHub returned an invalid {field_name}") from exc
    if parsed < 0:
        raise PermanentJobError(f"GitHub returned an invalid {field_name}")
    return parsed


def _login(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    return _optional_string(value.get("login"))


def _label_name(value: object) -> str | None:
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        return _optional_string(value.get("name"))
    return None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
