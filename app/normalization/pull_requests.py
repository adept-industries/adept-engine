"""
Pull-request normalization.

Parses a GitHub `pull_request` event payload and upserts a canonical row
into the `pull_requests` table.  The upsert uses ON CONFLICT DO UPDATE so
that retries and re-deliveries converge on a single correct row.

Unique key: (repository_id, github_pr_id)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import Engine, text

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def upsert_pull_request(
    database_engine: Engine,
    workspace_id: UUID,
    repository_id: UUID,
    pr_data: dict[str, Any],
    action: str | None = None,
) -> UUID:
    """
    Parse *pr_data* (the ``pull_request`` object from the GitHub event) and
    upsert the corresponding row in ``pull_requests``.

    Returns the database UUID of the upserted row.
    """
    row = _build_row(workspace_id, repository_id, pr_data, action)
    return _run_upsert(database_engine, row)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_row(
    workspace_id: UUID,
    repository_id: UUID,
    pr: dict[str, Any],
    action: str | None,
) -> dict[str, Any]:
    """Transform a raw GitHub PR dict into a flat dict matching the DB schema."""
    merged = pr.get("merged", False)
    closed_at_raw = pr.get("closed_at")
    merged_at_raw = pr.get("merged_at")

    if pr.get("state") == "closed" and merged:
        state = "MERGED"
    elif pr.get("state") == "closed":
        state = "CLOSED"
    else:
        state = "OPEN"

    return {
        "workspace_id": str(workspace_id),
        "repository_id": str(repository_id),
        "github_pr_id": int(pr["id"]),
        "github_node_id": pr.get("node_id"),
        "number": int(pr["number"]),
        "title": pr.get("title", ""),
        "state": state,
        "draft": bool(pr.get("draft", False)),
        "author_login": _nested(pr, "user", "login"),
        "base_ref": _nested(pr, "base", "ref") or "",
        "head_ref": _nested(pr, "head", "ref") or "",
        "head_sha": _nested(pr, "head", "sha"),
        "merge_commit_sha": pr.get("merge_commit_sha"),
        "additions": int(pr.get("additions", 0)),
        "deletions": int(pr.get("deletions", 0)),
        "changed_files": int(pr.get("changed_files", 0)),
        "commit_count": int(pr.get("commits", 0)),
        "opened_at": _parse_ts(pr.get("created_at")),
        "first_commit_at": None,  # enriched in Phase 6 when we fetch commits
        "closed_at": _parse_ts(closed_at_raw),
        "merged_at": _parse_ts(merged_at_raw),
        "provider_updated_at": _parse_ts(pr.get("updated_at")),
        "raw_data": pr,
    }


def _run_upsert(database_engine: Engine, row: dict[str, Any]) -> UUID:
    """
    Insert or update on the unique (repository_id, github_pr_id) conflict key.
    All mutable columns are refreshed on conflict so retries produce the same result.
    """
    sql = text(
        """
        INSERT INTO pull_requests (
            workspace_id, repository_id, github_pr_id, github_node_id,
            number, title, state, draft, author_login,
            base_ref, head_ref, head_sha, merge_commit_sha,
            additions, deletions, changed_files, commit_count,
            opened_at, first_commit_at, closed_at, merged_at,
            last_synced_at, raw_data, updated_at, version
        ) VALUES (
            :workspace_id, :repository_id, :github_pr_id, :github_node_id,
            :number, :title, :state, :draft, :author_login,
            :base_ref, :head_ref, :head_sha, :merge_commit_sha,
            :additions, :deletions, :changed_files, :commit_count,
            :opened_at, :first_commit_at, :closed_at, :merged_at,
            COALESCE(:provider_updated_at, now()), :raw_data, now(), 0
        )
        ON CONFLICT (repository_id, github_pr_id)
        DO UPDATE SET
            title             = EXCLUDED.title,
            state             = EXCLUDED.state,
            draft             = EXCLUDED.draft,
            author_login      = EXCLUDED.author_login,
            head_sha          = EXCLUDED.head_sha,
            merge_commit_sha  = EXCLUDED.merge_commit_sha,
            additions         = EXCLUDED.additions,
            deletions         = EXCLUDED.deletions,
            changed_files     = EXCLUDED.changed_files,
            commit_count      = EXCLUDED.commit_count,
            closed_at         = EXCLUDED.closed_at,
            merged_at         = EXCLUDED.merged_at,
            last_synced_at    = now(),
            raw_data          = EXCLUDED.raw_data,
            updated_at        = now(),
            version           = pull_requests.version + 1
        WHERE pull_requests.last_synced_at IS NULL
           OR (
               EXCLUDED.last_synced_at IS NOT NULL
               AND EXCLUDED.last_synced_at >= pull_requests.last_synced_at
           )
        RETURNING id
        """
    )

    # Serialize raw_data as JSON for psycopg
    import json

    params = dict(row)
    params["raw_data"] = json.dumps(params["raw_data"])

    with database_engine.begin() as connection:
        pr_id = connection.execute(sql, params).scalar_one_or_none()
        if pr_id is None:
            # A stale delivery was intentionally ignored; callers still receive
            # the stable canonical identifier for idempotent processing.
            pr_id = connection.execute(
                text(
                    """
                    SELECT id
                    FROM pull_requests
                    WHERE repository_id = :repository_id
                      AND github_pr_id = :github_pr_id
                    """
                ),
                params,
            ).scalar_one()

    return UUID(str(pr_id))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _nested(d: dict[str, Any], *keys: str) -> Any:
    """Safely traverse a nested dict path, returning None on any missing key."""
    current: Any = d
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _parse_ts(value: str | None) -> datetime | None:
    """Parse a GitHub ISO-8601 timestamp string into a timezone-aware datetime."""
    if not value:
        return None
    try:
        # GitHub timestamps end with 'Z'; Python 3.11+ fromisoformat handles it.
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError, AttributeError:
        return None
