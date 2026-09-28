from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, text

from app.normalization.pull_requests import replace_pull_request_reviews, upsert_pull_request


@dataclass(frozen=True, slots=True)
class ReviewRows:
    workspace_id: UUID
    repository_id: UUID


@pytest.fixture
def review_rows(database_engine: Engine) -> Iterator[ReviewRows]:
    rows = ReviewRows(uuid4(), uuid4())
    integration_id = uuid4()
    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO workspaces (id, name, slug, timezone)
                VALUES (:id, 'Review Test', :slug, 'UTC')
                """
            ),
            {"id": rows.workspace_id, "slug": f"review-{rows.workspace_id.hex}"},
        )
        connection.execute(
            text(
                """
                INSERT INTO github_integrations (
                    id, workspace_id, installation_id, account_external_id,
                    account_login, account_type, repository_selection, status
                ) VALUES (
                    :id, :workspace_id, 7101, 8101, 'adept', 'ORGANIZATION', 'ALL', 'ACTIVE'
                )
                """
            ),
            {"id": integration_id, "workspace_id": rows.workspace_id},
        )
        connection.execute(
            text(
                """
                INSERT INTO repositories (
                    id, workspace_id, github_integration_id, github_repo_id, owner_login,
                    name, full_name, default_branch, visibility, tracking_enabled, settings
                ) VALUES (
                    :id, :workspace_id, :integration_id, 9101, 'adept', 'api', 'adept/api',
                    'main', 'PRIVATE', true, '{}'::jsonb
                )
                """
            ),
            {
                "id": rows.repository_id,
                "workspace_id": rows.workspace_id,
                "integration_id": integration_id,
            },
        )
    yield rows
    with database_engine.begin() as connection:
        connection.execute(text("DELETE FROM workspaces WHERE id = :id"), {"id": rows.workspace_id})


def _pull_request(updated_at: datetime, *, draft: bool) -> dict[str, Any]:
    return {
        "id": 5001,
        "number": 12,
        "title": "Add cycle time",
        "state": "open",
        "draft": draft,
        "user": {"login": "author"},
        "base": {"ref": "main"},
        "head": {"ref": "feature", "sha": "abc"},
        "created_at": "2026-09-01T09:00:00Z",
        "updated_at": updated_at.isoformat(),
    }


def _review(review_id: int, state: str, submitted_at: str) -> dict[str, Any]:
    return {
        "id": review_id,
        "user": {"login": "reviewer", "type": "User"},
        "state": state,
        "submitted_at": submitted_at,
        "commit_id": "abc",
    }


@pytest.mark.integration
def test_ready_for_review_time_survives_later_syncs(
    database_engine: Engine, review_rows: ReviewRows
) -> None:
    ready_at = datetime(2026, 9, 2, 8, 0, tzinfo=UTC)
    pull_request_id = upsert_pull_request(
        database_engine,
        review_rows.workspace_id,
        review_rows.repository_id,
        _pull_request(ready_at, draft=False),
        "ready_for_review",
    )
    upsert_pull_request(
        database_engine,
        review_rows.workspace_id,
        review_rows.repository_id,
        _pull_request(ready_at + timedelta(hours=3), draft=False),
        "synchronize",
    )

    with database_engine.connect() as connection:
        stored = connection.execute(
            text("SELECT ready_for_review_at FROM pull_requests WHERE id = :id"),
            {"id": pull_request_id},
        ).scalar_one()
    assert stored == ready_at


@pytest.mark.integration
def test_review_resync_replaces_rows_with_current_provider_state(
    database_engine: Engine, review_rows: ReviewRows
) -> None:
    pull_request_id = upsert_pull_request(
        database_engine,
        review_rows.workspace_id,
        review_rows.repository_id,
        _pull_request(datetime(2026, 9, 2, tzinfo=UTC), draft=False),
        "opened",
        reviews=[
            _review(1, "CHANGES_REQUESTED", "2026-09-02T10:00:00Z"),
            _review(2, "APPROVED", "2026-09-02T15:00:00Z"),
        ],
    )

    # The approval was later dismissed and a new approval was submitted.
    replace_pull_request_reviews(
        database_engine,
        pull_request_id,
        [
            _review(1, "CHANGES_REQUESTED", "2026-09-02T10:00:00Z"),
            _review(2, "DISMISSED", "2026-09-02T15:00:00Z"),
            _review(3, "APPROVED", "2026-09-03T09:00:00Z"),
            {"id": 4, "user": {"login": "reviewer"}, "state": "PENDING"},
        ],
    )

    with database_engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT github_review_id, state FROM pull_request_reviews
                WHERE pull_request_id = :id ORDER BY github_review_id
                """
            ),
            {"id": pull_request_id},
        ).all()
    assert [tuple(row) for row in rows] == [
        (1, "CHANGES_REQUESTED"),
        (2, "DISMISSED"),
        (3, "APPROVED"),
    ]
