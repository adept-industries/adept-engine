from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, text

from app.db.models import ClaimedJob
from app.jobs.handlers import backfill_repository
from app.jobs.retry import PermanentJobError
from app.metrics.service import recalculate_repository_metrics
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


@pytest.mark.integration
def test_recalculation_measures_review_stages_from_human_reviews(
    database_engine: Engine, review_rows: ReviewRows
) -> None:
    monday = datetime(2026, 9, 21, tzinfo=UTC)
    merged_at = monday + timedelta(days=2, hours=17)

    def at(hours: float) -> str:
        return (monday + timedelta(hours=hours)).isoformat()

    pull_request = {
        **_pull_request(merged_at, draft=False),
        "state": "closed",
        "merged": True,
        "created_at": at(15),
        "closed_at": merged_at.isoformat(),
        "merged_at": merged_at.isoformat(),
        "merge_commit_sha": "merge42",
        "additions": 120,
        "deletions": 30,
    }
    commits = [{"sha": "c1", "commit": {"author": {"date": at(9)}}}]
    reviews = [
        # Bots and the author answering threads are not reviewers.
        {**_review(1, "COMMENTED", at(16)), "user": {"login": "ci[bot]", "type": "Bot"}},
        {**_review(2, "COMMENTED", at(17)), "user": {"login": "author", "type": "User"}},
        _review(3, "CHANGES_REQUESTED", at(59)),  # Wed 11:00: first human review
        _review(4, "APPROVED", at(62)),
        _review(5, "APPROVED", at(64)),  # Wed 16:00: final approval before merge
        _review(6, "APPROVED", at(70)),  # after merge, ignored
    ]
    upsert_pull_request(
        database_engine,
        review_rows.workspace_id,
        review_rows.repository_id,
        pull_request,
        "closed",
        commits,
        reviews,
    )
    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO deployments (
                    workspace_id, repository_id, source, external_deployment_id,
                    environment, is_production, status, commit_sha, started_at, finished_at
                ) VALUES (
                    :workspace_id, :repository_id, 'GITHUB_WORKFLOW', 'run-1',
                    'production', true, 'SUCCESS', 'merge42', :finished_at, :finished_at
                )
                """
            ),
            {
                "workspace_id": review_rows.workspace_id,
                "repository_id": review_rows.repository_id,
                "finished_at": monday + timedelta(days=3, hours=10),
            },
        )

    recalculate_repository_metrics(
        database_engine,
        review_rows.workspace_id,
        review_rows.repository_id,
        from_date=monday,
        to_date=monday + timedelta(days=7),
    )

    with database_engine.connect() as connection:
        rows = (
            connection.execute(
                text(
                    """
                    SELECT metric_type, value, sample_size, dimensions
                    FROM metric_snapshots
                    WHERE repository_id = :repository_id
                      AND granularity = 'DAY'
                      AND calculation_version = 'cycle-time-v2'
                      AND period_start = :merge_day
                    """
                ),
                {
                    "repository_id": review_rows.repository_id,
                    "merge_day": monday + timedelta(days=2),
                },
            )
            .mappings()
            .all()
        )
    values = {row["metric_type"]: float(row["value"]) for row in rows}
    assert values == {
        "PR_CODING_TIME_HOURS": 6.0,
        "PR_PICKUP_TIME_HOURS": 44.0,
        "PR_REVIEW_TIME_HOURS": 6.0,
        "PR_DEPLOY_TIME_HOURS": 17.0,
    }
    pickup = next(row for row in rows if row["metric_type"] == "PR_PICKUP_TIME_HOURS")
    assert pickup["sample_size"] == 1
    observation = pickup["dimensions"]["observations"][0]
    assert observation["reviewed"] is True


def _backfill_job(payload: dict[str, Any]) -> ClaimedJob:
    now = datetime.now(UTC)
    return ClaimedJob(
        id=uuid4(),
        job_type="BACKFILL_REPOSITORY",
        payload=payload,
        priority=200,
        attempts=1,
        max_attempts=8,
        locked_by="test-worker",
        created_at=now,
        updated_at=now,
        version=1,
    )


def test_backfill_modes_are_exclusive() -> None:
    job = _backfill_job({"repositoryId": str(uuid4()), "reviewsOnly": True, "riskOnly": True})

    with pytest.raises(PermanentJobError, match="exclusive"):
        backfill_repository.handle_backfill_repository(MagicMock(), job, "test-worker")


@pytest.mark.integration
def test_reviews_only_backfill_refreshes_merged_pull_requests_in_the_window(
    database_engine: Engine,
    review_rows: ReviewRows,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC).replace(microsecond=0)

    def merged(github_id: int, number: int, merged_at: datetime) -> UUID:
        return upsert_pull_request(
            database_engine,
            review_rows.workspace_id,
            review_rows.repository_id,
            {
                **_pull_request(merged_at, draft=False),
                "id": github_id,
                "number": number,
                "state": "closed",
                "merged": True,
                "created_at": (merged_at - timedelta(hours=5)).isoformat(),
                "closed_at": merged_at.isoformat(),
                "merged_at": merged_at.isoformat(),
            },
            "closed",
        )

    recent_id = merged(6001, 21, now - timedelta(days=2))
    old_id = merged(6002, 22, now - timedelta(days=120))
    client = MagicMock()
    client.__enter__.return_value = client
    client.list_pull_request_reviews.return_value = [
        _review(1, "APPROVED", (now - timedelta(days=2, hours=3)).isoformat())
    ]
    monkeypatch.setattr(backfill_repository, "GithubClient", MagicMock(return_value=client))
    monkeypatch.setattr(backfill_repository, "get_settings", MagicMock())

    backfill_repository.handle_backfill_repository(
        database_engine,
        _backfill_job({"repositoryId": str(review_rows.repository_id), "reviewsOnly": True}),
        "test-worker",
    )

    # Only the PR merged inside the default 90-day window is refreshed.
    client.list_pull_request_reviews.assert_called_once_with("adept", "api", 21)
    with database_engine.connect() as connection:
        reviewed = dict(
            connection.execute(
                text(
                    """
                    SELECT pull_request_id, count(*) FROM pull_request_reviews
                    WHERE pull_request_id IN (:recent, :old) GROUP BY pull_request_id
                    """
                ),
                {"recent": recent_id, "old": old_id},
            ).all()
        )
        pickup_samples = connection.execute(
            text(
                """
                SELECT sum(sample_size) FROM metric_snapshots
                WHERE repository_id = :repository_id
                  AND metric_type = 'PR_PICKUP_TIME_HOURS'
                  AND granularity = 'DAY'
                  AND calculation_version = 'cycle-time-v2'
                """
            ),
            {"repository_id": review_rows.repository_id},
        ).scalar_one()
    assert reviewed == {recent_id: 1}
    assert pickup_samples == 1
