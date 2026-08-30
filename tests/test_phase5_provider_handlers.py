from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import Engine, text

from app.core.config import Settings
from app.db.models import ClaimedJob
from app.jobs.claimer import claim_jobs
from app.jobs.dispatcher import dispatch_job
from app.jobs.handlers import (
    backfill_repository,
    github_event,
    jira_event,
    provider_support,
    renew_jira_webhook,
    sync_github_repositories,
    sync_jira_projects,
)
from app.jobs.retry import PermanentJobError
from app.metrics.service import link_deployments_to_pull_requests
from app.normalization import deployments as deployment_normalization
from app.normalization import pull_requests as pull_request_normalization
from app.providers.crypto import encrypt_integration_secret
from app.providers.github import ProviderPage
from app.providers.jira import JiraOAuthTokens
from tests.conftest import JobFactory


@dataclass(frozen=True, slots=True)
class ProviderRows:
    user_id: UUID
    workspace_id: UUID
    github_integration_id: UUID
    repository_id: UUID
    jira_integration_id: UUID
    jira_project_id: UUID


@pytest.fixture
def provider_rows(database_engine: Engine) -> Iterator[ProviderRows]:
    values = ProviderRows(*(uuid4() for _ in range(6)))
    suffix = values.workspace_id.hex
    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO users (id, email, password_hash, display_name)
                VALUES (:id, :email, 'hash', 'Provider Test')
                """
            ),
            {"id": values.user_id, "email": f"provider-{suffix}@example.test"},
        )
        connection.execute(
            text(
                """
                INSERT INTO workspaces (id, name, slug, timezone)
                VALUES (:id, 'Provider Test', :slug, 'UTC')
                """
            ),
            {"id": values.workspace_id, "slug": f"provider-{suffix}"},
        )
        connection.execute(
            text(
                """
                INSERT INTO github_integrations (
                    id, workspace_id, installation_id, account_external_id,
                    account_login, account_type, repository_selection, status
                ) VALUES (
                    :id, :workspace_id, 7001, 8001,
                    'adept-industries', 'ORGANIZATION', 'ALL', 'ACTIVE'
                )
                """
            ),
            {
                "id": values.github_integration_id,
                "workspace_id": values.workspace_id,
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO repositories (
                    id, workspace_id, github_integration_id, github_repo_id,
                    owner_login, name, full_name, default_branch, visibility,
                    tracking_enabled, settings
                ) VALUES (
                    :id, :workspace_id, :integration_id, 9001,
                    'adept-industries', 'engine', 'adept-industries/engine',
                    'main', 'PRIVATE', true, CAST(:settings AS jsonb)
                )
                """
            ),
            {
                "id": values.repository_id,
                "workspace_id": values.workspace_id,
                "integration_id": values.github_integration_id,
                "settings": json.dumps({"deploymentSignal": "WORKFLOW_RUN"}),
            },
        )
        has_webhook_token_hash = bool(
            connection.execute(
                text(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'jira_integrations'
                          AND column_name = 'webhook_token_hash'
                    )
                    """
                )
            ).scalar()
        )
        if has_webhook_token_hash:
            connection.execute(
                text(
                    """
                    INSERT INTO jira_integrations (
                        id, workspace_id, cloud_id, site_url, display_name,
                        access_token_enc, refresh_token_enc, encryption_key_version,
                        access_token_expires_at, status, webhook_id,
                        webhook_expires_at, webhook_token_hash
                    ) VALUES (
                        :id, :workspace_id, 'cloud-1', 'https://example.atlassian.net',
                        'Example Jira', 'unused', 'unused', 1, now() + interval '1 day',
                        'ACTIVE', 991, now() + interval '20 days', :webhook_token_hash
                    )
                    """
                ),
                {
                    "id": values.jira_integration_id,
                    "workspace_id": values.workspace_id,
                    "webhook_token_hash": "a" * 64,
                },
            )
        else:
            connection.execute(
                text(
                    """
                    INSERT INTO jira_integrations (
                        id, workspace_id, cloud_id, site_url, display_name,
                        access_token_enc, refresh_token_enc, encryption_key_version,
                        access_token_expires_at, status, webhook_id,
                        webhook_expires_at
                    ) VALUES (
                        :id, :workspace_id, 'cloud-1', 'https://example.atlassian.net',
                        'Example Jira', 'unused', 'unused', 1, now() + interval '1 day',
                        'ACTIVE', 991, now() + interval '20 days'
                    )
                    """
                ),
                {
                    "id": values.jira_integration_id,
                    "workspace_id": values.workspace_id,
                },
            )
        connection.execute(
            text(
                """
                INSERT INTO jira_projects (
                    id, workspace_id, jira_integration_id, jira_project_id,
                    project_key, project_name, project_type, tracking_enabled
                ) VALUES (
                    :id, :workspace_id, :integration_id, '10000',
                    'ADEPT', 'Adept', 'software', true
                )
                """
            ),
            {
                "id": values.jira_project_id,
                "workspace_id": values.workspace_id,
                "integration_id": values.jira_integration_id,
            },
        )

    yield values

    with database_engine.begin() as connection:
        connection.execute(
            text("DELETE FROM workspaces WHERE id = :id"), {"id": values.workspace_id}
        )
        connection.execute(text("DELETE FROM users WHERE id = :id"), {"id": values.user_id})


def _job(job_type: str, payload: dict[str, Any]) -> ClaimedJob:
    now = datetime.now(UTC)
    return ClaimedJob(
        id=uuid4(),
        job_type=job_type,
        payload=payload,
        priority=100,
        attempts=1,
        max_attempts=8,
        locked_by="provider-test-worker",
        created_at=now,
        updated_at=now,
        version=1,
    )


def _raw_event(
    database_engine: Engine,
    rows: ProviderRows,
    *,
    source: str,
    event_type: str,
    action: str | None,
    payload: dict[str, Any],
    repository_id: UUID | None = None,
) -> UUID:
    raw_event_id = uuid4()
    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO raw_webhook_events (
                    id, workspace_id, repository_id, source, delivery_id,
                    event_type, action, payload, status, signature_valid
                ) VALUES (
                    :id, :workspace_id, :repository_id, :source, :delivery_id,
                    :event_type, :action, CAST(:payload AS jsonb), 'QUEUED', true
                )
                """
            ),
            {
                "id": raw_event_id,
                "workspace_id": rows.workspace_id,
                "repository_id": repository_id,
                "source": source,
                "delivery_id": f"test-{raw_event_id}",
                "event_type": event_type,
                "action": action,
                "payload": json.dumps(payload),
            },
        )
    return raw_event_id


def test_workflow_production_classification_requires_branch_and_workflow_patterns() -> None:
    settings = {
        "productionBranchPatterns": ["main", "release/*"],
        "deploymentWorkflowNamePatterns": ["deploy-*"],
        "doraExclusions": ["*preview*"],
    }

    assert deployment_normalization._is_production_workflow(
        "deploy-production", "main", "main", settings
    )
    assert not deployment_normalization._is_production_workflow(
        "unit-tests", "main", "main", settings
    )
    assert not deployment_normalization._is_production_workflow(
        "deploy-production", "feature/example", "main", settings
    )
    assert not deployment_normalization._is_production_workflow(
        "deploy-preview", "main", "main", settings
    )


@pytest.mark.integration
def test_production_outcomes_drive_incidents_and_one_pending_recalculation(
    database_engine: Engine,
    provider_rows: ProviderRows,
) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE repositories
                SET settings = CAST(:settings AS jsonb)
                WHERE id = :repository_id
                """
            ),
            {
                "repository_id": provider_rows.repository_id,
                "settings": json.dumps(
                    {
                        "productionBranchPatterns": ["main"],
                        "deploymentWorkflowNamePatterns": ["deploy-*"],
                        "doraExclusions": ["*preview*"],
                    }
                ),
            },
        )

    def workflow(run_id: int, name: str, conclusion: str, finished_at: datetime) -> dict[str, Any]:
        return {
            "repository": {"default_branch": "main"},
            "workflow_run": {
                "id": run_id,
                "name": name,
                "conclusion": conclusion,
                "head_branch": "main",
                "head_sha": f"sha-{run_id}",
                "run_started_at": (finished_at - timedelta(minutes=5)).isoformat(),
                "updated_at": finished_at.isoformat(),
            },
        }

    deployment_normalization.upsert_deployment_from_workflow_run(
        database_engine,
        provider_rows.workspace_id,
        provider_rows.repository_id,
        workflow(6001, "unit-tests", "success", now),
    )
    failed_event = workflow(6002, "deploy-production", "failure", now + timedelta(minutes=10))
    failed_id = deployment_normalization.upsert_deployment_from_workflow_run(
        database_engine,
        provider_rows.workspace_id,
        provider_rows.repository_id,
        failed_event,
    )
    deployment_normalization.upsert_deployment_from_workflow_run(
        database_engine,
        provider_rows.workspace_id,
        provider_rows.repository_id,
        failed_event,
    )
    recovered_id = deployment_normalization.upsert_deployment_from_workflow_run(
        database_engine,
        provider_rows.workspace_id,
        provider_rows.repository_id,
        workflow(6003, "deploy-production", "success", now + timedelta(minutes=25)),
    )

    with database_engine.connect() as connection:
        production_flags = connection.execute(
            text(
                """
                SELECT external_deployment_id, is_production
                FROM deployments
                WHERE repository_id = :repository_id
                ORDER BY external_deployment_id
                """
            ),
            {"repository_id": provider_rows.repository_id},
        ).all()
        incident = (
            connection.execute(
                text(
                    """
                    SELECT status, failed_deployment_id, recovery_deployment_id, resolved_at
                    FROM incidents
                    WHERE repository_id = :repository_id AND source = 'GITHUB'
                    """
                ),
                {"repository_id": provider_rows.repository_id},
            )
            .mappings()
            .one()
        )
        pending_job = (
            connection.execute(
                text(
                    """
                    SELECT payload
                    FROM processing_jobs
                    WHERE repository_id = :repository_id
                      AND job_type = 'RECALCULATE_METRICS'
                      AND status = 'PENDING'
                    """
                ),
                {"repository_id": provider_rows.repository_id},
            )
            .mappings()
            .one()
        )

    assert production_flags == [("6001", False), ("6002", True), ("6003", True)]
    assert incident["status"] == "RESOLVED"
    assert incident["failed_deployment_id"] == failed_id
    assert incident["recovery_deployment_id"] == recovered_id
    assert incident["resolved_at"] == now + timedelta(minutes=25)
    assert datetime.fromisoformat(pending_job["payload"]["affected_from"]) == now + timedelta(
        minutes=10
    )
    assert datetime.fromisoformat(pending_job["payload"]["affected_to"]) == now + timedelta(
        minutes=25
    )


@pytest.mark.integration
def test_incident_reconciliation_is_chronological_when_events_arrive_out_of_order(
    database_engine: Engine,
    provider_rows: ProviderRows,
) -> None:
    now = datetime.now(UTC).replace(microsecond=0)

    def workflow(run_id: int, conclusion: str, finished_at: datetime) -> dict[str, Any]:
        return {
            "repository": {"default_branch": "main"},
            "workflow_run": {
                "id": run_id,
                "name": "deploy-production",
                "conclusion": conclusion,
                "head_branch": "main",
                "head_sha": f"sha-{run_id}",
                "run_started_at": (finished_at - timedelta(minutes=5)).isoformat(),
                "updated_at": finished_at.isoformat(),
            },
        }

    recovery_id = deployment_normalization.upsert_deployment_from_workflow_run(
        database_engine,
        provider_rows.workspace_id,
        provider_rows.repository_id,
        workflow(6102, "success", now + timedelta(minutes=20)),
    )
    failed_id = deployment_normalization.upsert_deployment_from_workflow_run(
        database_engine,
        provider_rows.workspace_id,
        provider_rows.repository_id,
        workflow(6101, "failure", now),
    )

    with database_engine.connect() as connection:
        incident = (
            connection.execute(
                text(
                    """
                    SELECT status, failed_deployment_id, recovery_deployment_id, resolved_at
                    FROM incidents
                    WHERE repository_id = :repository_id AND source = 'GITHUB'
                    """
                ),
                {"repository_id": provider_rows.repository_id},
            )
            .mappings()
            .one()
        )

    assert incident["status"] == "RESOLVED"
    assert incident["failed_deployment_id"] == failed_id
    assert incident["recovery_deployment_id"] == recovery_id
    assert incident["resolved_at"] == now + timedelta(minutes=20)


@pytest.mark.integration
def test_deployment_link_rebuild_uses_exact_bootstrap_and_bounded_fallback(
    database_engine: Engine,
    provider_rows: ProviderRows,
) -> None:
    first_deployment_at = datetime.now(UTC).replace(microsecond=0) - timedelta(hours=3)

    def pull_request(
        github_id: int,
        number: int,
        head_sha: str,
        merged_at: datetime,
    ) -> None:
        pull_request_normalization.upsert_pull_request(
            database_engine,
            provider_rows.workspace_id,
            provider_rows.repository_id,
            {
                "id": github_id,
                "node_id": f"PR_{github_id}",
                "number": number,
                "title": f"PR {number}",
                "state": "closed",
                "merged": True,
                "draft": False,
                "user": {"login": "developer"},
                "base": {"ref": "main"},
                "head": {"ref": f"feature-{number}", "sha": head_sha},
                "merge_commit_sha": f"merge-{number}",
                "created_at": (merged_at - timedelta(hours=1)).isoformat(),
                "updated_at": merged_at.isoformat(),
                "closed_at": merged_at.isoformat(),
                "merged_at": merged_at.isoformat(),
                "additions": 1,
                "deletions": 0,
                "changed_files": 1,
                "commits": 1,
            },
            "closed",
            [
                {
                    "sha": head_sha,
                    "commit": {"author": {"date": (merged_at - timedelta(hours=1)).isoformat()}},
                }
            ],
        )

    pull_request(6201, 1, "ancient-sha", first_deployment_at - timedelta(days=10))
    pull_request(6202, 2, "bootstrap-sha", first_deployment_at - timedelta(minutes=30))
    pull_request(6203, 3, "window-sha", first_deployment_at + timedelta(hours=1))

    def workflow(run_id: int, head_sha: str, finished_at: datetime) -> dict[str, Any]:
        return {
            "repository": {"default_branch": "main"},
            "workflow_run": {
                "id": run_id,
                "name": "deploy-production",
                "conclusion": "success",
                "head_branch": "main",
                "head_sha": head_sha,
                "run_started_at": (finished_at - timedelta(minutes=5)).isoformat(),
                "updated_at": finished_at.isoformat(),
            },
        }

    deployment_normalization.upsert_deployment_from_workflow_run(
        database_engine,
        provider_rows.workspace_id,
        provider_rows.repository_id,
        workflow(6201, "bootstrap-sha", first_deployment_at),
    )
    deployment_normalization.upsert_deployment_from_workflow_run(
        database_engine,
        provider_rows.workspace_id,
        provider_rows.repository_id,
        workflow(6202, "second-deployment-sha", first_deployment_at + timedelta(hours=2)),
    )
    link_deployments_to_pull_requests(database_engine, provider_rows.repository_id)

    with database_engine.connect() as connection:
        links = connection.execute(
            text(
                """
                SELECT d.external_deployment_id, pr.number, dpr.link_method
                FROM deployment_pull_requests dpr
                JOIN deployments d ON d.id = dpr.deployment_id
                JOIN pull_requests pr ON pr.id = dpr.pull_request_id
                WHERE d.repository_id = :repository_id
                ORDER BY d.external_deployment_id, pr.number
                """
            ),
            {"repository_id": provider_rows.repository_id},
        ).all()

    assert links == [("6201", 2, "MERGE_SHA"), ("6202", 3, "MERGE_WINDOW")]


@pytest.mark.integration
def test_github_lifecycle_events_work_without_repository_id(
    database_engine: Engine,
    provider_rows: ProviderRows,
) -> None:
    suspend_id = _raw_event(
        database_engine,
        provider_rows,
        source="GITHUB",
        event_type="installation",
        action="suspend",
        payload={"installation": {"id": 7001}},
    )
    github_event.handle_process_github_event(
        database_engine,
        _job("PROCESS_GITHUB_EVENT", {"rawEventId": str(suspend_id)}),
        "provider-test-worker",
    )

    with database_engine.connect() as connection:
        integration = (
            connection.execute(
                text("SELECT status, suspended_at FROM github_integrations WHERE id = :id"),
                {"id": provider_rows.github_integration_id},
            )
            .mappings()
            .one()
        )
        raw = (
            connection.execute(
                text(
                    """
                    SELECT status, attempt_count, processed_at
                    FROM raw_webhook_events WHERE id = :id
                    """
                ),
                {"id": suspend_id},
            )
            .mappings()
            .one()
        )
    assert integration["status"] == "SUSPENDED"
    assert integration["suspended_at"] is not None
    assert raw["status"] == "PROCESSED"
    assert raw["attempt_count"] == 1
    assert raw["processed_at"] is not None

    # Restore the integration so the following lifecycle deliveries represent
    # a normal active installation.
    with database_engine.begin() as connection:
        connection.execute(
            text("UPDATE github_integrations SET status = 'ACTIVE' WHERE id = :id"),
            {"id": provider_rows.github_integration_id},
        )

    repositories_id = _raw_event(
        database_engine,
        provider_rows,
        source="GITHUB",
        event_type="installation_repositories",
        action="added",
        payload={
            "installation": {"id": 7001},
            # GitHub uses this compact shape for installation repository changes.
            "repositories_added": [
                {
                    "id": 9002,
                    "node_id": "R_9002",
                    "name": "compact",
                    "full_name": "other-owner/compact",
                    "private": True,
                }
            ],
            "repositories_removed": [{"id": 9001}],
        },
    )
    github_event.handle_process_github_event(
        database_engine,
        _job("PROCESS_GITHUB_EVENT", {"rawEventId": str(repositories_id)}),
        "provider-test-worker",
    )

    rename_id = _raw_event(
        database_engine,
        provider_rows,
        source="GITHUB",
        event_type="repository",
        action="renamed",
        payload={
            "installation": {"id": 7001},
            "repository": {
                "id": 9001,
                "name": "engine-renamed",
                "full_name": "new-owner/engine-renamed",
                "owner": {"login": "new-owner"},
                "default_branch": "trunk",
                "visibility": "private",
                "archived": False,
            },
        },
    )
    github_event.handle_process_github_event(
        database_engine,
        _job("PROCESS_GITHUB_EVENT", {"rawEventId": str(rename_id)}),
        "provider-test-worker",
    )

    with database_engine.begin() as connection:
        connection.execute(
            text("UPDATE repositories SET tracking_enabled = true WHERE id = :id"),
            {"id": provider_rows.repository_id},
        )
    deleted_id = _raw_event(
        database_engine,
        provider_rows,
        source="GITHUB",
        event_type="installation",
        action="deleted",
        payload={"installation": {"id": 7001}},
    )
    github_event.handle_process_github_event(
        database_engine,
        _job("PROCESS_GITHUB_EVENT", {"rawEventId": str(deleted_id)}),
        "provider-test-worker",
    )

    with database_engine.connect() as connection:
        original = (
            connection.execute(
                text(
                    """
                SELECT owner_login, name, full_name, default_branch,
                       archived, tracking_enabled
                FROM repositories WHERE id = :id
                """
                ),
                {"id": provider_rows.repository_id},
            )
            .mappings()
            .one()
        )
        compact = (
            connection.execute(
                text(
                    """
                SELECT owner_login, default_branch, visibility, tracking_enabled
                FROM repositories
                WHERE workspace_id = :workspace_id AND github_repo_id = 9002
                """
                ),
                {"workspace_id": provider_rows.workspace_id},
            )
            .mappings()
            .one()
        )
    assert dict(original) == {
        "owner_login": "new-owner",
        "name": "engine-renamed",
        "full_name": "new-owner/engine-renamed",
        "default_branch": "trunk",
        "archived": False,
        "tracking_enabled": False,
    }
    assert dict(compact) == {
        "owner_login": "other-owner",
        "default_branch": "main",
        "visibility": "PRIVATE",
        "tracking_enabled": False,
    }
    with database_engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT status FROM github_integrations WHERE id = :id"),
                {"id": provider_rows.github_integration_id},
            ).scalar_one()
            == "REVOKED"
        )


@pytest.mark.integration
def test_github_repository_sync_is_idempotent_and_reconciles_catalog(
    database_engine: Engine,
    provider_rows: ProviderRows,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGithubClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeGithubClient:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def list_installation_repositories(self, page: int) -> ProviderPage[dict[str, Any]]:
            assert page == 1
            return ProviderPage(
                [
                    {
                        "id": 9001,
                        "name": "engine",
                        "full_name": "adept-industries/engine",
                        "owner": {"login": "adept-industries"},
                        "default_branch": "main",
                        "visibility": "PRIVATE",
                        "archived": True,
                    },
                    {
                        "id": 9002,
                        "name": "catalogued",
                        "full_name": "other-owner/catalogued",
                        "private": False,
                    },
                ],
                None,
            )

    monkeypatch.setattr(sync_github_repositories, "GithubClient", FakeGithubClient)
    job = _job(
        "SYNC_GITHUB_REPOSITORIES",
        {
            "workspaceId": str(provider_rows.workspace_id),
            "githubIntegrationId": str(provider_rows.github_integration_id),
        },
    )
    sync_github_repositories.handle_sync_github_repositories(database_engine, job, job.locked_by)
    sync_github_repositories.handle_sync_github_repositories(database_engine, job, job.locked_by)

    with database_engine.connect() as connection:
        rows = (
            connection.execute(
                text(
                    """
                SELECT github_repo_id, owner_login, archived, tracking_enabled
                FROM repositories WHERE workspace_id = :workspace_id
                ORDER BY github_repo_id
                """
                ),
                {"workspace_id": provider_rows.workspace_id},
            )
            .mappings()
            .all()
        )
    assert [row["github_repo_id"] for row in rows] == [9001, 9002]
    assert rows[0]["archived"] is True
    assert rows[0]["tracking_enabled"] is False
    assert rows[1]["owner_login"] == "other-owner"


@pytest.mark.integration
def test_repository_backfill_pages_without_spending_retry_attempts(
    database_engine: Engine,
    provider_rows: ProviderRows,
    job_factory: JobFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC).replace(microsecond=0)

    class FakeGithubClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeGithubClient:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def list_open_pull_requests(
            self, _owner: str, _repository: str, page: int
        ) -> ProviderPage[dict[str, Any]]:
            assert page == 1
            return ProviderPage([], None)

        def list_closed_pull_requests(
            self, _owner: str, _repository: str, page: int
        ) -> ProviderPage[dict[str, Any]]:
            assert page == 1
            return ProviderPage(
                [
                    {
                        "id": 4001,
                        "number": 4,
                        "merged_at": now.isoformat(),
                        "updated_at": now.isoformat(),
                    }
                ],
                None,
            )

        def get_pull_request(self, _owner: str, _repository: str, number: int) -> dict[str, Any]:
            assert number == 4
            return {
                "id": 4001,
                "number": 4,
                "title": "Deployable change",
                "state": "closed",
                "merged": True,
                "draft": False,
                "user": {"login": "developer"},
                "base": {"ref": "main"},
                "head": {"ref": "feature", "sha": "abc123"},
                "created_at": (now - timedelta(days=1)).isoformat(),
                "closed_at": now.isoformat(),
                "merged_at": now.isoformat(),
                "additions": 10,
                "deletions": 2,
                "changed_files": 3,
                "commits": 2,
            }

        def list_pull_request_commits(
            self, _owner: str, _repository: str, number: int
        ) -> list[dict[str, Any]]:
            assert number == 4
            return [
                {
                    "sha": "first123",
                    "commit": {"author": {"date": (now - timedelta(days=2)).isoformat()}},
                },
                {
                    "sha": "abc123",
                    "commit": {"author": {"date": (now - timedelta(days=1)).isoformat()}},
                },
            ]

        def list_workflow_runs(
            self,
            _owner: str,
            _repository: str,
            page: int,
            *,
            branch: str | None,
            created_from: str,
            created_to: str,
            per_page: int = 50,
        ) -> ProviderPage[dict[str, Any]]:
            assert page == 1
            assert branch is None
            assert created_from
            assert created_to
            return ProviderPage(
                [
                    {
                        "id": 5001,
                        "name": "deploy-production",
                        "conclusion": "success",
                        "head_branch": "main",
                        "head_sha": "abc123",
                        "run_started_at": now.isoformat(),
                        "updated_at": now.isoformat(),
                    }
                ],
                None,
            )

    monkeypatch.setattr(backfill_repository, "GithubClient", FakeGithubClient)
    job_id = job_factory.insert(
        job_type="BACKFILL_REPOSITORY",
        payload={
            "repositoryId": str(provider_rows.repository_id),
            "backfillDays": 30,
        },
    )
    first = claim_jobs(database_engine, "provider-test-worker", 1)[0]
    dispatch_job(database_engine, first, "provider-test-worker")
    requeued = job_factory.row(job_id)
    assert requeued["status"] == "PENDING"
    assert requeued["attempts"] == 0
    assert requeued["payload"]["cursor"]["stage"] == "pull_requests"
    assert requeued["payload"]["cursor"]["page"] == 1

    second = claim_jobs(database_engine, "provider-test-worker", 1)[0]
    dispatch_job(database_engine, second, "provider-test-worker")
    requeued = job_factory.row(job_id)
    assert requeued["status"] == "PENDING"
    assert requeued["attempts"] == 0
    assert requeued["payload"]["cursor"]["stage"] == "workflow_runs"
    assert requeued["payload"]["cursor"]["page"] == 1

    third = claim_jobs(database_engine, "provider-test-worker", 1)[0]
    dispatch_job(database_engine, third, "provider-test-worker")
    assert job_factory.row(job_id)["status"] == "SUCCEEDED"
    with database_engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT count(*) FROM pull_requests WHERE repository_id = :id"),
                {"id": provider_rows.repository_id},
            ).scalar_one()
            == 1
        )
        assert (
            connection.execute(
                text("SELECT count(*) FROM deployments WHERE repository_id = :id"),
                {"id": provider_rows.repository_id},
            ).scalar_one()
            == 1
        )


@pytest.mark.integration
def test_jira_project_sync_preserves_tracking_and_is_idempotent(
    database_engine: Engine,
    provider_rows: ProviderRows,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeJiraClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeJiraClient:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def list_projects(self, start_at: int) -> ProviderPage[dict[str, Any]]:
            assert start_at == 0
            return ProviderPage(
                [
                    {
                        "id": "10000",
                        "key": "ADEPT",
                        "name": "Adept Renamed",
                        "projectTypeKey": "software",
                    }
                ],
                None,
            )

    monkeypatch.setattr(sync_jira_projects, "JiraClient", FakeJiraClient)
    monkeypatch.setattr(
        sync_jira_projects,
        "get_valid_jira_access_token",
        lambda _engine, integration, _settings: ("access-token", integration),
    )
    job = _job(
        "SYNC_JIRA_PROJECTS",
        {
            "workspaceId": str(provider_rows.workspace_id),
            "jiraIntegrationId": str(provider_rows.jira_integration_id),
        },
    )
    sync_jira_projects.handle_sync_jira_projects(database_engine, job, job.locked_by)
    sync_jira_projects.handle_sync_jira_projects(database_engine, job, job.locked_by)

    with database_engine.connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                SELECT project_name, tracking_enabled
                FROM jira_projects WHERE id = :id
                """
                ),
                {"id": provider_rows.jira_project_id},
            )
            .mappings()
            .one()
        )
        count = connection.execute(
            text("SELECT count(*) FROM jira_projects WHERE jira_integration_id = :id"),
            {"id": provider_rows.jira_integration_id},
        ).scalar_one()
    assert row["project_name"] == "Adept Renamed"
    assert row["tracking_enabled"] is True
    assert count == 1


@pytest.mark.integration
def test_concurrent_jira_jobs_rotate_one_refresh_token_once(
    database_engine: Engine,
    provider_rows: ProviderRows,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        postgres_password=SecretStr("test"),
        app_integration_encryption_active_key_version=1,
        app_integration_encryption_key_v1_base64=SecretStr(
            base64.b64encode(os.urandom(32)).decode("ascii")
        ),
    )
    access_enc, key_version = encrypt_integration_secret("old-access", settings)
    refresh_enc, _ = encrypt_integration_secret("old-refresh", settings)
    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE jira_integrations
                SET access_token_enc = :access_token_enc,
                    refresh_token_enc = :refresh_token_enc,
                    encryption_key_version = :key_version,
                    access_token_expires_at = now() - interval '1 hour'
                WHERE id = :integration_id
                """
            ),
            {
                "access_token_enc": access_enc,
                "refresh_token_enc": refresh_enc,
                "key_version": key_version,
                "integration_id": provider_rows.jira_integration_id,
            },
        )

    stale = provider_support.load_jira_integration(
        database_engine,
        provider_rows.jira_integration_id,
        provider_rows.workspace_id,
    )
    refresh_calls: list[str] = []

    def fake_refresh(_settings: Settings, refresh_token: str) -> JiraOAuthTokens:
        refresh_calls.append(refresh_token)
        time.sleep(0.15)
        return JiraOAuthTokens(
            access_token="new-access",
            refresh_token="new-refresh",
            expires_in_seconds=3600,
            scopes=["read:jira-work", "manage:jira-webhook", "offline_access"],
        )

    monkeypatch.setattr(provider_support, "refresh_oauth_token", fake_refresh)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _index: provider_support.get_valid_jira_access_token(
                    database_engine, stale, settings
                ),
                range(2),
            )
        )

    assert refresh_calls == ["old-refresh"]
    assert [token for token, _integration in results] == ["new-access", "new-access"]


@pytest.mark.integration
def test_jira_webhook_renewal_refreshes_and_requeues_five_days_before_expiry(
    database_engine: Engine,
    provider_rows: ProviderRows,
    job_factory: JobFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expiration = datetime.now(UTC).replace(microsecond=0) + timedelta(days=20)

    class FakeJiraClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeJiraClient:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def refresh_webhook(self, webhook_id: int) -> str:
            assert webhook_id == 991
            return expiration.isoformat()

        def webhook_exists(self, webhook_id: int) -> bool:
            assert webhook_id == 991
            return True

    monkeypatch.setattr(renew_jira_webhook, "JiraClient", FakeJiraClient)
    monkeypatch.setattr(
        renew_jira_webhook,
        "load_jira_integration",
        lambda engine, integration_id, workspace_id=None: (
            loaded := provider_support.load_jira_integration(engine, integration_id, workspace_id),
            replace(loaded, webhook_token_hash="a" * 64)
            if not loaded.webhook_token_hash
            else loaded,
        )[1],
    )
    monkeypatch.setattr(
        renew_jira_webhook,
        "get_valid_jira_access_token",
        lambda _engine, integration, _settings: ("access-token", integration),
    )
    current_id = job_factory.insert(
        job_type="RENEW_JIRA_WEBHOOK",
        payload={
            "workspaceId": str(provider_rows.workspace_id),
            "jiraIntegrationId": str(provider_rows.jira_integration_id),
        },
        attempts=3,
    )
    current = claim_jobs(database_engine, "provider-test-worker", 1)[0]
    dispatch_job(database_engine, current, "provider-test-worker")

    with database_engine.connect() as connection:
        integration_expiration = connection.execute(
            text("SELECT webhook_expires_at FROM jira_integrations WHERE id = :id"),
            {"id": provider_rows.jira_integration_id},
        ).scalar_one()
        scheduled = (
            connection.execute(
                text(
                    """
                    SELECT status, available_at, attempts, locked_at, locked_by
                    FROM processing_jobs
                    WHERE id = :current_id
                    """
                ),
                {"current_id": current_id},
            )
            .mappings()
            .one()
        )
        renewal_count = connection.execute(
            text(
                """
                SELECT count(*)
                FROM processing_jobs
                WHERE job_type = 'RENEW_JIRA_WEBHOOK'
                  AND payload->>'jiraIntegrationId' = :integration_id
                """
            ),
            {"integration_id": str(provider_rows.jira_integration_id)},
        ).scalar_one()
    assert integration_expiration == expiration
    assert scheduled["status"] == "PENDING"
    assert scheduled["attempts"] == 0
    assert scheduled["locked_at"] is None
    assert scheduled["locked_by"] is None
    assert renewal_count == 1
    assert abs((scheduled["available_at"] - (expiration - timedelta(days=5))).total_seconds()) < 2


@pytest.mark.integration
def test_jira_webhook_renewal_marks_missing_remote_webhook_for_reconnect(
    database_engine: Engine,
    provider_rows: ProviderRows,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeJiraClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeJiraClient:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def webhook_exists(self, webhook_id: int) -> bool:
            assert webhook_id == 991
            return False

        def refresh_webhook(self, _webhook_id: int) -> str:
            raise AssertionError("a missing webhook must not be refreshed")

    monkeypatch.setattr(renew_jira_webhook, "JiraClient", FakeJiraClient)
    monkeypatch.setattr(
        renew_jira_webhook,
        "load_jira_integration",
        lambda engine, integration_id, workspace_id=None: (
            loaded := provider_support.load_jira_integration(engine, integration_id, workspace_id),
            replace(loaded, webhook_token_hash="a" * 64)
            if not loaded.webhook_token_hash
            else loaded,
        )[1],
    )
    monkeypatch.setattr(
        renew_jira_webhook,
        "get_valid_jira_access_token",
        lambda _engine, integration, _settings: ("access-token", integration),
    )

    with pytest.raises(PermanentJobError, match="reconnect Jira"):
        renew_jira_webhook.handle_renew_jira_webhook(
            database_engine,
            _job(
                "RENEW_JIRA_WEBHOOK",
                {
                    "workspaceId": str(provider_rows.workspace_id),
                    "jiraIntegrationId": str(provider_rows.jira_integration_id),
                },
            ),
            "provider-test-worker",
        )

    with database_engine.connect() as connection:
        status = connection.execute(
            text("SELECT status FROM jira_integrations WHERE id = :id"),
            {"id": provider_rows.jira_integration_id},
        ).scalar_one()
    assert status == "ERROR"


@pytest.mark.integration
def test_stale_jira_renewal_is_reclaimed_without_a_successor_collision(
    database_engine: Engine,
    provider_rows: ProviderRows,
    job_factory: JobFactory,
) -> None:
    renewal_id = job_factory.insert(
        job_type="RENEW_JIRA_WEBHOOK",
        payload={
            "workspaceId": str(provider_rows.workspace_id),
            "jiraIntegrationId": str(provider_rows.jira_integration_id),
        },
        status="RUNNING",
        attempts=1,
        locked_by="crashed-provider-worker",
        locked_offset_seconds=-3600,
    )

    claimed = claim_jobs(
        database_engine,
        "replacement-provider-worker",
        1,
        stale_after_seconds=30,
    )

    assert [job.id for job in claimed] == [renewal_id]
    row = job_factory.row(renewal_id)
    assert row["status"] == "RUNNING"
    assert row["locked_by"] == "replacement-provider-worker"


@pytest.mark.integration
def test_jira_issue_delivery_is_idempotent_and_updates_raw_lifecycle(
    database_engine: Engine,
    provider_rows: ProviderRows,
) -> None:
    payload = {
        "issue": {
            "id": "20001",
            "key": "ADEPT-1",
            "fields": {
                "project": {"id": "10000"},
                "issuetype": {"name": "Bug"},
                "status": {"name": "In Progress"},
                "priority": {"name": "High"},
                "summary": "Provider retry issue",
                "created": "2026-08-20T10:00:00.000+0000",
                "updated": "2026-08-20T11:00:00.000+0000",
            },
        }
    }
    raw_event_id = _raw_event(
        database_engine,
        provider_rows,
        source="JIRA",
        event_type="jira:issue_updated",
        action=None,
        payload=payload,
    )
    job = _job(
        "PROCESS_JIRA_EVENT",
        {
            "rawEventId": str(raw_event_id),
            "jiraIntegrationId": str(provider_rows.jira_integration_id),
        },
    )
    jira_event.handle_process_jira_event(database_engine, job, job.locked_by)
    jira_event.handle_process_jira_event(database_engine, job, job.locked_by)

    with database_engine.connect() as connection:
        issue = (
            connection.execute(
                text(
                    """
                SELECT issue_key, summary, version
                FROM jira_issues
                WHERE jira_project_id = :project_id AND jira_issue_id = '20001'
                """
                ),
                {"project_id": provider_rows.jira_project_id},
            )
            .mappings()
            .one()
        )
        raw = (
            connection.execute(
                text(
                    """
                SELECT status, attempt_count, processed_at, last_error
                FROM raw_webhook_events WHERE id = :id
                """
                ),
                {"id": raw_event_id},
            )
            .mappings()
            .one()
        )
    assert issue["issue_key"] == "ADEPT-1"
    assert issue["summary"] == "Provider retry issue"
    assert issue["version"] == 2
    assert raw["status"] == "PROCESSED"
    assert raw["attempt_count"] == 2
    assert raw["processed_at"] is not None
    assert raw["last_error"] is None


@pytest.mark.integration
def test_webhook_data_events_ignore_untracked_catalog_entries(
    database_engine: Engine,
    provider_rows: ProviderRows,
) -> None:
    with database_engine.begin() as connection:
        connection.execute(
            text("UPDATE repositories SET tracking_enabled = false WHERE id = :id"),
            {"id": provider_rows.repository_id},
        )
        connection.execute(
            text("UPDATE jira_projects SET tracking_enabled = false WHERE id = :id"),
            {"id": provider_rows.jira_project_id},
        )

    github_raw_id = _raw_event(
        database_engine,
        provider_rows,
        source="GITHUB",
        event_type="pull_request",
        action="opened",
        repository_id=provider_rows.repository_id,
        payload={"pull_request": {"id": 1, "number": 1}},
    )
    github_event.handle_process_github_event(
        database_engine,
        _job("PROCESS_GITHUB_EVENT", {"rawEventId": str(github_raw_id)}),
        "provider-test-worker",
    )

    jira_raw_id = _raw_event(
        database_engine,
        provider_rows,
        source="JIRA",
        event_type="jira:issue_updated",
        action=None,
        payload={
            "issue": {
                "id": "20002",
                "key": "ADEPT-2",
                "fields": {"project": {"id": "10000"}},
            }
        },
    )
    jira_job = _job(
        "PROCESS_JIRA_EVENT",
        {
            "rawEventId": str(jira_raw_id),
            "jiraIntegrationId": str(provider_rows.jira_integration_id),
        },
    )
    jira_event.handle_process_jira_event(database_engine, jira_job, jira_job.locked_by)

    with database_engine.connect() as connection:
        github_status = connection.execute(
            text("SELECT status FROM raw_webhook_events WHERE id = :id"),
            {"id": github_raw_id},
        ).scalar_one()
        counts = connection.execute(
            text(
                """
                SELECT
                    (SELECT count(*) FROM pull_requests WHERE repository_id = :repository_id),
                    (SELECT count(*) FROM jira_issues WHERE jira_project_id = :jira_project_id)
                """
            ),
            {
                "repository_id": provider_rows.repository_id,
                "jira_project_id": provider_rows.jira_project_id,
            },
        ).one()
    assert github_status == "IGNORED"
    assert tuple(counts) == (0, 0)
