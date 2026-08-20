from collections.abc import Iterator
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, text

from app.jobs.claimer import claim_jobs
from app.jobs.dispatcher import dispatch_job
from tests.conftest import JobFactory

pytestmark = pytest.mark.integration


class WorkspaceFactory:
    def __init__(self, database_engine: Engine) -> None:
        self.database_engine = database_engine
        self.user_ids: list[UUID] = []
        self.workspace_ids: list[UUID] = []

    def insert_user(self) -> UUID:
        user_id = uuid4()
        with self.database_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO users (id, email, password_hash, display_name, email_verified_at)
                    VALUES (:id, :email, 'test-password-hash', 'Engine Test User', now())
                    """
                ),
                {"id": user_id, "email": f"{user_id}@engine.test"},
            )
            connection.execute(
                text(
                    """
                    INSERT INTO refresh_tokens (
                        id, user_id, family_id, token_hash, expires_at
                    ) VALUES (
                        :id, :user_id, :family_id, :token_hash, now() + interval '1 day'
                    )
                    """
                ),
                {
                    "id": uuid4(),
                    "user_id": user_id,
                    "family_id": uuid4(),
                    "token_hash": f"engine-test-{uuid4()}",
                },
            )
        self.user_ids.append(user_id)
        return user_id

    def insert_workspace(
        self,
        user_id: UUID,
        *,
        status: str,
        with_tenant_data: bool = False,
    ) -> dict[str, UUID]:
        workspace_id = uuid4()
        membership_id = uuid4()
        identifiers = {
            "workspace_id": workspace_id,
            "membership_id": membership_id,
        }

        with self.database_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO workspaces (id, name, slug, timezone, status)
                    VALUES (:id, 'Engine Test Workspace', :slug, 'UTC', :status)
                    """
                ),
                {
                    "id": workspace_id,
                    "slug": f"engine-test-{workspace_id}",
                    "status": status,
                },
            )
            connection.execute(
                text(
                    """
                    INSERT INTO memberships (id, workspace_id, user_id, role, status)
                    VALUES (:id, :workspace_id, :user_id, 'MANAGER', 'ACTIVE')
                    """
                ),
                {
                    "id": membership_id,
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                },
            )

            if with_tenant_data:
                identifiers.update(
                    self._insert_tenant_data(connection, workspace_id, membership_id, user_id)
                )

        self.workspace_ids.append(workspace_id)
        return identifiers

    def _insert_tenant_data(
        self,
        connection: Any,
        workspace_id: UUID,
        membership_id: UUID,
        user_id: UUID,
    ) -> dict[str, UUID]:
        github_integration_id = uuid4()
        repository_id = uuid4()
        project_id = uuid4()
        jira_integration_id = uuid4()
        jira_project_id = uuid4()
        raw_event_id = uuid4()
        workspace_job_id = uuid4()

        connection.execute(
            text(
                """
                INSERT INTO github_integrations (
                    id, workspace_id, installation_id, account_external_id,
                    account_login, account_type, repository_selection, status,
                    installed_by_membership_id
                ) VALUES (
                    :id, :workspace_id, :installation_id, :account_external_id,
                    'engine-test', 'ORGANIZATION', 'ALL', 'SUSPENDED', :membership_id
                )
                """
            ),
            {
                "id": github_integration_id,
                "workspace_id": workspace_id,
                "installation_id": uuid4().int % 9_000_000_000 + 1,
                "account_external_id": uuid4().int % 9_000_000_000 + 1,
                "membership_id": membership_id,
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO repositories (
                    id, workspace_id, github_integration_id, github_repo_id,
                    owner_login, name, full_name, tracking_enabled
                ) VALUES (
                    :id, :workspace_id, :integration_id, :github_repo_id,
                    'engine-test', 'repository', 'engine-test/repository', true
                )
                """
            ),
            {
                "id": repository_id,
                "workspace_id": workspace_id,
                "integration_id": github_integration_id,
                "github_repo_id": uuid4().int % 9_000_000_000 + 1,
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO projects (
                    id, workspace_id, name, created_by_membership_id
                ) VALUES (:id, :workspace_id, 'Engine Test Project', :membership_id)
                """
            ),
            {
                "id": project_id,
                "workspace_id": workspace_id,
                "membership_id": membership_id,
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO project_repositories (project_id, repository_id, workspace_id)
                VALUES (:project_id, :repository_id, :workspace_id)
                """
            ),
            {
                "project_id": project_id,
                "repository_id": repository_id,
                "workspace_id": workspace_id,
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO jira_integrations (
                    id, workspace_id, cloud_id, site_url, display_name,
                    access_token_enc, refresh_token_enc, encryption_key_version,
                    access_token_expires_at, status, connected_by_membership_id
                ) VALUES (
                    :id, :workspace_id, :cloud_id, 'https://engine-test.atlassian.net',
                    'Engine Test Jira', 'encrypted-access', 'encrypted-refresh', 1,
                    now() + interval '1 hour', 'SUSPENDED', :membership_id
                )
                """
            ),
            {
                "id": jira_integration_id,
                "workspace_id": workspace_id,
                "cloud_id": str(uuid4()),
                "membership_id": membership_id,
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO jira_projects (
                    id, workspace_id, jira_integration_id, jira_project_id,
                    project_key, project_name, tracking_enabled
                ) VALUES (
                    :id, :workspace_id, :integration_id, :external_id,
                    :project_key, 'Engine Test Jira Project', true
                )
                """
            ),
            {
                "id": jira_project_id,
                "workspace_id": workspace_id,
                "integration_id": jira_integration_id,
                "external_id": str(uuid4()),
                "project_key": f"E{str(jira_project_id)[:7]}".upper(),
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO repository_jira_projects (repository_id, jira_project_id)
                VALUES (:repository_id, :jira_project_id)
                """
            ),
            {"repository_id": repository_id, "jira_project_id": jira_project_id},
        )
        connection.execute(
            text(
                """
                INSERT INTO raw_webhook_events (
                    id, workspace_id, repository_id, source, delivery_id,
                    event_type, payload, signature_valid
                ) VALUES (
                    :id, :workspace_id, :repository_id, 'GITHUB', :delivery_id,
                    'push', '{}'::jsonb, true
                )
                """
            ),
            {
                "id": raw_event_id,
                "workspace_id": workspace_id,
                "repository_id": repository_id,
                "delivery_id": str(uuid4()),
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO processing_jobs (
                    id, workspace_id, repository_id, raw_event_id,
                    job_type, payload, status
                ) VALUES (
                    :id, :workspace_id, :repository_id, :raw_event_id,
                    'PROCESS_GITHUB_EVENT', '{}'::jsonb, 'SUCCEEDED'
                )
                """
            ),
            {
                "id": workspace_job_id,
                "workspace_id": workspace_id,
                "repository_id": repository_id,
                "raw_event_id": raw_event_id,
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO audit_logs (
                    id, workspace_id, actor_user_id, actor_membership_id,
                    action, entity_type, entity_id
                ) VALUES (
                    :id, :workspace_id, :user_id, :membership_id,
                    'WORKSPACE_DELETION_REQUESTED', 'WORKSPACE', :workspace_id
                )
                """
            ),
            {
                "id": uuid4(),
                "workspace_id": workspace_id,
                "user_id": user_id,
                "membership_id": membership_id,
            },
        )

        return {
            "github_integration_id": github_integration_id,
            "repository_id": repository_id,
            "project_id": project_id,
            "jira_integration_id": jira_integration_id,
            "jira_project_id": jira_project_id,
            "raw_event_id": raw_event_id,
            "workspace_job_id": workspace_job_id,
        }

    def cleanup(self) -> None:
        with self.database_engine.begin() as connection:
            for workspace_id in self.workspace_ids:
                connection.execute(
                    text("DELETE FROM workspaces WHERE id = :id"),
                    {"id": workspace_id},
                )
            for user_id in self.user_ids:
                connection.execute(
                    text("DELETE FROM users WHERE id = :id"),
                    {"id": user_id},
                )


@pytest.fixture
def workspace_factory(database_engine: Engine) -> Iterator[WorkspaceFactory]:
    factory = WorkspaceFactory(database_engine)
    yield factory
    factory.cleanup()


def _claim_and_dispatch(database_engine: Engine, job_id: UUID, worker_id: str) -> None:
    jobs = claim_jobs(database_engine, worker_id, 1)
    assert [job.id for job in jobs] == [job_id]
    dispatch_job(database_engine, jobs[0], worker_id)


def test_delete_workspace_removes_only_target_tenant_data(
    database_engine: Engine,
    job_factory: JobFactory,
    workspace_factory: WorkspaceFactory,
) -> None:
    user_id = workspace_factory.insert_user()
    target = workspace_factory.insert_workspace(
        user_id,
        status="DELETING",
        with_tenant_data=True,
    )
    retained = workspace_factory.insert_workspace(
        user_id,
        status="ACTIVE",
        with_tenant_data=True,
    )
    job_id = job_factory.insert(
        job_type="DELETE_WORKSPACE",
        payload={"workspaceId": str(target["workspace_id"])},
        priority=-1000,
    )

    _claim_and_dispatch(database_engine, job_id, "delete-workspace-test")

    with database_engine.connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                    SELECT
                        (SELECT count(*) FROM workspaces WHERE id = :target_id)
                            AS target_workspaces,
                        (SELECT count(*) FROM memberships WHERE workspace_id = :target_id)
                            AS target_memberships,
                        (SELECT count(*) FROM github_integrations WHERE workspace_id = :target_id)
                            AS target_github_integrations,
                        (SELECT count(*) FROM repositories WHERE workspace_id = :target_id)
                            AS target_repositories,
                        (SELECT count(*) FROM projects WHERE workspace_id = :target_id)
                            AS target_projects,
                        (SELECT count(*) FROM project_repositories WHERE workspace_id = :target_id)
                            AS target_project_repositories,
                        (SELECT count(*) FROM jira_integrations WHERE workspace_id = :target_id)
                            AS target_jira_integrations,
                        (SELECT count(*) FROM raw_webhook_events WHERE workspace_id = :target_id)
                            AS target_raw_events,
                        (SELECT count(*) FROM processing_jobs WHERE workspace_id = :target_id)
                            AS target_jobs,
                        (SELECT count(*) FROM audit_logs WHERE workspace_id = :target_id)
                            AS target_audit_logs,
                        (SELECT count(*) FROM users WHERE id = :user_id) AS users,
                        (SELECT count(*) FROM refresh_tokens WHERE user_id = :user_id)
                            AS refresh_tokens,
                        (SELECT count(*) FROM workspaces WHERE id = :retained_id)
                            AS retained_workspaces,
                        (SELECT count(*) FROM memberships WHERE workspace_id = :retained_id)
                            AS retained_memberships,
                        (SELECT count(*) FROM github_integrations WHERE workspace_id = :retained_id)
                            AS retained_github_integrations,
                        (SELECT count(*) FROM repositories WHERE workspace_id = :retained_id)
                            AS retained_repositories,
                        (SELECT count(*) FROM projects WHERE workspace_id = :retained_id)
                            AS retained_projects,
                        (SELECT count(*) FROM project_repositories
                         WHERE workspace_id = :retained_id)
                            AS retained_project_repositories,
                        (SELECT count(*) FROM jira_integrations WHERE workspace_id = :retained_id)
                            AS retained_jira_integrations,
                        (SELECT count(*) FROM raw_webhook_events WHERE workspace_id = :retained_id)
                            AS retained_raw_events,
                        (SELECT count(*) FROM processing_jobs WHERE workspace_id = :retained_id)
                            AS retained_jobs,
                        (SELECT count(*) FROM audit_logs WHERE workspace_id = :retained_id)
                            AS retained_audit_logs
                    """
                ),
                {
                    "target_id": target["workspace_id"],
                    "user_id": user_id,
                    "retained_id": retained["workspace_id"],
                },
            )
            .mappings()
            .one()
        )

    assert all(row[column] == 0 for column in row if column.startswith("target_"))
    assert row["users"] == 1
    assert row["refresh_tokens"] == 1
    assert all(row[column] == 1 for column in row if column.startswith("retained_"))
    assert job_factory.row(job_id)["status"] == "SUCCEEDED"


def test_delete_workspace_is_idempotent_when_workspace_is_already_absent(
    database_engine: Engine,
    job_factory: JobFactory,
) -> None:
    job_id = job_factory.insert(
        job_type="DELETE_WORKSPACE",
        payload={"workspaceId": str(uuid4())},
        priority=-1000,
    )

    _claim_and_dispatch(database_engine, job_id, "delete-missing-workspace-test")

    assert job_factory.row(job_id)["status"] == "SUCCEEDED"


def test_delete_workspace_refuses_active_workspace(
    database_engine: Engine,
    job_factory: JobFactory,
    workspace_factory: WorkspaceFactory,
) -> None:
    user_id = workspace_factory.insert_user()
    workspace = workspace_factory.insert_workspace(user_id, status="ACTIVE")
    job_id = job_factory.insert(
        job_type="DELETE_WORKSPACE",
        payload={"workspaceId": str(workspace["workspace_id"])},
        priority=-1000,
    )

    _claim_and_dispatch(database_engine, job_id, "delete-active-workspace-test")

    assert job_factory.row(job_id)["status"] == "DEAD"
    with database_engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT count(*) FROM workspaces WHERE id = :id"),
                {"id": workspace["workspace_id"]},
            ).scalar_one()
            == 1
        )


@pytest.mark.parametrize("payload", [{}, {"workspaceId": "not-a-uuid"}])
def test_delete_workspace_rejects_invalid_payload(
    database_engine: Engine,
    job_factory: JobFactory,
    payload: dict[str, str],
) -> None:
    job_id = job_factory.insert(
        job_type="DELETE_WORKSPACE",
        payload=payload,
        priority=-1000,
    )

    _claim_and_dispatch(database_engine, job_id, f"delete-invalid-payload-{job_id}")

    assert job_factory.row(job_id)["status"] == "DEAD"


def test_delete_workspace_rejects_a_tenant_scoped_deletion_job(
    database_engine: Engine,
    job_factory: JobFactory,
    workspace_factory: WorkspaceFactory,
) -> None:
    user_id = workspace_factory.insert_user()
    workspace = workspace_factory.insert_workspace(user_id, status="DELETING")
    job_id = job_factory.insert(
        job_type="DELETE_WORKSPACE",
        payload={"workspaceId": str(workspace["workspace_id"])},
        priority=-1000,
    )
    with database_engine.begin() as connection:
        connection.execute(
            text("UPDATE processing_jobs SET workspace_id = :workspace_id WHERE id = :id"),
            {"workspace_id": workspace["workspace_id"], "id": job_id},
        )

    _claim_and_dispatch(database_engine, job_id, "delete-scoped-job-test")

    assert job_factory.row(job_id)["status"] == "DEAD"
    with database_engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT count(*) FROM workspaces WHERE id = :id"),
                {"id": workspace["workspace_id"]},
            ).scalar_one()
            == 1
        )
