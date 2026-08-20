from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, text

from app.core.config import Settings, get_settings
from app.jobs.retry import PermanentJobError
from app.providers import ProviderConfigurationError, ProviderPermanentError
from app.providers.crypto import decrypt_integration_secret, encrypt_integration_secret
from app.providers.jira import refresh_oauth_token


@dataclass(frozen=True, slots=True)
class GithubIntegrationContext:
    id: UUID
    workspace_id: UUID
    installation_id: int
    status: str


@dataclass(frozen=True, slots=True)
class GithubRepositoryContext:
    id: UUID
    workspace_id: UUID
    integration_id: UUID
    installation_id: int
    owner_login: str
    name: str
    full_name: str
    default_branch: str
    tracking_enabled: bool
    settings: dict[str, Any]
    integration_status: str


@dataclass(frozen=True, slots=True)
class JiraIntegrationContext:
    id: UUID
    workspace_id: UUID
    cloud_id: str
    access_token_enc: str
    refresh_token_enc: str
    encryption_key_version: int
    access_token_expires_at: datetime
    status: str
    webhook_id: int | None
    webhook_token_hash: str | None
    version: int


def parse_uuid(value: object, field_name: str) -> UUID:
    if value is None or value == "":
        raise PermanentJobError(f"Missing {field_name}")
    try:
        return UUID(str(value))
    except (ValueError, TypeError) as exc:
        raise PermanentJobError(f"Invalid {field_name}") from exc


def load_github_integration(
    database_engine: Engine,
    workspace_id: UUID,
    integration_id: UUID | None = None,
) -> GithubIntegrationContext:
    with database_engine.connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                    SELECT id, workspace_id, installation_id, status
                    FROM github_integrations
                    WHERE workspace_id = :workspace_id
                      AND (:integration_id IS NULL OR id = :integration_id)
                    ORDER BY CASE status WHEN 'ACTIVE' THEN 0 ELSE 1 END, created_at DESC
                    LIMIT 1
                    """
                ),
                {"workspace_id": workspace_id, "integration_id": integration_id},
            )
            .mappings()
            .one_or_none()
        )
    if row is None:
        raise PermanentJobError("GitHub integration not found in the job workspace")
    if row["status"] != "ACTIVE":
        raise PermanentJobError(f"GitHub integration is {row['status']}")
    return GithubIntegrationContext(
        id=UUID(str(row["id"])),
        workspace_id=UUID(str(row["workspace_id"])),
        installation_id=int(row["installation_id"]),
        status=str(row["status"]),
    )


def load_github_repository(database_engine: Engine, repository_id: UUID) -> GithubRepositoryContext:
    with database_engine.connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                    SELECT r.id, r.workspace_id, r.github_integration_id,
                           r.owner_login, r.name, r.full_name, r.default_branch,
                           r.tracking_enabled, r.settings,
                           gi.installation_id, gi.status AS integration_status
                    FROM repositories r
                    JOIN github_integrations gi ON gi.id = r.github_integration_id
                    WHERE r.id = :repository_id
                    """
                ),
                {"repository_id": repository_id},
            )
            .mappings()
            .one_or_none()
        )
    if row is None:
        raise PermanentJobError("Repository not found")
    settings = row["settings"] if isinstance(row["settings"], dict) else {}
    return GithubRepositoryContext(
        id=UUID(str(row["id"])),
        workspace_id=UUID(str(row["workspace_id"])),
        integration_id=UUID(str(row["github_integration_id"])),
        installation_id=int(row["installation_id"]),
        owner_login=str(row["owner_login"]),
        name=str(row["name"]),
        full_name=str(row["full_name"]),
        default_branch=str(row["default_branch"]),
        tracking_enabled=bool(row["tracking_enabled"]),
        settings=settings,
        integration_status=str(row["integration_status"]),
    )


def load_jira_integration(
    database_engine: Engine,
    integration_id: UUID,
    workspace_id: UUID | None = None,
) -> JiraIntegrationContext:
    with database_engine.connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                    SELECT id, workspace_id, cloud_id, access_token_enc,
                           refresh_token_enc, encryption_key_version,
                           access_token_expires_at, status, webhook_id,
                           to_jsonb(ji)->>'webhook_token_hash' AS webhook_token_hash,
                           version
                    FROM jira_integrations ji
                    WHERE id = :integration_id
                      AND (:workspace_id IS NULL OR workspace_id = :workspace_id)
                    """
                ),
                {"integration_id": integration_id, "workspace_id": workspace_id},
            )
            .mappings()
            .one_or_none()
        )
    if row is None:
        raise PermanentJobError("Jira integration not found in the job workspace")
    if row["status"] != "ACTIVE":
        raise PermanentJobError(f"Jira integration is {row['status']}")
    return JiraIntegrationContext(
        id=UUID(str(row["id"])),
        workspace_id=UUID(str(row["workspace_id"])),
        cloud_id=str(row["cloud_id"]),
        access_token_enc=str(row["access_token_enc"]),
        refresh_token_enc=str(row["refresh_token_enc"]),
        encryption_key_version=int(row["encryption_key_version"]),
        access_token_expires_at=row["access_token_expires_at"],
        status=str(row["status"]),
        webhook_id=int(row["webhook_id"]) if row["webhook_id"] is not None else None,
        webhook_token_hash=(
            str(row["webhook_token_hash"]) if row["webhook_token_hash"] is not None else None
        ),
        version=int(row["version"]),
    )


def get_valid_jira_access_token(
    database_engine: Engine,
    integration: JiraIntegrationContext,
    settings: Settings | None = None,
) -> tuple[str, JiraIntegrationContext]:
    active_settings = settings or get_settings()
    if integration.access_token_expires_at > datetime.now(UTC) + timedelta(minutes=5):
        return (
            decrypt_integration_secret(
                integration.access_token_enc,
                integration.encryption_key_version,
                active_settings,
            ),
            integration,
        )

    # Jira rotates refresh tokens. Serialize refresh per integration and re-read
    # the row after acquiring the lock so concurrent jobs never submit the same
    # one-time refresh token to Atlassian.
    with database_engine.begin() as connection:
        row = (
            connection.execute(
                text(
                    """
                    SELECT id, workspace_id, cloud_id, access_token_enc,
                           refresh_token_enc, encryption_key_version,
                           access_token_expires_at, status, webhook_id,
                           to_jsonb(ji)->>'webhook_token_hash' AS webhook_token_hash,
                           version
                    FROM jira_integrations ji
                    WHERE id = :integration_id
                      AND workspace_id = :workspace_id
                    FOR UPDATE
                    """
                ),
                {
                    "integration_id": integration.id,
                    "workspace_id": integration.workspace_id,
                },
            )
            .mappings()
            .one_or_none()
        )
        if row is None or row["status"] != "ACTIVE":
            raise PermanentJobError("Jira integration changed while refreshing its token")
        locked = _jira_context_from_row(row)
        if locked.access_token_expires_at > datetime.now(UTC) + timedelta(minutes=5):
            return (
                decrypt_integration_secret(
                    locked.access_token_enc,
                    locked.encryption_key_version,
                    active_settings,
                ),
                locked,
            )

        current_refresh_token = decrypt_integration_secret(
            locked.refresh_token_enc,
            locked.encryption_key_version,
            active_settings,
        )
        refreshed = refresh_oauth_token(active_settings, current_refresh_token)
        access_enc, key_version = encrypt_integration_secret(
            refreshed.access_token, active_settings
        )
        refresh_enc, refresh_key_version = encrypt_integration_secret(
            refreshed.refresh_token, active_settings
        )
        if refresh_key_version != key_version:
            raise ProviderConfigurationError(
                "Jira tokens were encrypted with different key versions"
            )
        expires_at = datetime.now(UTC) + timedelta(seconds=refreshed.expires_in_seconds)

        connection.execute(
            text(
                """
                UPDATE jira_integrations
                SET access_token_enc = :access_token_enc,
                    refresh_token_enc = :refresh_token_enc,
                    encryption_key_version = :key_version,
                    access_token_expires_at = :expires_at,
                    scopes = CASE
                        WHEN cardinality(CAST(:scopes AS varchar[])) = 0 THEN scopes
                        ELSE CAST(:scopes AS varchar[])
                    END,
                    updated_at = now(),
                    version = version + 1
                WHERE id = :integration_id
                  AND status = 'ACTIVE'
                """
            ),
            {
                "access_token_enc": access_enc,
                "refresh_token_enc": refresh_enc,
                "key_version": key_version,
                "expires_at": expires_at,
                "scopes": refreshed.scopes,
                "integration_id": integration.id,
            },
        )
        updated = replace(
            locked,
            access_token_enc=access_enc,
            refresh_token_enc=refresh_enc,
            encryption_key_version=key_version,
            access_token_expires_at=expires_at,
            version=locked.version + 1,
        )
        return refreshed.access_token, updated


def _jira_context_from_row(row: Any) -> JiraIntegrationContext:
    return JiraIntegrationContext(
        id=UUID(str(row["id"])),
        workspace_id=UUID(str(row["workspace_id"])),
        cloud_id=str(row["cloud_id"]),
        access_token_enc=str(row["access_token_enc"]),
        refresh_token_enc=str(row["refresh_token_enc"]),
        encryption_key_version=int(row["encryption_key_version"]),
        access_token_expires_at=row["access_token_expires_at"],
        status=str(row["status"]),
        webhook_id=int(row["webhook_id"]) if row["webhook_id"] is not None else None,
        webhook_token_hash=(
            str(row["webhook_token_hash"]) if row["webhook_token_hash"] is not None else None
        ),
        version=int(row["version"]),
    )


def mark_jira_integration_error(database_engine: Engine, integration_id: UUID) -> None:
    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE jira_integrations
                SET status = 'ERROR', updated_at = now(), version = version + 1
                WHERE id = :integration_id AND status = 'ACTIVE'
                """
            ),
            {"integration_id": integration_id},
        )


def provider_exception_as_job_error(exc: Exception) -> Exception:
    if isinstance(exc, (ProviderPermanentError, ProviderConfigurationError)):
        return PermanentJobError(str(exc))
    return exc
