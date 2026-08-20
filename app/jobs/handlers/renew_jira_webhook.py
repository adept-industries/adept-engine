from __future__ import annotations

from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import Engine, text

from app.core.config import get_settings
from app.db.models import ClaimedJob
from app.jobs.handlers.provider_support import (
    get_valid_jira_access_token,
    load_jira_integration,
    mark_jira_integration_error,
    parse_uuid,
    provider_exception_as_job_error,
)
from app.jobs.retry import PermanentJobError, requeue_with_payload
from app.providers.jira import JiraClient

logger = structlog.get_logger()


def handle_renew_jira_webhook(database_engine: Engine, job: ClaimedJob, worker_id: str) -> None:
    integration_id = parse_uuid(job.payload.get("jiraIntegrationId"), "jiraIntegrationId")
    workspace_raw = job.payload.get("workspaceId")
    workspace_id = parse_uuid(workspace_raw, "workspaceId") if workspace_raw else None
    integration = load_jira_integration(database_engine, integration_id, workspace_id)
    webhook_id = integration.webhook_id
    if webhook_id is None or not integration.webhook_token_hash:
        mark_jira_integration_error(database_engine, integration_id)
        raise PermanentJobError(
            "Jira webhook authentication is incomplete; reconnect Jira to create an "
            "authenticated webhook"
        )

    settings = get_settings()
    bound_logger = logger.bind(
        job_id=str(job.id),
        workspace_id=str(integration.workspace_id),
        jira_integration_id=str(integration_id),
        webhook_id=webhook_id,
    )
    bound_logger.info("renew_jira_webhook_started")

    try:
        access_token, integration = get_valid_jira_access_token(
            database_engine, integration, settings
        )
        with JiraClient(settings, integration.cloud_id, access_token) as client:
            if not client.webhook_exists(webhook_id):
                raise PermanentJobError(
                    "Jira no longer lists the registered webhook; reconnect Jira to restore "
                    "authenticated delivery"
                )
            expiration = _parse_jira_timestamp(client.refresh_webhook(webhook_id))
    except Exception as exc:
        converted = provider_exception_as_job_error(exc)
        if isinstance(converted, PermanentJobError) or job.attempts >= job.max_attempts:
            mark_jira_integration_error(database_engine, integration_id)
        if converted is exc:
            raise
        raise converted from exc

    with database_engine.begin() as connection:
        result = connection.execute(
            text(
                """
                UPDATE jira_integrations
                SET webhook_expires_at = :expiration,
                    updated_at = now(),
                    version = version + 1
                WHERE id = :integration_id
                  AND status = 'ACTIVE'
                  AND webhook_id = :webhook_id
                """
            ),
            {
                "expiration": expiration,
                "integration_id": integration_id,
                "webhook_id": webhook_id,
            },
        )
        if result.rowcount != 1:
            raise PermanentJobError("Jira integration changed while its webhook was renewed")

    # A renewal is periodic work, so reuse the currently owned row instead of
    # creating a successor. If the worker crashes after the provider/DB update
    # but before this requeue, stale-lock recovery can safely move this single
    # RUNNING row back to FAILED without colliding with V12's uniqueness guard.
    now = datetime.now(UTC)
    next_renewal_at = max(expiration - timedelta(days=5), now + timedelta(days=1))
    bound_logger.info(
        "renew_jira_webhook_rescheduled",
        webhook_expires_at=expiration.isoformat(),
        next_renewal_at=next_renewal_at.isoformat(),
    )
    requeue_with_payload(
        database_engine,
        job.id,
        worker_id,
        {
            "workspaceId": str(integration.workspace_id),
            "jiraIntegrationId": str(integration_id),
        },
        delay_seconds=(next_renewal_at - now).total_seconds(),
        reset_attempts=True,
    )


def _parse_jira_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PermanentJobError("Jira returned an invalid webhook expiration") from exc
    if parsed.tzinfo is None:
        raise PermanentJobError("Jira returned an invalid webhook expiration")
    return parsed.astimezone(UTC)
