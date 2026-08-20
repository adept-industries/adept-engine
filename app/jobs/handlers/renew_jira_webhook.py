import structlog
from sqlalchemy import Engine, text

from app.db.models import ClaimedJob
from app.jobs.retry import PermanentJobError

logger = structlog.get_logger()


def handle_renew_jira_webhook(database_engine: Engine, job: ClaimedJob, worker_id: str) -> None:
    """
    Stub for Jira webhook renewal.
    """
    integration_id = job.payload.get("jiraIntegrationId")

    if not integration_id:
        raise PermanentJobError("Missing jiraIntegrationId")

    logger.info("renew_jira_webhook_start", job_id=str(job.id), integration_id=integration_id)

    # In a real implementation:
    # 1. client.renew_webhook()
    # 2. Update webhook_expires_at in DB

    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE jira_integrations
                SET webhook_expires_at = now() + interval '30 days',
                    updated_at = now(),
                    version = version + 1
                WHERE id = :integration_id
                """
            ),
            {"integration_id": integration_id},
        )

    logger.info("renew_jira_webhook_done", job_id=str(job.id))
