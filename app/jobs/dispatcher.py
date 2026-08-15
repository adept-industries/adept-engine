import structlog
from sqlalchemy import Engine

from app.db.models import ClaimedJob
from app.jobs.retry import mark_failed, mark_succeeded

logger = structlog.get_logger()


def handle_backfill_repository(job: ClaimedJob, database_engine: Engine) -> None:
    repository_id = job.payload.get("repositoryId")
    backfill_days = job.payload.get("backfillDays", 90)
    logger.info(
        "handling_backfill_repository",
        job_id=str(job.id),
        repository_id=repository_id,
        backfill_days=backfill_days,
    )
    # Placeholder: repository backfill ingestion pipeline logic


def handle_sync_github_repositories(job: ClaimedJob, database_engine: Engine) -> None:
    integration_id = job.payload.get("integrationId")
    logger.info(
        "handling_sync_github_repositories",
        job_id=str(job.id),
        integration_id=integration_id,
    )


def handle_sync_jira_projects(job: ClaimedJob, database_engine: Engine) -> None:
    integration_id = job.payload.get("integrationId")
    logger.info(
        "handling_sync_jira_projects",
        job_id=str(job.id),
        integration_id=integration_id,
    )


def handle_renew_jira_webhook(job: ClaimedJob, database_engine: Engine) -> None:
    integration_id = job.payload.get("integrationId")
    logger.info(
        "handling_renew_jira_webhook",
        job_id=str(job.id),
        integration_id=integration_id,
    )


HANDLERS = {
    "BACKFILL_REPOSITORY": handle_backfill_repository,
    "SYNC_GITHUB_REPOSITORIES": handle_sync_github_repositories,
    "SYNC_JIRA_PROJECTS": handle_sync_jira_projects,
    "RENEW_JIRA_WEBHOOK": handle_renew_jira_webhook,
}


def dispatch_job(database_engine: Engine, job: ClaimedJob, worker_id: str) -> None:
    handler = HANDLERS.get(job.job_type)
    if handler is None:
        logger.warning("unsupported_job_type", job_type=job.job_type, job_id=str(job.id))
        mark_succeeded(database_engine, job.id, worker_id)
        return

    try:
        handler(job, database_engine)
        mark_succeeded(database_engine, job.id, worker_id)
        logger.info("job_completed_successfully", job_id=str(job.id), job_type=job.job_type)
    except Exception as exc:
        logger.error(
            "job_execution_failed",
            job_id=str(job.id),
            job_type=job.job_type,
            error=str(exc),
        )
        mark_failed(database_engine, job.id, worker_id, str(exc))
