from enum import StrEnum

import structlog
from sqlalchemy import Engine

from app.db.models import ClaimedJob
from app.jobs.handlers.backfill_repository import handle_backfill_repository
from app.jobs.handlers.delete_workspace import handle_delete_workspace
from app.jobs.handlers.evaluate_alerts import handle_evaluate_alerts
from app.jobs.handlers.github_event import handle_process_github_event
from app.jobs.handlers.jira_event import handle_process_jira_event
from app.jobs.handlers.recalculate_metrics import handle_recalculate_metrics
from app.jobs.handlers.renew_jira_webhook import handle_renew_jira_webhook
from app.jobs.handlers.sync_github_repositories import handle_sync_github_repositories
from app.jobs.handlers.sync_jira_projects import handle_sync_jira_projects
from app.jobs.retry import PermanentJobError, RequeueWithPayloadError, mark_failed, mark_succeeded
from app.providers import ProviderError

logger = structlog.get_logger()


HANDLERS = {
    "PROCESS_GITHUB_EVENT": handle_process_github_event,
    "PROCESS_JIRA_EVENT": handle_process_jira_event,
    "BACKFILL_REPOSITORY": handle_backfill_repository,
    "SYNC_GITHUB_REPOSITORIES": handle_sync_github_repositories,
    "SYNC_JIRA_PROJECTS": handle_sync_jira_projects,
    "RENEW_JIRA_WEBHOOK": handle_renew_jira_webhook,
    "DELETE_WORKSPACE": handle_delete_workspace,
    "RECALCULATE_METRICS": handle_recalculate_metrics,
    "EVALUATE_ALERTS": handle_evaluate_alerts,
}


class JobDispatchOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    RETRY_SCHEDULED = "retry_scheduled"
    DEAD_LETTERED = "dead_lettered"
    CONTINUATION_REQUEUED = "continuation_requeued"


def _failure_outcome(status: str) -> JobDispatchOutcome:
    if status == "FAILED":
        return JobDispatchOutcome.RETRY_SCHEDULED
    if status == "DEAD":
        return JobDispatchOutcome.DEAD_LETTERED
    raise RuntimeError(f"unexpected durable job failure status: {status}")


def dispatch_job(database_engine: Engine, job: ClaimedJob, worker_id: str) -> JobDispatchOutcome:
    handler = HANDLERS.get(job.job_type)
    if handler is None:
        # Unknown job types are a permanent configuration/data error; marking DEAD
        # avoids filling the retry queue with jobs that will never succeed.
        logger.warning(
            "unsupported_job_type_marked_dead", job_type=job.job_type, job_id=str(job.id)
        )
        return _failure_outcome(
            mark_failed(
                database_engine,
                job.id,
                worker_id,
                f"UNSUPPORTED_JOB_TYPE: {job.job_type}",
                permanent=True,
            )
        )

    try:
        # Handlers own business work only. They return on success, raise
        # PermanentJobError for non-retryable input, or use the explicit requeue
        # signal for paginated work. The dispatcher alone writes terminal states.
        handler(database_engine, job, worker_id)
    except RequeueWithPayloadError:
        logger.info("job_requeued", job_id=str(job.id), job_type=job.job_type)
        return JobDispatchOutcome.CONTINUATION_REQUEUED
    except PermanentJobError as exc:
        logger.warning(
            "job_permanent_failure",
            job_id=str(job.id),
            job_type=job.job_type,
            error=str(exc),
        )
        return _failure_outcome(
            mark_failed(database_engine, job.id, worker_id, str(exc), permanent=True)
        )
    except ProviderError as exc:
        logger.error(
            "provider_job_execution_failed",
            job_id=str(job.id),
            job_type=job.job_type,
            error=str(exc),
            retry_after_seconds=exc.retry_after_seconds,
        )
        return _failure_outcome(
            mark_failed(
                database_engine,
                job.id,
                worker_id,
                str(exc),
                retry_after_seconds=exc.retry_after_seconds,
            )
        )
    except Exception as exc:
        logger.error(
            "job_execution_failed",
            job_id=str(job.id),
            job_type=job.job_type,
            error=str(exc),
        )
        return _failure_outcome(mark_failed(database_engine, job.id, worker_id, str(exc)))

    # Keep this outside the handler try/except. If ownership was lost while the
    # handler ran, mark_succeeded must surface that operational error instead of
    # incorrectly attempting a second state transition through mark_failed.
    mark_succeeded(database_engine, job.id, worker_id)
    logger.info("job_completed_successfully", job_id=str(job.id), job_type=job.job_type)
    return JobDispatchOutcome.SUCCEEDED
