import structlog
from sqlalchemy import Engine
from app.db.models import ClaimedJob
from app.jobs.retry import mark_failed, mark_succeeded, requeue_with_payload

logger = structlog.get_logger()


def handle_sync_jira_projects(database_engine: Engine, job: ClaimedJob, worker_id: str) -> None:
    """
    Stub for Jira projects sync.
    """
    workspace_id = job.payload.get("workspaceId")
    integration_id = job.payload.get("jiraIntegrationId")

    if not workspace_id or not integration_id:
        mark_failed(
            database_engine,
            job.id,
            worker_id,
            "Missing workspaceId or jiraIntegrationId",
            permanent=True,
        )
        return

    logger.info("sync_jira_projects_start", job_id=str(job.id))

    # In a real implementation:
    # 1. client.list_projects()
    # 2. upsert batch
    # 3. Handle cursors if paginated, though Jira projects are often small enough for 1-2 pages

    logger.info("sync_jira_projects_done", job_id=str(job.id))
    mark_succeeded(database_engine, job.id, worker_id)
