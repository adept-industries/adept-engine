import structlog
from sqlalchemy import Engine

from app.db.models import ClaimedJob
from app.jobs.retry import PermanentJobError

logger = structlog.get_logger()


def handle_sync_jira_projects(database_engine: Engine, job: ClaimedJob, worker_id: str) -> None:
    """
    Stub for Jira projects sync.
    """
    workspace_id = job.payload.get("workspaceId")
    integration_id = job.payload.get("jiraIntegrationId")

    if not workspace_id or not integration_id:
        raise PermanentJobError("Missing workspaceId or jiraIntegrationId")

    logger.info("sync_jira_projects_start", job_id=str(job.id))

    # In a real implementation:
    # 1. client.list_projects()
    # 2. upsert batch
    # 3. Handle cursors if paginated, though Jira projects are often small enough for 1-2 pages

    logger.info("sync_jira_projects_done", job_id=str(job.id))
