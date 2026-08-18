import structlog
from sqlalchemy import Engine, text

from app.db.models import ClaimedJob
from app.jobs.retry import mark_failed, mark_succeeded
from app.normalization.jira_issues import upsert_jira_issue

logger = structlog.get_logger()


def handle_process_jira_event(database_engine: Engine, job: ClaimedJob, worker_id: str) -> None:
    """
    Handles a Jira webhook event.
    """
    raw_event_id = job.payload.get("rawEventId")
    if not raw_event_id:
        mark_failed(
            database_engine, job.id, worker_id, "Missing rawEventId in payload", permanent=True
        )
        return

    with database_engine.begin() as connection:
        raw_event = (
            connection.execute(
                text("SELECT * FROM raw_webhook_events WHERE id = :id"),
                {"id": raw_event_id},
            )
            .mappings()
            .one_or_none()
        )

    if not raw_event:
        mark_failed(
            database_engine,
            job.id,
            worker_id,
            f"Raw event {raw_event_id} not found",
            permanent=True,
        )
        return

    workspace_id = raw_event["workspace_id"]
    event_type = raw_event["event_type"]
    payload = raw_event["payload"]
    jira_integration_id = job.payload.get("jiraIntegrationId")

    if not workspace_id or not jira_integration_id:
        # Without these, we can't map to a project correctly or save it.
        mark_failed(
            database_engine,
            job.id,
            worker_id,
            "Missing workspace or integration ID",
            permanent=True,
        )
        return

    logger.info(
        "processing_jira_event",
        job_id=str(job.id),
        event_type=event_type,
        raw_event_id=str(raw_event_id),
        workspace_id=str(workspace_id),
    )

    if event_type.startswith("jira:issue_"):
        issue = payload.get("issue")
        if issue:
            # We need to resolve the jira_project_id
            project_id_str = issue.get("fields", {}).get("project", {}).get("id")
            if not project_id_str:
                logger.warning("jira_issue_missing_project", issue_id=issue.get("id"))
                _mark_processed(database_engine, raw_event_id)
                mark_succeeded(database_engine, job.id, worker_id)
                return

            with database_engine.begin() as connection:
                project = (
                    connection.execute(
                        text("""
                        SELECT id FROM jira_projects 
                        WHERE jira_integration_id = :integration_id 
                        AND jira_project_id = :remote_project_id
                    """),
                        {
                            "integration_id": jira_integration_id,
                            "remote_project_id": str(project_id_str),
                        },
                    )
                    .mappings()
                    .one_or_none()
                )

            if project:
                jira_project_id = project["id"]
                upsert_jira_issue(database_engine, workspace_id, jira_project_id, issue)
            else:
                logger.debug("jira_project_not_mapped", remote_project_id=project_id_str)

    else:
        logger.debug("unhandled_jira_event_type", event_type=event_type)

    _mark_processed(database_engine, raw_event_id)
    mark_succeeded(database_engine, job.id, worker_id)


def _mark_processed(database_engine: Engine, raw_event_id: str) -> None:
    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE raw_webhook_events
                SET status = 'PROCESSED',
                    updated_at = now(),
                    version = version + 1
                WHERE id = :id
                """
            ),
            {"id": raw_event_id},
        )
