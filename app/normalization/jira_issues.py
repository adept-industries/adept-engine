import json
import structlog
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, text

logger = structlog.get_logger()

def parse_jira_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        # Jira format is typically: 2026-08-18T10:00:00.000+0000
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f%z")
    except ValueError:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None


def upsert_jira_issue(
    database_engine: Engine,
    workspace_id: UUID,
    jira_project_id: UUID,
    issue_payload: dict[str, Any],
) -> None:
    """
    Idempotent upsert of a Jira issue.
    """
    issue_id_str = str(issue_payload.get("id"))
    issue_key = issue_payload.get("key")
    
    if not issue_id_str or not issue_key:
        logger.warning(
            "jira_issue_missing_identifiers",
            workspace_id=str(workspace_id),
            jira_project_id=str(jira_project_id),
        )
        return

    fields = issue_payload.get("fields", {})
    
    issue_type = fields.get("issuetype", {}).get("name")
    status_name = fields.get("status", {}).get("name")
    priority_name = fields.get("priority", {}).get("name")
    summary = fields.get("summary", "")
    
    # We will let the project mapping rules determine if it's an incident later,
    # but for now we default to false.
    is_incident = False
    
    jira_created_at = parse_jira_timestamp(fields.get("created"))
    jira_updated_at = parse_jira_timestamp(fields.get("updated"))
    resolved_at = parse_jira_timestamp(fields.get("resolutiondate"))
    
    raw_data_json = json.dumps(issue_payload)

    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO jira_issues (
                    workspace_id,
                    jira_project_id,
                    jira_issue_id,
                    issue_key,
                    issue_type,
                    status_name,
                    priority_name,
                    summary,
                    is_incident,
                    jira_created_at,
                    jira_updated_at,
                    resolved_at,
                    raw_data,
                    created_at,
                    updated_at,
                    version
                )
                VALUES (
                    :workspace_id,
                    :jira_project_id,
                    :jira_issue_id,
                    :issue_key,
                    :issue_type,
                    :status_name,
                    :priority_name,
                    :summary,
                    :is_incident,
                    :jira_created_at,
                    :jira_updated_at,
                    :resolved_at,
                    cast(:raw_data as jsonb),
                    now(),
                    now(),
                    1
                )
                ON CONFLICT (jira_project_id, jira_issue_id) DO UPDATE SET
                    issue_key = EXCLUDED.issue_key,
                    issue_type = EXCLUDED.issue_type,
                    status_name = EXCLUDED.status_name,
                    priority_name = EXCLUDED.priority_name,
                    summary = EXCLUDED.summary,
                    jira_created_at = COALESCE(EXCLUDED.jira_created_at, jira_issues.jira_created_at),
                    jira_updated_at = EXCLUDED.jira_updated_at,
                    resolved_at = EXCLUDED.resolved_at,
                    raw_data = EXCLUDED.raw_data,
                    updated_at = now(),
                    version = jira_issues.version + 1
                """
            ),
            {
                "workspace_id": workspace_id,
                "jira_project_id": jira_project_id,
                "jira_issue_id": issue_id_str,
                "issue_key": issue_key,
                "issue_type": issue_type,
                "status_name": status_name,
                "priority_name": priority_name,
                "summary": summary,
                "is_incident": is_incident,
                "jira_created_at": jira_created_at,
                "jira_updated_at": jira_updated_at,
                "resolved_at": resolved_at,
                "raw_data": raw_data_json,
            },
        )
        
        logger.info(
            "jira_issue_upserted",
            workspace_id=str(workspace_id),
            jira_project_id=str(jira_project_id),
            issue_key=issue_key,
        )
