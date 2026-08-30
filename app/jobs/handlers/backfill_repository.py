from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import Engine, text

from app.core.config import get_settings
from app.db.models import ClaimedJob
from app.jobs.handlers.provider_support import (
    load_github_repository,
    parse_uuid,
    provider_exception_as_job_error,
)
from app.jobs.retry import PermanentJobError, requeue_with_payload
from app.metrics.service import recalculate_repository_metrics
from app.normalization.deployments import (
    reclassify_repository_deployments,
    upsert_deployment_from_deployment_status,
    upsert_deployment_from_workflow_run,
)
from app.normalization.github_issues import upsert_github_issue
from app.normalization.pull_requests import upsert_pull_request
from app.providers.github import GithubClient, ProviderPage
from app.risk.service import calculate_and_persist_pull_request_risk

logger = structlog.get_logger()

OPEN_PULL_REQUEST_STAGE = "open_pull_requests"
PULL_REQUEST_STAGE = "pull_requests"
WORKFLOW_RUN_STAGE = "workflow_runs"
DEPLOYMENT_STAGE = "deployments"
ISSUE_STAGE = "issues"


def handle_backfill_repository(database_engine: Engine, job: ClaimedJob, worker_id: str) -> None:
    repository_id = parse_uuid(job.payload.get("repositoryId"), "repositoryId")
    backfill_days = _bounded_days(job.payload.get("backfillDays", 90))
    risk_only = job.payload.get("riskOnly") is True
    issues_only = job.payload.get("issuesOnly") is True
    if risk_only and issues_only:
        raise PermanentJobError("backfill cannot be both risk-only and issues-only")
    started_at = _started_at(job.payload.get("backfillStartedAt"))
    cutoff = started_at - timedelta(days=backfill_days)
    cursor = _cursor(
        job.payload.get("cursor"),
        cutoff,
        started_at,
        initial_stage=ISSUE_STAGE if issues_only else OPEN_PULL_REQUEST_STAGE,
    )
    repository = load_github_repository(database_engine, repository_id)

    if repository.integration_status != "ACTIVE":
        raise PermanentJobError(f"GitHub integration is {repository.integration_status}")
    if repository.archived or not repository.tracking_enabled:
        logger.info(
            "backfill_repository_cancelled_ineligible",
            job_id=str(job.id),
            repository_id=str(repository_id),
            archived=repository.archived,
        )
        return

    stage = cursor["stage"]
    page = cursor["page"]
    bound_logger = logger.bind(
        job_id=str(job.id),
        repository_id=str(repository_id),
        stage=stage,
        page=page,
    )
    bound_logger.info("backfill_repository_page_started")

    try:
        with GithubClient(get_settings(), repository.installation_id) as client:
            next_cursor, item_count = _process_page(
                database_engine,
                client,
                repository,
                stage,
                page,
                cutoff,
                started_at,
                cursor,
                risk_only,
                issues_only,
            )
    except Exception as exc:
        converted = provider_exception_as_job_error(exc)
        if converted is exc:
            raise
        raise converted from exc

    if next_cursor is not None:
        bound_logger.info(
            "backfill_repository_page_completed",
            normalized_count=item_count,
            next_cursor=next_cursor,
        )
        requeue_with_payload(
            database_engine,
            job.id,
            worker_id,
            {
                **job.payload,
                "cursor": next_cursor,
                "backfillStartedAt": started_at.isoformat(),
            },
        )
        return

    if not risk_only and not issues_only:
        reclassify_repository_deployments(database_engine, repository.id)
        recalculate_repository_metrics(
            database_engine,
            repository.workspace_id,
            repository.id,
            from_date=cutoff,
            to_date=started_at + timedelta(days=1),
        )

    bound_logger.info("backfill_repository_completed", normalized_count=item_count)


def _process_page(
    database_engine: Engine,
    client: GithubClient,
    repository: Any,
    stage: str,
    page: int,
    cutoff: datetime,
    started_at: datetime,
    cursor: dict[str, Any],
    risk_only: bool,
    issues_only: bool,
) -> tuple[dict[str, Any] | None, int]:
    if issues_only:
        if stage != ISSUE_STAGE:
            raise PermanentJobError("issues-only backfill cannot process non-issue stages")
        return _process_issue_page(database_engine, client, repository, page, started_at)
    if stage == OPEN_PULL_REQUEST_STAGE:
        return _process_open_pull_request_page(
            database_engine,
            client,
            repository,
            page,
            risk_only=risk_only,
        )
    if risk_only:
        raise PermanentJobError("risk-only backfill cannot process non-risk stages")
    if stage == PULL_REQUEST_STAGE:
        return _process_pull_request_page(
            database_engine, client, repository, page, cutoff, started_at
        )
    if stage == WORKFLOW_RUN_STAGE:
        return _process_workflow_run_page(database_engine, client, repository, cursor)
    if stage == DEPLOYMENT_STAGE:
        return _process_deployment_page(database_engine, client, repository, page, cutoff)
    raise PermanentJobError("Invalid backfill cursor stage")


def _process_open_pull_request_page(
    database_engine: Engine,
    client: GithubClient,
    repository: Any,
    page: int,
    *,
    risk_only: bool,
) -> tuple[dict[str, Any] | None, int]:
    """Normalize and score every currently open PR, regardless of its age."""
    result = client.list_open_pull_requests(repository.owner_login, repository.name, page)
    count = 0
    for summary in result.items:
        number = summary.get("number")
        if not isinstance(number, int):
            raise PermanentJobError("GitHub returned a pull request without a number")
        pull_request = client.get_pull_request(repository.owner_login, repository.name, number)
        commits = client.list_pull_request_commits(
            repository.owner_login,
            repository.name,
            number,
        )
        pull_request_id = upsert_pull_request(
            database_engine,
            repository.workspace_id,
            repository.id,
            pull_request,
            "synchronize",
            commits,
        )
        changed_files = _changed_files(pull_request)
        if changed_files > 3_000:
            logger.warning(
                "backfill_pr_risk_skipped_github_file_cap",
                repository_id=str(repository.id),
                pull_request_number=number,
                changed_files=changed_files,
            )
        else:
            files = client.list_pull_request_files(
                repository.owner_login,
                repository.name,
                number,
            )
            calculate_and_persist_pull_request_risk(
                database_engine,
                repository.workspace_id,
                repository.id,
                pull_request_id,
                pull_request,
                files,
                commits,
            )
        count += 1

    if result.next_page is not None:
        return {"stage": OPEN_PULL_REQUEST_STAGE, "page": result.next_page}, count
    if risk_only:
        return None, count
    return {"stage": PULL_REQUEST_STAGE, "page": 1}, count


def _process_pull_request_page(
    database_engine: Engine,
    client: GithubClient,
    repository: Any,
    page: int,
    cutoff: datetime,
    started_at: datetime,
) -> tuple[dict[str, Any] | None, int]:
    result = client.list_closed_pull_requests(repository.owner_login, repository.name, page)
    count = 0
    for summary in result.items:
        merged_at = _github_timestamp(summary.get("merged_at"))
        if merged_at is None or merged_at < cutoff:
            continue
        number = summary.get("number")
        if not isinstance(number, int):
            raise PermanentJobError("GitHub returned a pull request without a number")
        pull_request = client.get_pull_request(repository.owner_login, repository.name, number)
        commits = client.list_pull_request_commits(
            repository.owner_login,
            repository.name,
            number,
        )
        upsert_pull_request(
            database_engine,
            repository.workspace_id,
            repository.id,
            pull_request,
            "closed",
            commits,
        )
        count += 1

    oldest_update = min(
        (
            timestamp
            for item in result.items
            if (timestamp := _github_timestamp(item.get("updated_at"))) is not None
        ),
        default=None,
    )
    if result.next_page is not None and (oldest_update is None or oldest_update >= cutoff):
        return {"stage": PULL_REQUEST_STAGE, "page": result.next_page}, count

    signal = str(repository.settings.get("deploymentSignal", "WORKFLOW_RUN")).upper()
    if signal == "WORKFLOW_RUN":
        return _workflow_cursor(cutoff, started_at), count
    if signal == "DEPLOYMENT":
        return {"stage": DEPLOYMENT_STAGE, "page": 1}, count
    if signal == "PUSH":
        # GitHub does not expose historical push webhook deliveries. Pull requests
        # are still backfilled; deployment history starts with the next live push.
        return None, count
    raise PermanentJobError("Repository has an invalid deploymentSignal setting")


def _process_workflow_run_page(
    database_engine: Engine,
    client: GithubClient,
    repository: Any,
    cursor: dict[str, Any],
) -> tuple[dict[str, Any] | None, int]:
    page = cursor["page"]
    window_start = _github_timestamp(cursor.get("windowStart"))
    window_end = _github_timestamp(cursor.get("windowEnd"))
    if window_start is None or window_end is None or window_start >= window_end:
        raise PermanentJobError("Invalid workflow-run backfill window")
    result = client.list_workflow_runs(
        repository.owner_login,
        repository.name,
        page,
        branch=None,
        created_from=window_start.isoformat(),
        created_to=window_end.isoformat(),
    )
    # GitHub caps filtered workflow-run results at 1,000. Split a capped
    # interval before normalizing it so pagination can never silently omit
    # older runs. Inclusive boundaries may duplicate one run; upserts are
    # idempotent and therefore preferable to a gap.
    if page == 1 and result.total_count is not None and result.total_count >= 1_000:
        if window_end - window_start <= timedelta(seconds=1):
            raise PermanentJobError(
                "GitHub returned at least 1,000 workflow runs in one second; "
                "narrower lossless backfill is impossible"
            )
        midpoint = window_start + (window_end - window_start) / 2
        pending = [
            {"start": window_start.isoformat(), "end": midpoint.isoformat()},
            *cursor["pendingWindows"],
        ]
        return {
            "stage": WORKFLOW_RUN_STAGE,
            "page": 1,
            "windowStart": midpoint.isoformat(),
            "windowEnd": window_end.isoformat(),
            "pendingWindows": pending,
        }, 0
    count = 0
    for workflow_run in result.items:
        normalized_id = upsert_deployment_from_workflow_run(
            database_engine,
            repository.workspace_id,
            repository.id,
            {
                "action": "completed",
                "workflow_run": workflow_run,
                "repository": {"default_branch": repository.default_branch},
            },
        )
        count += int(normalized_id is not None)
    if result.next_page is not None:
        next_cursor = {**cursor, "page": result.next_page}
    elif cursor["pendingWindows"]:
        next_window, *remaining = cursor["pendingWindows"]
        next_cursor = {
            "stage": WORKFLOW_RUN_STAGE,
            "page": 1,
            "windowStart": next_window["start"],
            "windowEnd": next_window["end"],
            "pendingWindows": remaining,
        }
    else:
        next_cursor = None
    return next_cursor, count


def _process_deployment_page(
    database_engine: Engine,
    client: GithubClient,
    repository: Any,
    page: int,
    cutoff: datetime,
) -> tuple[dict[str, Any] | None, int]:
    result: ProviderPage[dict[str, Any]] = client.list_deployments(
        repository.owner_login, repository.name, page
    )
    count = 0
    reached_cutoff = False
    for deployment in result.items:
        created_at = _github_timestamp(deployment.get("created_at"))
        if created_at is not None and created_at < cutoff:
            reached_cutoff = True
            continue
        deployment_id = deployment.get("id")
        if not isinstance(deployment_id, int):
            raise PermanentJobError("GitHub returned a deployment without an id")
        status = client.terminal_deployment_status(
            repository.owner_login, repository.name, deployment_id
        )
        if status is None:
            continue
        normalized_id = upsert_deployment_from_deployment_status(
            database_engine,
            repository.workspace_id,
            repository.id,
            {"deployment": deployment, "deployment_status": status},
        )
        count += int(normalized_id is not None)

    next_cursor = None
    if result.next_page is not None and not reached_cutoff:
        next_cursor = {"stage": DEPLOYMENT_STAGE, "page": result.next_page}
    return next_cursor, count


def _process_issue_page(
    database_engine: Engine,
    client: GithubClient,
    repository: Any,
    page: int,
    sync_started_at: datetime,
) -> tuple[dict[str, Any] | None, int]:
    result = client.list_open_issues(repository.owner_login, repository.name, page)
    count = 0
    for issue in result.items:
        normalized_id = upsert_github_issue(
            database_engine,
            repository.workspace_id,
            repository.id,
            issue,
            observed_at=sync_started_at,
        )
        count += int(normalized_id is not None)

    if result.next_page is not None:
        return {"stage": ISSUE_STAGE, "page": result.next_page}, count

    # Anything that was open locally but absent from GitHub's complete open
    # result is now closed, deleted, or transferred out of this repository.
    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE github_issues
                SET state = 'CLOSED',
                    closed_at = COALESCE(closed_at, :sync_started_at),
                    updated_at = now(),
                    version = version + 1
                WHERE repository_id = :repository_id
                  AND workspace_id = :workspace_id
                  AND state = 'OPEN'
                  AND last_synced_at < :sync_started_at
                """
            ),
            {
                "repository_id": repository.id,
                "workspace_id": repository.workspace_id,
                "sync_started_at": sync_started_at,
            },
        )
    return None, count


def _cursor(
    value: object,
    cutoff: datetime,
    started_at: datetime,
    *,
    initial_stage: str = OPEN_PULL_REQUEST_STAGE,
) -> dict[str, Any]:
    if value is None:
        return {"stage": initial_stage, "page": 1}
    if not isinstance(value, dict):
        raise PermanentJobError("Invalid backfill cursor")
    stage = value.get("stage")
    if stage not in {
        OPEN_PULL_REQUEST_STAGE,
        PULL_REQUEST_STAGE,
        WORKFLOW_RUN_STAGE,
        DEPLOYMENT_STAGE,
        ISSUE_STAGE,
    }:
        raise PermanentJobError("Invalid backfill cursor stage")
    page = value.get("page")
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise PermanentJobError("Invalid backfill cursor page")
    if stage != WORKFLOW_RUN_STAGE:
        return {"stage": stage, "page": page}

    window_start = _github_timestamp(value.get("windowStart")) or cutoff
    window_end = _github_timestamp(value.get("windowEnd")) or started_at
    pending_raw = value.get("pendingWindows", [])
    if not isinstance(pending_raw, list) or len(pending_raw) > 64:
        raise PermanentJobError("Invalid workflow-run backfill windows")
    pending: list[dict[str, str]] = []
    for window in pending_raw:
        if not isinstance(window, dict):
            raise PermanentJobError("Invalid workflow-run backfill window")
        start = _github_timestamp(window.get("start"))
        end = _github_timestamp(window.get("end"))
        if start is None or end is None or start >= end:
            raise PermanentJobError("Invalid workflow-run backfill window")
        pending.append({"start": start.isoformat(), "end": end.isoformat()})
    if window_start >= window_end:
        raise PermanentJobError("Invalid workflow-run backfill window")
    return {
        "stage": stage,
        "page": page,
        "windowStart": window_start.isoformat(),
        "windowEnd": window_end.isoformat(),
        "pendingWindows": pending,
    }


def _changed_files(pull_request: dict[str, Any]) -> int:
    value = pull_request.get("changed_files")
    if isinstance(value, bool):
        raise PermanentJobError("GitHub returned an invalid changed_files count")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise PermanentJobError("GitHub returned an invalid changed_files count") from exc
    if parsed < 0:
        raise PermanentJobError("GitHub returned a negative changed_files count")
    return parsed


def _workflow_cursor(cutoff: datetime, started_at: datetime) -> dict[str, Any]:
    return {
        "stage": WORKFLOW_RUN_STAGE,
        "page": 1,
        "windowStart": cutoff.isoformat(),
        "windowEnd": started_at.isoformat(),
        "pendingWindows": [],
    }


def _bounded_days(value: object) -> int:
    if isinstance(value, bool):
        raise PermanentJobError("Invalid backfillDays")
    if not isinstance(value, (int, str)):
        raise PermanentJobError("Invalid backfillDays")
    try:
        days = int(value)
    except (TypeError, ValueError) as exc:
        raise PermanentJobError("Invalid backfillDays") from exc
    if days < 1 or days > 365:
        raise PermanentJobError("backfillDays must be between 1 and 365")
    return days


def _started_at(value: object) -> datetime:
    if value is None:
        return datetime.now(UTC)
    timestamp = _github_timestamp(value)
    if timestamp is None:
        raise PermanentJobError("Invalid backfillStartedAt")
    return timestamp


def _github_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)
