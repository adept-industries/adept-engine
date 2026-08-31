"""
Job handler for EVALUATE_ALERTS.

Evaluates enabled alert rules for a repository when new metrics or risk predictions
are produced, creates deterministic notification deliveries, and dispatches notifications.
"""

from __future__ import annotations

import json
from decimal import Decimal
from uuid import UUID

import structlog
from sqlalchemy import Connection, Engine, text

from app.core.config import Settings, get_settings
from app.db.models import ClaimedJob
from app.jobs.retry import PermanentJobError
from app.providers.email import send_email

logger = structlog.get_logger()


def matches(comparator: str, actual: Decimal, threshold: Decimal) -> bool:
    """
    Evaluate comparator between actual and threshold values using Decimal.
    Raises ValueError for unsupported comparators.
    """
    comparisons = {
        "GT": actual > threshold,
        "GTE": actual >= threshold,
        "LT": actual < threshold,
        "LTE": actual <= threshold,
        "EQ": actual == threshold,
    }
    if comparator not in comparisons:
        raise ValueError(f"Unsupported comparator: {comparator}")
    return comparisons[comparator]


def enqueue_evaluate_alerts_job(
    connection: Connection,
    workspace_id: UUID,
    repository_id: UUID,
    trigger_source: str = "METRIC_SNAPSHOT",
) -> UUID:
    """
    Queue an EVALUATE_ALERTS processing job.
    Deduplicates against any existing PENDING EVALUATE_ALERTS job for the repository.
    """
    existing_job = connection.execute(
        text(
            """
            SELECT id FROM processing_jobs
            WHERE repository_id = :repository_id
              AND job_type = 'EVALUATE_ALERTS'
              AND status = 'PENDING'
            LIMIT 1
            """
        ),
        {"repository_id": repository_id},
    ).scalar_one_or_none()

    if existing_job is not None:
        return UUID(str(existing_job))

    payload = {
        "workspace_id": str(workspace_id),
        "repository_id": str(repository_id),
        "trigger_source": trigger_source,
    }

    job_id = connection.execute(
        text(
            """
            INSERT INTO processing_jobs (
                workspace_id, repository_id, job_type, payload, status, priority, available_at
            ) VALUES (
                :workspace_id, :repository_id, 'EVALUATE_ALERTS',
                CAST(:payload AS jsonb), 'PENDING', 100, now()
            )
            RETURNING id
            """
        ),
        {
            "workspace_id": workspace_id,
            "repository_id": repository_id,
            "payload": json.dumps(payload),
        },
    ).scalar_one()

    return UUID(str(job_id))


def _fetch_workspace_and_repo_names(
    connection: Connection,
    workspace_id: UUID,
    repository_id: UUID,
) -> tuple[str, str, str]:
    """Returns (workspace_name, workspace_slug, repo_full_name)."""
    row = (
        connection.execute(
            text(
                """
            SELECT w.name AS workspace_name, w.slug AS workspace_slug, r.full_name AS repo_full_name
            FROM workspaces w
            JOIN repositories r ON r.workspace_id = w.id
            WHERE w.id = :workspace_id AND r.id = :repository_id
            """
            ),
            {"workspace_id": workspace_id, "repository_id": repository_id},
        )
        .mappings()
        .one_or_none()
    )

    if row is None:
        return ("Unknown Workspace", str(workspace_id), "Unknown Repository")
    return (str(row["workspace_name"]), str(row["workspace_slug"]), str(row["repo_full_name"]))


def _build_email_content(
    rule_name: str,
    metric_type: str,
    comparator: str,
    actual: Decimal,
    threshold: Decimal,
    evaluation_window_minutes: int,
    period_start: str | None,
    period_end: str | None,
    workspace_name: str,
    workspace_slug: str,
    repo_full_name: str,
    repository_id: UUID,
    settings: Settings,
) -> tuple[str, str, str]:
    """
    Generates (subject, plain_text, html_body) for the alert.
    Safely includes workspace, repository, metric, actual value, threshold,
    evaluation period, and a link to the filtered dashboard. Does NOT include diffs.
    """
    subject = f"[Adept Alert] {rule_name} triggered for {repo_full_name}"
    path = f"/workspaces/{workspace_slug}/analytics?repo={repository_id}"
    dashboard_link = f"{settings.app_frontend_base_url}{path}"

    period_str = f"{evaluation_window_minutes} minutes"
    if period_start and period_end:
        period_str += f" ({period_start} to {period_end})"

    plain_text = (
        f"Alert Rule: {rule_name}\n"
        f"Workspace: {workspace_name}\n"
        f"Repository: {repo_full_name}\n"
        f"Metric: {metric_type}\n"
        f"Condition: {comparator} {threshold}\n"
        f"Actual Value: {actual}\n"
        f"Evaluation Period: {period_str}\n\n"
        f"View repository metrics dashboard:\n{dashboard_link}\n"
    )

    link_style = (
        "background-color: #2563eb; color: white; padding: 8px 16px; "
        "text-decoration: none; border-radius: 4px;"
    )
    html_body = (
        f"<h2>Alert Triggered: {rule_name}</h2>"
        f"<p><strong>Workspace:</strong> {workspace_name}<br/>"
        f"<strong>Repository:</strong> {repo_full_name}<br/>"
        f"<strong>Metric:</strong> {metric_type}<br/>"
        f"<strong>Condition:</strong> {comparator} {threshold}<br/>"
        f"<strong>Actual Value:</strong> {actual}<br/>"
        f"<strong>Evaluation Period:</strong> {period_str}</p>"
        f'<p><a href="{dashboard_link}" style="{link_style}">View Filtered Dashboard</a></p>'
    )

    return subject, plain_text, html_body


def handle_evaluate_alerts(
    database_engine: Engine,
    job: ClaimedJob,
    worker_id: str,
) -> None:
    """
    Processes an EVALUATE_ALERTS job:
    1. Loads enabled rules for the repository.
    2. Evaluates newest values (metric snapshot or risk prediction).
    3. Checks cooldown.
    4. Generates deterministic event_key.
    5. Inserts notification_deliveries row and updates rule's last_triggered_at.
    6. Sends email and updates delivery status (retries update existing row).
    """
    settings = get_settings()
    payload = job.payload or {}
    repo_id_str = payload.get("repository_id") or payload.get("repositoryId")
    workspace_id_str = payload.get("workspace_id") or payload.get("workspaceId")

    if not repo_id_str or not workspace_id_str:
        raise PermanentJobError(
            f"EVALUATE_ALERTS job {job.id} missing repository_id or workspace_id"
        )

    try:
        repository_id = UUID(str(repo_id_str))
        workspace_id = UUID(str(workspace_id_str))
    except ValueError as exc:
        raise PermanentJobError(f"Invalid UUID in EVALUATE_ALERTS payload: {exc}") from exc

    with database_engine.connect() as connection:
        workspace_name, workspace_slug, repo_full_name = _fetch_workspace_and_repo_names(
            connection, workspace_id, repository_id
        )

        rules = (
            connection.execute(
                text(
                    """
                SELECT id, name, metric_type, comparator, threshold_value,
                       evaluation_window_minutes, cooldown_minutes, channel, destination,
                       last_triggered_at
                FROM alert_rules
                WHERE repository_id = :repository_id
                  AND enabled = true
                ORDER BY created_at ASC
                """
                ),
                {"repository_id": repository_id},
            )
            .mappings()
            .all()
        )

    if not rules:
        logger.info("no_enabled_alert_rules", repository_id=str(repository_id))
        return

    for rule in rules:
        rule_id = UUID(str(rule["id"]))
        rule_name = str(rule["name"])
        metric_type = str(rule["metric_type"])
        comparator = str(rule["comparator"])
        threshold_value = Decimal(str(rule["threshold_value"]))
        evaluation_window_minutes = int(rule["evaluation_window_minutes"])
        cooldown_minutes = int(rule["cooldown_minutes"])
        channel = str(rule["channel"])
        destination = str(rule["destination"])

        # Fetch newest value and source entity ID
        actual_value: Decimal | None = None
        source_entity_id: UUID | None = None
        period_start_str: str | None = None
        period_end_str: str | None = None

        with database_engine.connect() as connection:
            if metric_type == "PR_RISK_SCORE":
                # Evaluate newest risk prediction for this repository
                pred = (
                    connection.execute(
                        text(
                            """
                        SELECT id, risk_score, predicted_at
                        FROM risk_predictions
                        WHERE repository_id = :repository_id
                        ORDER BY predicted_at DESC, created_at DESC
                        LIMIT 1
                        """
                        ),
                        {"repository_id": repository_id},
                    )
                    .mappings()
                    .one_or_none()
                )

                if pred is not None:
                    actual_value = Decimal(str(pred["risk_score"]))
                    source_entity_id = UUID(str(pred["id"]))
                    period_start_str = str(pred["predicted_at"])
                    period_end_str = str(pred["predicted_at"])
            else:
                # DORA metric: evaluate newest snapshot matching granularity or metric_type
                # Window <= 1440 min -> DAY, <= 10080 -> WEEK, else MONTH
                granularity = (
                    "DAY"
                    if evaluation_window_minutes <= 1440
                    else ("WEEK" if evaluation_window_minutes <= 10080 else "MONTH")
                )
                snap = (
                    connection.execute(
                        text(
                            """
                        SELECT id, value, period_start, period_end
                        FROM metric_snapshots
                        WHERE repository_id = :repository_id
                          AND metric_type = :metric_type
                          AND granularity = :granularity
                        ORDER BY period_start DESC, calculated_at DESC
                        LIMIT 1
                        """
                        ),
                        {
                            "repository_id": repository_id,
                            "metric_type": metric_type,
                            "granularity": granularity,
                        },
                    )
                    .mappings()
                    .one_or_none()
                )

                # Fallback to newest snapshot of any granularity if preferred not found
                if snap is None:
                    snap = (
                        connection.execute(
                            text(
                                """
                            SELECT id, value, period_start, period_end
                            FROM metric_snapshots
                            WHERE repository_id = :repository_id
                              AND metric_type = :metric_type
                            ORDER BY period_start DESC, calculated_at DESC
                            LIMIT 1
                            """
                            ),
                            {
                                "repository_id": repository_id,
                                "metric_type": metric_type,
                            },
                        )
                        .mappings()
                        .one_or_none()
                    )

                if snap is not None:
                    actual_value = Decimal(str(snap["value"]))
                    source_entity_id = UUID(str(snap["id"]))
                    period_start_str = str(snap["period_start"])
                    period_end_str = str(snap["period_end"])

        if actual_value is None or source_entity_id is None:
            continue

        # Check comparator match
        try:
            is_match = matches(comparator, actual_value, threshold_value)
        except ValueError as exc:
            logger.error("invalid_comparator_in_rule", rule_id=str(rule_id), error=str(exc))
            continue

        if not is_match:
            continue

        # Check cooldown: suppress if now < last_triggered_at + cooldown_minutes
        with database_engine.connect() as connection:
            cooldown_sql = text(
                """
                SELECT (
                    last_triggered_at IS NOT NULL
                    AND now() < last_triggered_at + make_interval(mins => :cooldown_mins)
                ) AS in_cooldown
                FROM alert_rules
                WHERE id = :rule_id
                """
            )
            in_cooldown = connection.execute(
                cooldown_sql,
                {"rule_id": rule_id, "cooldown_mins": cooldown_minutes},
            ).scalar_one_or_none()

            if in_cooldown:
                logger.info(
                    "alert_suppressed_by_cooldown",
                    rule_id=str(rule_id),
                    cooldown_minutes=cooldown_minutes,
                )
                continue

        event_key = f"{rule_id}:{source_entity_id}"
        delivery_payload = {
            "rule_id": str(rule_id),
            "rule_name": rule_name,
            "metric_type": metric_type,
            "comparator": comparator,
            "threshold_value": str(threshold_value),
            "actual_value": str(actual_value),
            "source_entity_id": str(source_entity_id),
            "evaluation_window_minutes": evaluation_window_minutes,
            "period_start": period_start_str,
            "period_end": period_end_str,
            "workspace_id": str(workspace_id),
            "repository_id": str(repository_id),
        }

        # Step 5 & 7: Insert delivery if absent and set last_triggered_at in same tx
        delivery_id: UUID | None = None
        should_send = False

        with database_engine.begin() as connection:
            # Check existing delivery
            existing_delivery = (
                connection.execute(
                    text(
                        """
                    SELECT id, status, attempts
                    FROM notification_deliveries
                    WHERE alert_rule_id = :rule_id
                      AND event_key = :event_key
                    FOR UPDATE
                    """
                    ),
                    {"rule_id": rule_id, "event_key": event_key},
                )
                .mappings()
                .one_or_none()
            )

            if existing_delivery is None:
                new_id = connection.execute(
                    text(
                        """
                        INSERT INTO notification_deliveries (
                            workspace_id, repository_id, alert_rule_id, event_key,
                            channel, destination, status, payload, attempts,
                            created_at, updated_at
                        ) VALUES (
                            :workspace_id, :repository_id, :rule_id, :event_key,
                            :channel, :destination, 'PENDING', CAST(:payload AS jsonb), 0,
                            now(), now()
                        )
                        RETURNING id
                        """
                    ),
                    {
                        "workspace_id": workspace_id,
                        "repository_id": repository_id,
                        "rule_id": rule_id,
                        "event_key": event_key,
                        "channel": channel,
                        "destination": destination,
                        "payload": json.dumps(delivery_payload),
                    },
                ).scalar_one()

                # Set last_triggered_at on alert_rule in same transaction
                connection.execute(
                    text(
                        """
                        UPDATE alert_rules
                        SET last_triggered_at = now(),
                            updated_at = now(),
                            version = version + 1
                        WHERE id = :rule_id
                        """
                    ),
                    {"rule_id": rule_id},
                )
                delivery_id = UUID(str(new_id))
                should_send = True
            else:
                delivery_id = UUID(str(existing_delivery["id"]))
                curr_status = str(existing_delivery["status"])
                if curr_status in ("PENDING", "FAILED"):
                    should_send = True

        if not should_send or delivery_id is None:
            continue

        # Step 6: Send email and update delivery status
        subject, text_content, html_content = _build_email_content(
            rule_name=rule_name,
            metric_type=metric_type,
            comparator=comparator,
            actual=actual_value,
            threshold=threshold_value,
            evaluation_window_minutes=evaluation_window_minutes,
            period_start=period_start_str,
            period_end=period_end_str,
            workspace_name=workspace_name,
            workspace_slug=workspace_slug,
            repo_full_name=repo_full_name,
            repository_id=repository_id,
            settings=settings,
        )

        try:
            send_email(
                to_address=destination,
                subject=subject,
                text_content=text_content,
                html_content=html_content,
                settings=settings,
            )
            with database_engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        UPDATE notification_deliveries
                        SET status = 'SENT',
                            sent_at = now(),
                            attempts = attempts + 1,
                            last_error = NULL,
                            updated_at = now(),
                            version = version + 1
                        WHERE id = :delivery_id
                        """
                    ),
                    {"delivery_id": delivery_id},
                )
            logger.info(
                "alert_notification_sent", delivery_id=str(delivery_id), destination=destination
            )
        except Exception as exc:
            error_msg = str(exc)
            logger.error("alert_notification_failed", delivery_id=str(delivery_id), error=error_msg)
            with database_engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        UPDATE notification_deliveries
                        SET attempts = attempts + 1,
                            status = CASE WHEN attempts + 1 >= 5 THEN 'DEAD' ELSE 'FAILED' END,
                            last_error = :error_msg,
                            updated_at = now(),
                            version = version + 1
                        WHERE id = :delivery_id
                        """
                    ),
                    {"delivery_id": delivery_id, "error_msg": error_msg},
                )
            raise
