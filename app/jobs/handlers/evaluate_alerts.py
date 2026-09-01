"""
Job handler for EVALUATE_ALERTS.

Evaluates enabled alert rules for a repository when new metrics or risk predictions
are produced and creates deterministic notification deliveries for the platform mailer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from html import escape
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import Connection, Engine, text

from app.core.config import Settings, get_settings
from app.db.models import ClaimedJob
from app.jobs.retry import PermanentJobError

logger = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class EvaluatedMetric:
    actual: Decimal
    source_entity_id: UUID
    observation_count: int
    period_start: datetime
    period_end: datetime


def _parse_observation_time(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except TypeError, ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _median(values: list[Decimal]) -> Decimal:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / Decimal(2)


def aggregate_dora_metric(
    metric_type: str,
    snapshots: Sequence[Mapping[str, Any]],
    window_start: datetime,
    window_end: datetime,
) -> EvaluatedMetric | None:
    """Aggregate canonical DORA observations inside an exact rolling window."""
    if not snapshots:
        return None

    observations: dict[str, Decimal] = {}
    for snapshot in snapshots:
        dimensions = snapshot.get("dimensions")
        raw_observations = (
            dimensions.get("observations") if isinstance(dimensions, Mapping) else None
        )
        if not isinstance(raw_observations, list):
            continue

        for index, raw_observation in enumerate(raw_observations):
            if not isinstance(raw_observation, dict):
                continue
            observed_at = _parse_observation_time(raw_observation.get("at"))
            if observed_at is None or not window_start <= observed_at < window_end:
                continue
            try:
                value = Decimal(str(raw_observation.get("value")))
            except InvalidOperation, ValueError:
                continue
            if not value.is_finite():
                continue
            raw_key = raw_observation.get("key")
            key = str(raw_key).strip() if raw_key is not None else ""
            if not key:
                key = f"{snapshot['id']}:{index}"
            observations.setdefault(key, value)

    values = list(observations.values())
    if metric_type == "DEPLOYMENT_FREQUENCY":
        actual = Decimal(len(values))
    elif metric_type == "CHANGE_FAILURE_RATE_PERCENT":
        if not values:
            return None
        failed = sum(1 for value in values if value >= Decimal("0.5"))
        actual = Decimal(failed) * Decimal(100) / Decimal(len(values))
    elif metric_type in {
        "CHANGE_LEAD_TIME_HOURS",
        "FAILED_DEPLOYMENT_RECOVERY_TIME_HOURS",
    }:
        if not values:
            return None
        actual = _median(values)
    else:
        return None

    newest_snapshot = max(
        snapshots,
        key=lambda snapshot: (
            snapshot.get("period_start") or window_start,
            snapshot.get("calculated_at") or window_start,
        ),
    )
    return EvaluatedMetric(
        actual=actual,
        source_entity_id=UUID(str(newest_snapshot["id"])),
        observation_count=len(values),
        period_start=window_start,
        period_end=window_end,
    )


def _safe_header(value: str) -> str:
    return " ".join(value.replace("\r", " ").replace("\n", " ").split())


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
) -> tuple[str, str]:
    """Returns (workspace_name, repo_full_name)."""
    row = (
        connection.execute(
            text(
                """
            SELECT w.name AS workspace_name, r.full_name AS repo_full_name
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
        return ("Unknown Workspace", "Unknown Repository")
    return (str(row["workspace_name"]), str(row["repo_full_name"]))


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
    repo_full_name: str,
    settings: Settings,
) -> tuple[str, str, str]:
    """
    Generates (subject, plain_text, html_body) for the alert.
    Safely includes workspace, repository, metric, actual value, threshold,
    evaluation period, and a safe dashboard link. Does NOT include diffs.
    """
    subject = (
        f"[Adept Alert] {_safe_header(rule_name)} triggered for {_safe_header(repo_full_name)}"
    )
    dashboard_link = f"{settings.app_frontend_base_url.rstrip('/')}/dashboard"

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
        f"<h2>Alert Triggered: {escape(rule_name)}</h2>"
        f"<p><strong>Workspace:</strong> {escape(workspace_name)}<br/>"
        f"<strong>Repository:</strong> {escape(repo_full_name)}<br/>"
        f"<strong>Metric:</strong> {escape(metric_type)}<br/>"
        f"<strong>Condition:</strong> {escape(comparator)} {escape(str(threshold))}<br/>"
        f"<strong>Actual Value:</strong> {escape(str(actual))}<br/>"
        f"<strong>Evaluation Period:</strong> {escape(period_str)}</p>"
        f'<p><a href="{escape(dashboard_link, quote=True)}" style="{link_style}">'
        "View Dashboard</a></p>"
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
    2. Evaluates observations inside each rule's exact rolling window.
    3. Locks the rule and checks cooldown atomically.
    4. Inserts one deterministic PENDING delivery per evaluation job.
    5. Leaves SMTP delivery and retry ownership to adept-api.
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
        workspace_name, repo_full_name = _fetch_workspace_and_repo_names(
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

    evaluation_end = datetime.now(UTC)

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

        evaluation_start = evaluation_end - timedelta(minutes=evaluation_window_minutes)
        evaluated_metric: EvaluatedMetric | None = None

        with database_engine.connect() as connection:
            if metric_type == "PR_RISK_SCORE":
                # A risk rule only evaluates predictions produced inside its rolling window.
                pred = (
                    connection.execute(
                        text(
                            """
                        SELECT id, risk_score, predicted_at
                        FROM risk_predictions
                        WHERE repository_id = :repository_id
                          AND predicted_at >= :window_start
                          AND predicted_at < :window_end
                        ORDER BY predicted_at DESC, created_at DESC
                        LIMIT 1
                        """
                        ),
                        {
                            "repository_id": repository_id,
                            "window_start": evaluation_start,
                            "window_end": evaluation_end,
                        },
                    )
                    .mappings()
                    .one_or_none()
                )

                if pred is not None:
                    evaluated_metric = EvaluatedMetric(
                        actual=Decimal(str(pred["risk_score"])),
                        source_entity_id=UUID(str(pred["id"])),
                        observation_count=1,
                        period_start=evaluation_start,
                        period_end=evaluation_end,
                    )
            else:
                # DAY snapshots hold the canonical observations used by the dashboard.
                # Aggregate every overlapping bucket and filter its observations to the
                # exact rolling window instead of selecting one potentially stale value.
                snapshots = [
                    dict(snapshot)
                    for snapshot in connection.execute(
                        text(
                            """
                        SELECT id, period_start, period_end, dimensions, calculated_at
                        FROM metric_snapshots
                        WHERE repository_id = :repository_id
                          AND metric_type = :metric_type
                          AND granularity = 'DAY'
                          AND calculation_version = 'dora-v3'
                          AND period_end > :window_start
                          AND period_start < :window_end
                        ORDER BY period_start ASC, calculated_at ASC
                        """
                        ),
                        {
                            "repository_id": repository_id,
                            "metric_type": metric_type,
                            "window_start": evaluation_start,
                            "window_end": evaluation_end,
                        },
                    )
                    .mappings()
                    .all()
                ]
                evaluated_metric = aggregate_dora_metric(
                    metric_type,
                    snapshots,
                    evaluation_start,
                    evaluation_end,
                )

        if evaluated_metric is None:
            logger.info(
                "alert_rule_skipped_no_data",
                rule_id=str(rule_id),
                rule_name=rule_name,
                metric_type=metric_type,
                repository_id=str(repository_id),
            )
            continue

        actual_value = evaluated_metric.actual

        # Check comparator match
        try:
            is_match = matches(comparator, actual_value, threshold_value)
        except ValueError as exc:
            logger.error("invalid_comparator_in_rule", rule_id=str(rule_id), error=str(exc))
            continue

        if not is_match:
            logger.info(
                "alert_rule_condition_not_met",
                rule_id=str(rule_id),
                rule_name=rule_name,
                metric_type=metric_type,
                comparator=comparator,
                actual_value=str(actual_value),
                threshold_value=str(threshold_value),
            )
            continue

        # A processing job is the qualifying evaluation event. Retries keep the same
        # job ID; a later observation creates a new job and may notify after cooldown.
        event_key = f"evaluation:{job.id}"

        # Build email content before persisting delivery so payload stores pre-rendered content
        subject, text_content, html_content = _build_email_content(
            rule_name=rule_name,
            metric_type=metric_type,
            comparator=comparator,
            actual=actual_value,
            threshold=threshold_value,
            evaluation_window_minutes=evaluation_window_minutes,
            period_start=evaluated_metric.period_start.isoformat(),
            period_end=evaluated_metric.period_end.isoformat(),
            workspace_name=workspace_name,
            repo_full_name=repo_full_name,
            settings=settings,
        )

        delivery_payload = {
            "subject": subject,
            "text": text_content,
            "html": html_content,
            "rule_id": str(rule_id),
            "rule_name": rule_name,
            "metric_type": metric_type,
            "comparator": comparator,
            "threshold_value": str(threshold_value),
            "actual_value": str(actual_value),
            "source_entity_id": str(evaluated_metric.source_entity_id),
            "source_observation_count": evaluated_metric.observation_count,
            "evaluation_job_id": str(job.id),
            "evaluation_window_minutes": evaluation_window_minutes,
            "period_start": evaluated_metric.period_start.isoformat(),
            "period_end": evaluated_metric.period_end.isoformat(),
            "workspace_id": str(workspace_id),
            "repository_id": str(repository_id),
        }

        queued_delivery_id: UUID | None = None
        with database_engine.begin() as connection:
            # Serialize the cooldown decision for this rule so concurrent workers cannot
            # create two deliveries inside one cooldown window.
            cooldown = (
                connection.execute(
                    text(
                        """
                        SELECT (
                            last_triggered_at IS NOT NULL
                            AND now() < last_triggered_at + make_interval(mins => :cooldown_mins)
                        ) AS in_cooldown
                        FROM alert_rules
                        WHERE id = :rule_id
                          AND enabled = true
                        FOR UPDATE
                        """
                    ),
                    {"rule_id": rule_id, "cooldown_mins": cooldown_minutes},
                )
                .mappings()
                .one_or_none()
            )
            if cooldown is None:
                continue
            if bool(cooldown["in_cooldown"]):
                logger.info(
                    "alert_suppressed_by_cooldown",
                    rule_id=str(rule_id),
                    cooldown_minutes=cooldown_minutes,
                )
                continue

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
                    ON CONFLICT (alert_rule_id, event_key) DO NOTHING
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
            ).scalar_one_or_none()
            if new_id is None:
                logger.info(
                    "alert_delivery_event_already_queued",
                    rule_id=str(rule_id),
                    event_key=event_key,
                )
                continue

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
            queued_delivery_id = UUID(str(new_id))

        if queued_delivery_id is not None:
            logger.info(
                "alert_notification_queued",
                delivery_id=str(queued_delivery_id),
                rule_id=str(rule_id),
                event_key=event_key,
            )
