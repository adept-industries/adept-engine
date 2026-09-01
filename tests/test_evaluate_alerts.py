import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import Engine, text

from app.db.models import ClaimedJob
from app.jobs.handlers.evaluate_alerts import (
    aggregate_dora_metric,
    enqueue_evaluate_alerts_job,
    handle_evaluate_alerts,
    matches,
)
from app.jobs.retry import PermanentJobError
from tests.conftest import JobFactory


def test_matches_supported_comparators() -> None:
    # GT
    assert matches("GT", Decimal("10.5"), Decimal("10.0"))
    assert not matches("GT", Decimal("10.0"), Decimal("10.0"))
    assert not matches("GT", Decimal("9.5"), Decimal("10.0"))

    # GTE
    assert matches("GTE", Decimal("10.5"), Decimal("10.0"))
    assert matches("GTE", Decimal("10.0"), Decimal("10.0"))
    assert not matches("GTE", Decimal("9.5"), Decimal("10.0"))

    # LT
    assert matches("LT", Decimal("9.5"), Decimal("10.0"))
    assert not matches("LT", Decimal("10.0"), Decimal("10.0"))
    assert not matches("LT", Decimal("10.5"), Decimal("10.0"))

    # LTE
    assert matches("LTE", Decimal("9.5"), Decimal("10.0"))
    assert matches("LTE", Decimal("10.0"), Decimal("10.0"))
    assert not matches("LTE", Decimal("10.5"), Decimal("10.0"))

    # EQ
    assert matches("EQ", Decimal("10.0"), Decimal("10.0"))
    assert not matches("EQ", Decimal("10.1"), Decimal("10.0"))


def test_matches_unsupported_comparator_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported comparator: NEQ"):
        matches("NEQ", Decimal("10.0"), Decimal("10.0"))


def test_handle_evaluate_alerts_missing_payload_raises() -> None:
    job = ClaimedJob(
        id=uuid4(),
        job_type="EVALUATE_ALERTS",
        payload={},
        priority=100,
        attempts=1,
        max_attempts=8,
        locked_by="worker-1",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        version=1,
    )
    engine = MagicMock(spec=Engine)
    with pytest.raises(PermanentJobError, match="missing repository_id or workspace_id"):
        handle_evaluate_alerts(engine, job, "worker-1")


def test_handle_evaluate_alerts_invalid_uuid_raises() -> None:
    job = ClaimedJob(
        id=uuid4(),
        job_type="EVALUATE_ALERTS",
        payload={"workspace_id": "not-a-uuid", "repository_id": "invalid"},
        priority=100,
        attempts=1,
        max_attempts=8,
        locked_by="worker-1",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        version=1,
    )
    engine = MagicMock(spec=Engine)
    with pytest.raises(PermanentJobError, match="Invalid UUID"):
        handle_evaluate_alerts(engine, job, "worker-1")


def test_enqueue_evaluate_alerts_job_deduplication() -> None:
    mock_conn = MagicMock()
    existing_id = uuid4()
    mock_conn.execute.return_value.scalar_one_or_none.return_value = str(existing_id)

    ws_id = uuid4()
    repo_id = uuid4()
    returned_id = enqueue_evaluate_alerts_job(mock_conn, ws_id, repo_id)

    assert returned_id == existing_id
    mock_conn.execute.assert_called_once()


def test_aggregate_dora_metric_filters_window_and_deduplicates_observations() -> None:
    snapshot_id = uuid4()
    window_end = datetime(2026, 9, 1, 12, tzinfo=UTC)
    window_start = window_end - timedelta(days=1)
    snapshots = [
        {
            "id": snapshot_id,
            "period_start": window_start,
            "calculated_at": window_end,
            "dimensions": {
                "observations": [
                    {
                        "key": "deploy-1",
                        "at": (window_start + timedelta(hours=1)).isoformat(),
                        "value": 1,
                    },
                    {
                        "key": "deploy-2",
                        "at": (window_start + timedelta(hours=2)).isoformat(),
                        "value": 0,
                    },
                    {
                        "key": "deploy-2",
                        "at": (window_start + timedelta(hours=2)).isoformat(),
                        "value": 1,
                    },
                    {
                        "key": "too-old",
                        "at": (window_start - timedelta(seconds=1)).isoformat(),
                        "value": 1,
                    },
                ]
            },
        }
    ]

    result = aggregate_dora_metric(
        "CHANGE_FAILURE_RATE_PERCENT", snapshots, window_start, window_end
    )

    assert result is not None
    assert result.actual == Decimal("50")
    assert result.observation_count == 2
    assert result.source_entity_id == snapshot_id
    assert result.period_start == window_start
    assert result.period_end == window_end


def test_aggregate_dora_metric_uses_median_and_handles_no_deployments() -> None:
    snapshot_id = uuid4()
    window_end = datetime(2026, 9, 1, 12, tzinfo=UTC)
    window_start = window_end - timedelta(days=1)
    snapshot = {
        "id": snapshot_id,
        "period_start": window_start,
        "calculated_at": window_end,
        "dimensions": {
            "observations": [
                {
                    "key": "lead-1",
                    "at": (window_start + timedelta(hours=1)).isoformat(),
                    "value": "1.5",
                },
                {
                    "key": "lead-2",
                    "at": (window_start + timedelta(hours=2)).isoformat(),
                    "value": "4.5",
                },
            ]
        },
    }

    duration = aggregate_dora_metric("CHANGE_LEAD_TIME_HOURS", [snapshot], window_start, window_end)
    no_deployments = aggregate_dora_metric(
        "DEPLOYMENT_FREQUENCY",
        [{**snapshot, "dimensions": {"observations": []}}],
        window_start,
        window_end,
    )

    assert duration is not None
    assert duration.actual == Decimal("3.0")
    assert no_deployments is not None
    assert no_deployments.actual == Decimal("0")
    assert no_deployments.observation_count == 0


def test_build_email_content() -> None:
    from app.core.config import Settings
    from app.jobs.handlers.evaluate_alerts import _build_email_content

    settings = Settings(app_frontend_base_url="http://localhost:5173")
    subject, plain, html = _build_email_content(
        rule_name="DORA Change Failure",
        metric_type="CHANGE_FAILURE_RATE_PERCENT",
        comparator="GT",
        actual=Decimal("25.0"),
        threshold=Decimal("15.0"),
        evaluation_window_minutes=1440,
        period_start="2026-08-30T00:00:00Z",
        period_end="2026-08-31T00:00:00Z",
        workspace_name="Adept Corp",
        repo_full_name="adept-corp/main-repo",
        settings=settings,
    )
    assert subject == "[Adept Alert] DORA Change Failure triggered for adept-corp/main-repo"
    assert "Condition: GT 15.0" in plain
    assert "Actual Value: 25.0" in plain
    assert "http://localhost:5173/dashboard" in plain
    assert "http://localhost:5173/dashboard" in html
    # Invariant: Must not contain diffs
    assert "diff --git" not in plain
    assert "diff --git" not in html


def test_build_email_content_sanitizes_headers_and_escapes_html() -> None:
    from app.core.config import Settings
    from app.jobs.handlers.evaluate_alerts import _build_email_content

    subject, _, html = _build_email_content(
        rule_name="High risk\nBcc: attacker@example.test <script>",
        metric_type="PR_RISK_SCORE",
        comparator="GT",
        actual=Decimal("0.8"),
        threshold=Decimal("0.5"),
        evaluation_window_minutes=60,
        period_start=None,
        period_end=None,
        workspace_name="<script>alert(1)</script>",
        repo_full_name="org/<b>repo</b>",
        settings=Settings(app_frontend_base_url="https://adept.example/"),
    )

    assert "\n" not in subject
    assert "\r" not in subject
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "org/&lt;b&gt;repo&lt;/b&gt;" in html
    assert "https://adept.example/dashboard" in html


@pytest.mark.integration
def test_evaluate_alerts_integration_end_to_end(
    database_engine: Engine,
    job_factory: JobFactory,
) -> None:
    from app.jobs.claimer import claim_jobs
    from app.jobs.dispatcher import dispatch_job

    ws_id = uuid4()
    repo_id = uuid4()
    user_id = uuid4()
    integ_id = uuid4()
    rule_id = uuid4()
    snapshot_id = uuid4()
    observed_at = datetime.now(UTC) - timedelta(minutes=30)
    observations = [
        {
            "key": f"deployment-{index}",
            "at": (observed_at + timedelta(seconds=index)).isoformat(),
            "value": 1 if index == 0 else 0,
        }
        for index in range(4)
    ]

    with database_engine.begin() as conn:
        # Create user & workspace
        conn.execute(
            text(
                "INSERT INTO users (id, email, password_hash, display_name) "
                "VALUES (:id, :email, 'hash', 'Test')"
            ),
            {"id": user_id, "email": f"alert-{ws_id.hex}@example.test"},
        )
        conn.execute(
            text(
                "INSERT INTO workspaces (id, name, slug, timezone) "
                "VALUES (:id, 'Alert Workspace', :slug, 'UTC')"
            ),
            {"id": ws_id, "slug": f"alert-{ws_id.hex}"},
        )
        conn.execute(
            text(
                """
                INSERT INTO github_integrations (
                    id, workspace_id, installation_id, account_external_id,
                    account_login, account_type, repository_selection, status
                ) VALUES (
                    :id, :ws_id, 7200, 8200, 'adept-industries', 'ORGANIZATION', 'ALL', 'ACTIVE'
                )
                """
            ),
            {"id": integ_id, "ws_id": ws_id},
        )
        conn.execute(
            text(
                """
                INSERT INTO repositories (
                    id, workspace_id, github_integration_id, github_repo_id,
                    owner_login, name, full_name, default_branch, visibility, tracking_enabled
                ) VALUES (
                    :id, :ws_id, :integ_id, 9200,
                    'adept-industries', 'alert-repo', 'adept-industries/alert-repo',
                    'main', 'PRIVATE', true
                )
                """
            ),
            {"id": repo_id, "ws_id": ws_id, "integ_id": integ_id},
        )
        # Create enabled alert rule
        conn.execute(
            text(
                """
                INSERT INTO alert_rules (
                    id, workspace_id, repository_id, name, metric_type,
                    comparator, threshold_value, evaluation_window_minutes,
                    cooldown_minutes, channel, destination, enabled
                ) VALUES (
                    :rule_id, :ws_id, :repo_id, 'High CFR Alert', 'CHANGE_FAILURE_RATE_PERCENT',
                    'GT', 15.0, 1440, 60, 'EMAIL', 'devs@example.test', true
                )
                """
            ),
            {"rule_id": rule_id, "ws_id": ws_id, "repo_id": repo_id},
        )
        # Create metric snapshot where value = 25.0 (> 15.0)
        conn.execute(
            text(
                """
                INSERT INTO metric_snapshots (
                    id, workspace_id, repository_id, metric_type, granularity,
                    period_start, period_end, value, unit, sample_size,
                    calculation_version, dimensions
                ) VALUES (
                    :id, :ws_id, :repo_id, 'CHANGE_FAILURE_RATE_PERCENT', 'DAY',
                    :period_start, :period_end, 25.0, 'percent', 4,
                    'dora-v3', CAST(:dimensions AS jsonb)
                )
                """
            ),
            {
                "id": snapshot_id,
                "ws_id": ws_id,
                "repo_id": repo_id,
                "period_start": observed_at.replace(hour=0, minute=0, second=0, microsecond=0),
                "period_end": observed_at.replace(hour=0, minute=0, second=0, microsecond=0)
                + timedelta(days=1),
                "dimensions": json.dumps({"observations": observations}),
            },
        )

    # Queue an EVALUATE_ALERTS job
    evaluation_job_id = job_factory.insert(
        job_type="EVALUATE_ALERTS",
        payload={"workspace_id": str(ws_id), "repository_id": str(repo_id)},
    )

    jobs = claim_jobs(database_engine, "test-alert-worker", 1)
    assert len(jobs) == 1
    dispatch_job(database_engine, jobs[0], "test-alert-worker")

    # The engine owns evaluation only. adept-api owns SMTP delivery and retries.
    with database_engine.connect() as conn:
        delivery = (
            conn.execute(
                text(
                    """
                SELECT * FROM notification_deliveries
                WHERE alert_rule_id = :rule_id
                """
                ),
                {"rule_id": rule_id},
            )
            .mappings()
            .one()
        )
        assert delivery["status"] == "PENDING"
        assert delivery["event_key"] == f"evaluation:{evaluation_job_id}"
        assert delivery["attempts"] == 0
        assert delivery["sent_at"] is None
        assert delivery["payload"]["actual_value"] == "25"
        assert delivery["payload"]["source_observation_count"] == 4

        # Verify alert_rule last_triggered_at was updated
        rule_row = (
            conn.execute(
                text("SELECT last_triggered_at FROM alert_rules WHERE id = :id"),
                {"id": rule_id},
            )
            .mappings()
            .one()
        )
        assert rule_row["last_triggered_at"] is not None

    # A separate evaluation inside the cooldown window cannot create a second delivery.
    job_factory.insert(
        job_type="EVALUATE_ALERTS",
        payload={"workspace_id": str(ws_id), "repository_id": str(repo_id)},
    )
    cooldown_jobs = claim_jobs(database_engine, "test-alert-worker", 1)
    assert len(cooldown_jobs) == 1
    dispatch_job(database_engine, cooldown_jobs[0], "test-alert-worker")
    with database_engine.connect() as conn:
        delivery_count = conn.execute(
            text("SELECT count(*) FROM notification_deliveries WHERE alert_rule_id = :rule_id"),
            {"rule_id": rule_id},
        ).scalar_one()
    assert delivery_count == 1

    # Cleanup
    with database_engine.begin() as conn:
        conn.execute(text("DELETE FROM workspaces WHERE id = :id"), {"id": ws_id})
        conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": user_id})
