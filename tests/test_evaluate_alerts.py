from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import Engine, text

from app.db.models import ClaimedJob
from app.jobs.handlers.evaluate_alerts import (
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


def test_build_email_content() -> None:
    from app.core.config import Settings
    from app.jobs.handlers.evaluate_alerts import _build_email_content

    settings = Settings(app_frontend_base_url="http://localhost:5173")
    repo_id = uuid4()
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
        workspace_slug="adept-corp",
        repo_full_name="adept-corp/main-repo",
        repository_id=repo_id,
        settings=settings,
    )
    assert subject == "[Adept Alert] DORA Change Failure triggered for adept-corp/main-repo"
    assert "Condition: GT 15.0" in plain
    assert "Actual Value: 25.0" in plain
    assert f"http://localhost:5173/workspaces/adept-corp/analytics?repo={repo_id}" in plain
    assert f"http://localhost:5173/workspaces/adept-corp/analytics?repo={repo_id}" in html
    # Invariant: Must not contain diffs
    assert "diff --git" not in plain
    assert "diff --git" not in html


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
                    calculation_version
                ) VALUES (
                    :id, :ws_id, :repo_id, 'CHANGE_FAILURE_RATE_PERCENT', 'DAY',
                    now() - interval '1 day', now(), 25.0, 'percent', 10, '1.0.0'
                )
                """
            ),
            {"id": snapshot_id, "ws_id": ws_id, "repo_id": repo_id},
        )

    # Queue an EVALUATE_ALERTS job
    job_factory.insert(
        job_type="EVALUATE_ALERTS",
        payload={"workspace_id": str(ws_id), "repository_id": str(repo_id)},
    )

    with patch("app.jobs.handlers.evaluate_alerts.send_email") as mock_send_email:
        jobs = claim_jobs(database_engine, "test-alert-worker", 1)
        assert len(jobs) == 1
        dispatch_job(database_engine, jobs[0], "test-alert-worker")

        mock_send_email.assert_called_once()
        assert mock_send_email.call_args.kwargs["to_address"] == "devs@example.test"
        assert "High CFR Alert" in mock_send_email.call_args.kwargs["subject"]

    # Verify notification_deliveries row created and SENT
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
        assert delivery["status"] == "SENT"
        assert delivery["event_key"] == f"{rule_id}:{snapshot_id}"
        assert delivery["attempts"] == 1
        assert delivery["sent_at"] is not None

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

    # Cleanup
    with database_engine.begin() as conn:
        conn.execute(text("DELETE FROM workspaces WHERE id = :id"), {"id": ws_id})
        conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": user_id})
