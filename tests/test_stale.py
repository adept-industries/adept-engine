"""Tests for deterministic stale pull request detection."""

from datetime import UTC, datetime

from app.risk.stale import calculate_hours_since_activity, check_stale


def test_stale_boundary_conditions() -> None:
    # 1. Open PR under threshold (119.9h) -> not stale
    res_under = check_stale(is_open=True, hours_since_last_activity=119.9, threshold_hours=120.0)
    assert res_under["is_stale"] is False

    # 2. Open PR exactly at threshold (120.0h) -> stale
    res_exact = check_stale(is_open=True, hours_since_last_activity=120.0, threshold_hours=120.0)
    assert res_exact["is_stale"] is True

    # 3. Open PR above threshold (120.1h) -> stale
    res_over = check_stale(is_open=True, hours_since_last_activity=120.1, threshold_hours=120.0)
    assert res_over["is_stale"] is True

    # 4. Closed / merged PR above threshold (200h) -> NEVER stale
    res_closed = check_stale(is_open=False, hours_since_last_activity=200.0, threshold_hours=120.0)
    assert res_closed["is_stale"] is False


def test_calculate_hours_since_activity() -> None:
    now = datetime(2025, 6, 10, 12, 0, 0, tzinfo=UTC)
    pr_row = {
        "updated_at": datetime(2025, 6, 5, 12, 0, 0, tzinfo=UTC),  # 5 days = 120 hours ago
        "last_synced_at": datetime(2025, 6, 4, 12, 0, 0, tzinfo=UTC),
        "opened_at": datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC),
        "raw_data": {
            "comments": [
                {"createdAt": "2025-06-08T12:00:00Z"},  # 2 days = 48 hours ago
            ]
        },
    }

    hours = calculate_hours_since_activity(pr_row, now=now)
    assert round(hours, 1) == 48.0
