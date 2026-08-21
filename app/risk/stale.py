"""Deterministic stale pull request detection.

CRITICAL RULE:
Stale PR detection is non-ML and deterministic:
`is_open && hours_since_last_activity >= configured_threshold`
Default threshold is 120 hours (5 days).
"""

from datetime import UTC, datetime
from typing import Any

DEFAULT_STALE_HOURS_THRESHOLD = 120.0


def calculate_hours_since_activity(
    pr_row: dict[str, Any],
    now: datetime | None = None,
) -> float:
    """Calculates hours since the most recent activity on this pull request."""
    reference_now = now or datetime.now(UTC)
    if reference_now.tzinfo is None:
        reference_now = reference_now.replace(tzinfo=UTC)

    activity_timestamps: list[datetime] = []

    # 1. Base PR timestamps
    for field in ("updated_at", "last_synced_at", "opened_at"):
        val = pr_row.get(field)
        if val:
            dt = datetime.fromisoformat(val.replace("Z", "+00:00")) if isinstance(val, str) else val
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            activity_timestamps.append(dt)

    raw_data = pr_row.get("raw_data") or {}

    # 2. Latest commit timestamp
    for c in raw_data.get("commits") or []:
        c_date_str = c.get("committerDate") or c.get("authorDate")
        if c_date_str:
            try:
                dt = datetime.fromisoformat(c_date_str.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                activity_timestamps.append(dt)
            except Exception:
                pass

    # 3. Latest review timestamp
    for r in raw_data.get("reviews") or []:
        r_date_str = r.get("submittedAt")
        if r_date_str:
            try:
                dt = datetime.fromisoformat(r_date_str.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                activity_timestamps.append(dt)
            except Exception:
                pass

    # 4. Latest comment timestamp
    for cm in raw_data.get("comments") or []:
        cm_date_str = cm.get("updatedAt") or cm.get("createdAt")
        if cm_date_str:
            try:
                dt = datetime.fromisoformat(cm_date_str.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                activity_timestamps.append(dt)
            except Exception:
                pass

    if not activity_timestamps:
        return 0.0

    latest_activity = max(activity_timestamps)
    delta_seconds = (reference_now - latest_activity).total_seconds()
    return max(0.0, delta_seconds / 3600.0)


def check_stale(
    is_open: bool,
    hours_since_last_activity: float,
    threshold_hours: float = DEFAULT_STALE_HOURS_THRESHOLD,
) -> dict[str, Any]:
    """Determines whether an open PR exceeds the inactivity threshold."""
    is_stale = is_open and (hours_since_last_activity >= threshold_hours)
    reason = (
        f"Open PR inactive for {hours_since_last_activity:.1f}h (threshold: {threshold_hours:.1f}h)"
        if is_stale
        else "PR is active or closed"
    )
    return {
        "is_stale": is_stale,
        "hours_since_last_activity": round(hours_since_last_activity, 2),
        "threshold_hours": threshold_hours,
        "reason": reason,
    }
