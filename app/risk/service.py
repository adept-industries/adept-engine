"""PR Risk Analytics database integration service."""

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import Engine, text

from app.core.config import get_settings
from app.risk.dx import compute_dx_score
from app.risk.features import (
    FEATURE_SCHEMA_VERSION,
    extract_features_from_pr_record,
    validate_feature_dict,
)
from app.risk.model import risk_model
from app.risk.outcomes import DEFAULT_OBSERVATION_WINDOW_DAYS
from app.risk.stale import calculate_hours_since_activity, check_stale

logger = structlog.get_logger()


def _resolve_repo_and_pr(
    database_engine: Engine,
    repo_identifier: str | UUID,
    pr_number: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolves repository and pull_request database rows."""
    with database_engine.connect() as conn:
        # Check if repo_identifier is UUID or string
        try:
            repo_uuid = UUID(str(repo_identifier))
            repo_row = (
                conn.execute(
                    text("SELECT * FROM repositories WHERE id = :id"),
                    {"id": repo_uuid},
                )
                .mappings()
                .one_or_none()
            )
        except ValueError:
            repo_row = (
                conn.execute(
                    text("SELECT * FROM repositories WHERE full_name = :id OR name = :id"),
                    {"id": str(repo_identifier)},
                )
                .mappings()
                .one_or_none()
            )

        if repo_row is None:
            raise ValueError(f"Repository not found: {repo_identifier}")

        pr_row = (
            conn.execute(
                text(
                    """
                SELECT * FROM pull_requests
                WHERE repository_id = :repo_id AND number = :pr_number
                """
                ),
                {"repo_id": repo_row["id"], "pr_number": pr_number},
            )
            .mappings()
            .one_or_none()
        )

        if pr_row is None:
            raise ValueError(
                f"Pull request #{pr_number} not found for repository {repo_row['full_name']}"
            )

        return dict(repo_row), dict(pr_row)


def extract_and_persist_snapshot(
    database_engine: Engine,
    repo_identifier: str | UUID,
    pr_number: int,
    snapshot_at: datetime | None = None,
    stage: str = "live",
) -> dict[str, Any]:
    """Extracts leakage-safe features for a PR and persists a feature snapshot."""
    repo_row, pr_row = _resolve_repo_and_pr(database_engine, repo_identifier, pr_number)
    eff_snapshot_at = snapshot_at or datetime.now(UTC)
    if eff_snapshot_at.tzinfo is None:
        eff_snapshot_at = eff_snapshot_at.replace(tzinfo=UTC)

    # Compute repository historical context prior to snapshot_at
    historical_context = _compute_historical_context(
        database_engine, repo_row["id"], pr_row["author_login"], eff_snapshot_at
    )

    features = extract_features_from_pr_record(
        pr_row, historical_context=historical_context, snapshot_at=eff_snapshot_at
    )

    with database_engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO pr_feature_snapshots (
                    workspace_id, repository_id, pull_request_id, pr_number,
                    snapshot_at, stage, feature_schema_version, features
                ) VALUES (
                    :workspace_id, :repository_id, :pull_request_id, :pr_number,
                    :snapshot_at, :stage, :feature_schema_version, CAST(:features AS jsonb)
                ) ON CONFLICT (repository_id, pull_request_id, snapshot_at, stage)
                DO UPDATE SET
                    features = EXCLUDED.features,
                    updated_at = now()
                """
            ),
            {
                "workspace_id": pr_row["workspace_id"],
                "repository_id": repo_row["id"],
                "pull_request_id": pr_row["id"],
                "pr_number": pr_number,
                "snapshot_at": eff_snapshot_at,
                "stage": stage,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "features": json.dumps(features),
            },
        )

    return {
        "repository_id": str(repo_row["id"]),
        "pr_number": pr_number,
        "snapshot_at": eff_snapshot_at.isoformat(),
        "stage": stage,
        "features": features,
    }


def predict_and_persist(
    database_engine: Engine,
    repo_identifier: str | UUID,
    pr_number: int,
    custom_features: dict[str, Any] | None = None,
    stage: str = "live",
    snapshot_at: datetime | None = None,
) -> dict[str, Any]:
    """Runs model inference on PR features and persists the prediction."""
    repo_row, pr_row = _resolve_repo_and_pr(database_engine, repo_identifier, pr_number)
    eff_snapshot_at = snapshot_at or datetime.now(UTC)
    if eff_snapshot_at.tzinfo is None:
        eff_snapshot_at = eff_snapshot_at.replace(tzinfo=UTC)

    if custom_features is not None:
        features = validate_feature_dict(custom_features)
    else:
        historical_context = _compute_historical_context(
            database_engine, repo_row["id"], pr_row["author_login"], eff_snapshot_at
        )
        features = extract_features_from_pr_record(
            pr_row, historical_context=historical_context, snapshot_at=eff_snapshot_at
        )

    # Run ML prediction
    prediction = risk_model.predict(features)
    predicted_at = datetime.now(UTC)

    # Persist snapshot and risk prediction
    with database_engine.begin() as conn:
        # Snapshot
        conn.execute(
            text(
                """
                INSERT INTO pr_feature_snapshots (
                    workspace_id, repository_id, pull_request_id, pr_number,
                    snapshot_at, stage, feature_schema_version, features
                ) VALUES (
                    :workspace_id, :repository_id, :pull_request_id, :pr_number,
                    :snapshot_at, :stage, :feature_schema_version, CAST(:features AS jsonb)
                ) ON CONFLICT (repository_id, pull_request_id, snapshot_at, stage)
                DO UPDATE SET
                    features = EXCLUDED.features,
                    updated_at = now()
                """
            ),
            {
                "workspace_id": pr_row["workspace_id"],
                "repository_id": repo_row["id"],
                "pull_request_id": pr_row["id"],
                "pr_number": pr_number,
                "snapshot_at": eff_snapshot_at,
                "stage": stage,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "features": json.dumps(features),
            },
        )

        # Risk Prediction
        meta = risk_model.metadata or {}
        high_thresh = meta.get("high_threshold", 0.30)
        conn.execute(
            text(
                """
                INSERT INTO risk_predictions (
                    workspace_id, repository_id, pull_request_id,
                    model_name, model_version, risk_score, risk_level,
                    threshold_used, top_factors, predicted_at, stage
                ) VALUES (
                    :workspace_id, :repository_id, :pull_request_id,
                    :model_name, :model_version, :risk_score, :risk_level,
                    :threshold_used, CAST(:top_factors AS jsonb), :predicted_at, :stage
                ) ON CONFLICT (pull_request_id, model_name, model_version)
                DO UPDATE SET
                    risk_score = EXCLUDED.risk_score,
                    risk_level = EXCLUDED.risk_level,
                    threshold_used = EXCLUDED.threshold_used,
                    top_factors = EXCLUDED.top_factors,
                    predicted_at = EXCLUDED.predicted_at,
                    stage = EXCLUDED.stage,
                    updated_at = now()
                """
            ),
            {
                "workspace_id": pr_row["workspace_id"],
                "repository_id": repo_row["id"],
                "pull_request_id": pr_row["id"],
                "model_name": "pr-code-change-risk-xgb",
                "model_version": prediction["model_version"],
                "risk_score": prediction["probability"],
                "risk_level": prediction["risk_level"],
                "threshold_used": high_thresh,
                "top_factors": json.dumps(prediction["top_factors"]),
                "predicted_at": predicted_at,
                "stage": stage,
            },
        )

    return {
        "repository_id": str(repo_row["id"]),
        "pr_number": pr_number,
        "risk_probability": prediction["probability"],
        "risk_level": prediction["risk_level"],
        "model_version": prediction["model_version"],
        "top_factors": prediction["top_factors"],
        "predicted_at": predicted_at.isoformat(),
        "stage": stage,
        "features": features,
    }


def get_latest_risk_for_pr(
    database_engine: Engine,
    repo_identifier: str | UUID,
    pr_number: int,
) -> dict[str, Any] | None:
    """Retrieves the latest risk prediction for a pull request."""
    repo_row, pr_row = _resolve_repo_and_pr(database_engine, repo_identifier, pr_number)

    with database_engine.connect() as conn:
        pred_row = (
            conn.execute(
                text(
                    """
                SELECT * FROM risk_predictions
                WHERE pull_request_id = :pr_id
                ORDER BY predicted_at DESC
                LIMIT 1
                """
                ),
                {"pr_id": pr_row["id"]},
            )
            .mappings()
            .one_or_none()
        )

        if pred_row is None:
            return None

        # Fetch latest feature snapshot
        snap_row = (
            conn.execute(
                text(
                    """
                SELECT * FROM pr_feature_snapshots
                WHERE pull_request_id = :pr_id
                ORDER BY snapshot_at DESC
                LIMIT 1
                """
                ),
                {"pr_id": pr_row["id"]},
            )
            .mappings()
            .one_or_none()
        )

    return {
        "repository_id": str(repo_row["id"]),
        "pr_number": pr_number,
        "title": pr_row["title"],
        "state": pr_row["state"],
        "risk_probability": float(pred_row["risk_score"]),
        "risk_level": pred_row["risk_level"],
        "model_version": pred_row["model_version"],
        "threshold_used": float(pred_row["threshold_used"]) if pred_row["threshold_used"] else None,
        "top_factors": pred_row["top_factors"],
        "predicted_at": pred_row["predicted_at"].isoformat(),
        "stage": pred_row.get("stage", "live"),
        "features": snap_row["features"] if snap_row else {},
    }


def list_risky_prs_for_repository(
    database_engine: Engine,
    repo_identifier: str | UUID,
    min_level: str = "MEDIUM",
) -> list[dict[str, Any]]:
    """Lists current open pull requests with risk level matching or exceeding min_level."""
    with database_engine.connect() as conn:
        try:
            repo_uuid = UUID(str(repo_identifier))
            repo_row = (
                conn.execute(
                    text("SELECT * FROM repositories WHERE id = :id"),
                    {"id": repo_uuid},
                )
                .mappings()
                .one_or_none()
            )
        except ValueError:
            repo_row = (
                conn.execute(
                    text("SELECT * FROM repositories WHERE full_name = :id OR name = :id"),
                    {"id": str(repo_identifier)},
                )
                .mappings()
                .one_or_none()
            )

        if repo_row is None:
            raise ValueError(f"Repository not found: {repo_identifier}")

        allowed_levels = (
            ["HIGH", "CRITICAL"] if min_level.upper() == "HIGH" else ["MEDIUM", "HIGH", "CRITICAL"]
        )

        rows = (
            conn.execute(
                text(
                    """
                SELECT DISTINCT ON (p.id)
                    p.id AS pr_id,
                    p.number,
                    p.title,
                    p.author_login,
                    p.state,
                    p.opened_at,
                    r.risk_score,
                    r.risk_level,
                    r.top_factors,
                    r.predicted_at,
                    r.model_version
                FROM pull_requests p
                JOIN risk_predictions r ON p.id = r.pull_request_id
                WHERE p.repository_id = :repo_id
                  AND p.state = 'OPEN'
                  AND r.risk_level = ANY(:levels)
                ORDER BY p.id, r.predicted_at DESC
                """
                ),
                {"repo_id": repo_row["id"], "levels": allowed_levels},
            )
            .mappings()
            .all()
        )

    return [
        {
            "pr_number": row["number"],
            "title": row["title"],
            "author_login": row["author_login"],
            "risk_score": float(row["risk_score"]),
            "risk_level": row["risk_level"],
            "top_factors": row["top_factors"],
            "predicted_at": row["predicted_at"].isoformat(),
            "model_version": row["model_version"],
        }
        for row in rows
    ]


def check_stale_for_pr(
    database_engine: Engine,
    repo_identifier: str | UUID,
    pr_number: int,
    threshold_hours: float | None = None,
) -> dict[str, Any]:
    """Evaluates whether an open PR is stale."""
    repo_row, pr_row = _resolve_repo_and_pr(database_engine, repo_identifier, pr_number)
    settings = get_settings()
    thresh = threshold_hours or float(settings.stale_pr_hours_threshold)

    hours = calculate_hours_since_activity(pr_row)
    is_open = pr_row["state"] == "OPEN"
    result = check_stale(is_open=is_open, hours_since_last_activity=hours, threshold_hours=thresh)

    return {
        "repository_id": str(repo_row["id"]),
        "pr_number": pr_number,
        "is_open": is_open,
        **result,
    }


def record_outcome(
    database_engine: Engine,
    repo_identifier: str | UUID,
    pr_number: int,
    is_risky: bool,
    reason: str,
    evidence: dict[str, Any] | None = None,
    merged_at: datetime | None = None,
    observed_until: datetime | None = None,
) -> dict[str, Any]:
    """Records an adverse outcome or clean observation for a merged PR."""
    repo_row, pr_row = _resolve_repo_and_pr(database_engine, repo_identifier, pr_number)
    m_at = merged_at or pr_row.get("merged_at")
    obs_until = observed_until or (
        (m_at or datetime.now(UTC)) + timedelta(days=DEFAULT_OBSERVATION_WINDOW_DAYS)
    )

    with database_engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO pr_outcomes (
                    workspace_id, repository_id, pull_request_id, pr_number,
                    merged_at, observed_until, is_risky, reason, evidence
                ) VALUES (
                    :workspace_id, :repository_id, :pull_request_id, :pr_number,
                    :merged_at, :observed_until, :is_risky, :reason, CAST(:evidence AS jsonb)
                ) ON CONFLICT (repository_id, pull_request_id)
                DO UPDATE SET
                    merged_at = EXCLUDED.merged_at,
                    observed_until = EXCLUDED.observed_until,
                    is_risky = EXCLUDED.is_risky,
                    reason = EXCLUDED.reason,
                    evidence = EXCLUDED.evidence,
                    updated_at = now()
                """
            ),
            {
                "workspace_id": pr_row["workspace_id"],
                "repository_id": repo_row["id"],
                "pull_request_id": pr_row["id"],
                "pr_number": pr_number,
                "merged_at": m_at,
                "observed_until": obs_until,
                "is_risky": is_risky,
                "reason": reason,
                "evidence": json.dumps(evidence or {}),
            },
        )

    return {
        "status": "saved",
        "repository_id": str(repo_row["id"]),
        "pr_number": pr_number,
        "is_risky": is_risky,
        "reason": reason,
    }


def compute_repo_dx_score(
    database_engine: Engine,
    repo_identifier: str | UUID,
    custom_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Calculates repository Developer Experience score from database statistics."""
    with database_engine.connect() as conn:
        try:
            repo_uuid = UUID(str(repo_identifier))
            repo_row = (
                conn.execute(
                    text("SELECT * FROM repositories WHERE id = :id"),
                    {"id": repo_uuid},
                )
                .mappings()
                .one_or_none()
            )
        except ValueError:
            repo_row = (
                conn.execute(
                    text("SELECT * FROM repositories WHERE full_name = :id OR name = :id"),
                    {"id": str(repo_identifier)},
                )
                .mappings()
                .one_or_none()
            )

        if repo_row is None:
            raise ValueError(f"Repository not found: {repo_identifier}")

        # Compute cycle times and review wait from pull_requests
        prs = (
            conn.execute(
                text("SELECT * FROM pull_requests WHERE repository_id = :repo_id"),
                {"repo_id": repo_row["id"]},
            )
            .mappings()
            .all()
        )

        # Compute deployments
        deps = (
            conn.execute(
                text("SELECT * FROM deployments WHERE repository_id = :repo_id"),
                {"repo_id": repo_row["id"]},
            )
            .mappings()
            .all()
        )

    cycle_hours_list = []
    stale_count = 0
    open_count = 0

    for pr in prs:
        if pr["state"] == "OPEN":
            open_count += 1
            if calculate_hours_since_activity(dict(pr)) >= 120.0:
                stale_count += 1
        elif pr["state"] == "MERGED" and pr["merged_at"] and pr["opened_at"]:
            delta = (pr["merged_at"] - pr["opened_at"]).total_seconds() / 3600.0
            if delta > 0:
                cycle_hours_list.append(delta)

    median_cycle = (
        float(sorted(cycle_hours_list)[len(cycle_hours_list) // 2]) if cycle_hours_list else 48.0
    )
    stale_rate = float(stale_count / max(1, open_count)) if open_count > 0 else 0.05

    # Deployment failure rate
    failed_deps = sum(1 for d in deps if d["status"] == "FAILURE")
    total_deps = len(deps)
    cfr = float(failed_deps / max(1, total_deps)) if total_deps > 0 else 0.08
    ci_success = float(1.0 - cfr)

    dx_result = compute_dx_score(
        median_first_review_hours=8.0,
        median_pr_cycle_hours=median_cycle,
        stale_pr_rate=stale_rate,
        change_failure_rate=cfr,
        ci_success_rate=ci_success,
        custom_weights=custom_weights,
    )

    return {
        "repository_id": str(repo_row["id"]),
        "repository_name": repo_row["full_name"],
        **dx_result,
    }


def _compute_historical_context(
    database_engine: Engine,
    repository_id: UUID,
    author_login: str | None,
    snapshot_at: datetime,
) -> dict[str, Any]:
    """Computes leakage-safe author experience and repository hotspot metrics before snapshot_at."""
    with database_engine.connect() as conn:
        # Prior PR count by author in this repository strictly before snapshot_at
        author_pr_count = 0
        if author_login:
            author_pr_count = (
                conn.execute(
                    text(
                        """
                    SELECT count(*) FROM pull_requests
                    WHERE repository_id = :repo_id
                      AND author_login = :author
                      AND opened_at < :snapshot_at
                    """
                    ),
                    {
                        "repo_id": repository_id,
                        "author": author_login,
                        "snapshot_at": snapshot_at,
                    },
                ).scalar()
                or 0
            )

        # Repository total PRs before snapshot_at
        total_repo_prs = (
            conn.execute(
                text(
                    """
                SELECT count(*) FROM pull_requests
                WHERE repository_id = :repo_id
                  AND opened_at < :snapshot_at
                """
                ),
                {"repo_id": repository_id, "snapshot_at": snapshot_at},
            ).scalar()
            or 0
        )

    author_exp = min(1.0, float(author_pr_count) / 20.0)
    hotspot = min(1.0, float(total_repo_prs) / 100.0)

    return {
        "author_repo_experience": round(author_exp, 4),
        "author_file_familiarity": round(min(1.0, author_exp * 1.2), 4),
        "hotspot_score": round(hotspot, 4),
        "recent_file_bugfix_rate": 0.08,
        "recent_file_change_rate": round(min(1.0, hotspot * 0.8), 4),
        "ci_failures": 0,
    }
