"""Persist idempotent PR-risk features and predictions into API-owned tables."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, text

from app.risk.features import (
    FEATURE_ORDER,
    FEATURE_SCHEMA_VERSION,
    FILE_SCOPE,
    PullRequestRiskFeatures,
    extract_pull_request_features,
)
from app.risk.model import MODEL_NAME, MODEL_VERSION, JitFineRiskModel, risk_model


def calculate_and_persist_pull_request_risk(
    database_engine: Engine,
    workspace_id: UUID,
    repository_id: UUID,
    pull_request_id: UUID,
    pull_request: dict[str, Any],
    files: list[dict[str, Any]],
    commits: list[dict[str, Any]],
    *,
    model: JitFineRiskModel = risk_model,
) -> None:
    """Extract, predict, and atomically upsert one current model result."""
    features = extract_pull_request_features(pull_request, files, commits)
    prediction = model.predict(features)
    extracted_at = datetime.now(UTC)
    payload = _feature_payload(features)

    with database_engine.begin() as connection:
        feature_id = connection.execute(
            text(
                """
                INSERT INTO pull_request_features (
                    workspace_id, repository_id, pull_request_id,
                    feature_schema_version, lines_added, lines_deleted,
                    files_changed, commit_count, author_prior_pr_count,
                    entropy, feature_payload, extracted_at
                ) VALUES (
                    :workspace_id, :repository_id, :pull_request_id,
                    :feature_schema_version, :lines_added, :lines_deleted,
                    :files_changed, :commit_count, 0,
                    :entropy, CAST(:feature_payload AS jsonb), :extracted_at
                )
                ON CONFLICT (pull_request_id, feature_schema_version)
                DO UPDATE SET
                    workspace_id = EXCLUDED.workspace_id,
                    repository_id = EXCLUDED.repository_id,
                    lines_added = EXCLUDED.lines_added,
                    lines_deleted = EXCLUDED.lines_deleted,
                    files_changed = EXCLUDED.files_changed,
                    commit_count = EXCLUDED.commit_count,
                    entropy = EXCLUDED.entropy,
                    feature_payload = EXCLUDED.feature_payload,
                    extracted_at = EXCLUDED.extracted_at,
                    updated_at = now(),
                    version = pull_request_features.version + 1
                RETURNING id
                """
            ),
            {
                "workspace_id": workspace_id,
                "repository_id": repository_id,
                "pull_request_id": pull_request_id,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "lines_added": features.la,
                "lines_deleted": features.ld,
                "files_changed": features.nf,
                "commit_count": len(commits),
                "entropy": features.entropy,
                "feature_payload": json.dumps(payload),
                "extracted_at": extracted_at,
            },
        ).scalar_one()

        connection.execute(
            text(
                """
                INSERT INTO risk_predictions (
                    workspace_id, repository_id, pull_request_id, feature_id,
                    model_name, model_version, risk_score, risk_level,
                    threshold_used, top_factors, predicted_at
                ) VALUES (
                    :workspace_id, :repository_id, :pull_request_id, :feature_id,
                    :model_name, :model_version, :risk_score, :risk_level,
                    :threshold_used, CAST(:top_factors AS jsonb), :predicted_at
                )
                ON CONFLICT (pull_request_id, model_name, model_version)
                DO UPDATE SET
                    workspace_id = EXCLUDED.workspace_id,
                    repository_id = EXCLUDED.repository_id,
                    feature_id = EXCLUDED.feature_id,
                    risk_score = EXCLUDED.risk_score,
                    risk_level = EXCLUDED.risk_level,
                    threshold_used = EXCLUDED.threshold_used,
                    top_factors = EXCLUDED.top_factors,
                    predicted_at = EXCLUDED.predicted_at,
                    updated_at = now(),
                    version = risk_predictions.version + 1
                """
            ),
            {
                "workspace_id": workspace_id,
                "repository_id": repository_id,
                "pull_request_id": pull_request_id,
                "feature_id": feature_id,
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "risk_score": prediction.probability,
                "risk_level": prediction.level,
                "threshold_used": prediction.threshold_used,
                "top_factors": json.dumps(prediction.top_factors),
                "predicted_at": extracted_at,
            },
        )

        from app.jobs.handlers.evaluate_alerts import enqueue_evaluate_alerts_job

        enqueue_evaluate_alerts_job(
            connection,
            workspace_id=workspace_id,
            repository_id=repository_id,
            trigger_source="RISK_PREDICTION",
        )


def _feature_payload(features: PullRequestRiskFeatures) -> dict[str, Any]:
    return {
        "featureOrder": list(FEATURE_ORDER),
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "predictionUnit": "pull_request_aggregate",
        "fileScope": FILE_SCOPE,
        "fixRule": "pull_request_title_body_or_commit_message_keyword",
        "values": features.as_dict(),
    }
