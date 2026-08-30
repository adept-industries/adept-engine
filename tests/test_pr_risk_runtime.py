from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, text

from app.jobs.handlers import backfill_repository
from app.normalization.pull_requests import upsert_pull_request
from app.providers.github import ProviderPage
from app.risk.features import (
    FEATURE_ORDER,
    FEATURE_SCHEMA_VERSION,
    PullRequestRiskFeatures,
    RiskFeatureUnavailable,
    extract_pull_request_features,
)
from app.risk.model import (
    MODEL_NAME,
    MODEL_VERSION,
    JitFineRiskModel,
    RiskPredictionResult,
    _classify_probability,
)
from app.risk.service import calculate_and_persist_pull_request_risk
from ml_training.src.constants import (
    FEATURE_ORDER as TRAINING_FEATURE_ORDER,
)
from ml_training.src.constants import (
    FEATURE_SCHEMA_VERSION as TRAINING_FEATURE_SCHEMA_VERSION,
)
from ml_training.src.constants import (
    MODEL_NAME as TRAINING_MODEL_NAME,
)
from ml_training.src.constants import (
    MODEL_VERSION as TRAINING_MODEL_VERSION,
)


def test_extracts_the_frozen_seven_features_from_complete_github_data() -> None:
    pull_request = {
        "changed_files": 3,
        "additions": 40,
        "deletions": 0,
        "title": "Refactor storage",
        "body": None,
    }
    files = [
        {"filename": "src/api.py", "additions": 10, "deletions": 0},
        {"filename": "src/db/store.py", "additions": 30, "deletions": 0},
        {"filename": "README.md", "additions": 0, "deletions": 0},
    ]
    commits = [{"commit": {"message": "Fix storage race"}}]

    features = extract_pull_request_features(pull_request, files, commits)

    assert FEATURE_ORDER == ("ns", "nd", "nf", "entropy", "la", "ld", "fix")
    assert features.as_dict() == {
        "ns": 2,
        "nd": 3,
        "nf": 3,
        "entropy": pytest.approx(0.8112781244591328),
        "la": 40,
        "ld": 0,
        "fix": 1,
    }
    assert features.as_ordered_values() == pytest.approx(
        (2.0, 3.0, 3.0, 0.8112781244591328, 40.0, 0.0, 1.0)
    )


def test_training_runtime_and_api_owned_identity_cannot_drift() -> None:
    assert FEATURE_ORDER == TRAINING_FEATURE_ORDER
    assert FEATURE_SCHEMA_VERSION == TRAINING_FEATURE_SCHEMA_VERSION
    assert MODEL_NAME == TRAINING_MODEL_NAME
    assert MODEL_VERSION == TRAINING_MODEL_VERSION


@pytest.mark.parametrize(
    ("pull_request", "files", "message"),
    [
        (
            {"changed_files": 2, "additions": 1, "deletions": 0},
            [{"filename": "a.py", "additions": 1, "deletions": 0}],
            "returned 1 files",
        ),
        (
            {"changed_files": 3_001, "additions": 0, "deletions": 0},
            [],
            "caps pull-request file results",
        ),
        (
            {"changed_files": 1, "additions": 2, "deletions": 0},
            [{"filename": "a.py", "additions": 1, "deletions": 0}],
            "totals changed",
        ),
    ],
)
def test_declines_to_score_incomplete_github_file_data(
    pull_request: dict[str, object],
    files: list[dict[str, object]],
    message: str,
) -> None:
    with pytest.raises(RiskFeatureUnavailable, match=message):
        extract_pull_request_features(pull_request, files, [])


def test_zero_churn_is_finite_and_fix_keyword_matching_is_bounded() -> None:
    features = extract_pull_request_features(
        {
            "changed_files": 1,
            "additions": 0,
            "deletions": 0,
            "title": "Update fixture",
            "body": "No behavioural change",
        },
        [{"filename": "fixture.json", "additions": 0, "deletions": 0}],
        [],
    )

    assert features.entropy == 0.0
    assert math.isfinite(features.entropy)
    assert features.fix == 0


def test_approved_artifact_loads_and_predicts_in_the_frozen_order() -> None:
    model = JitFineRiskModel()
    model.load()

    result = model.predict(PullRequestRiskFeatures(2, 3, 4, 1.5, 100, 20, 0))

    assert model.ready
    assert model.metadata is not None
    assert model.metadata["modelName"] == MODEL_NAME
    assert model.metadata["modelVersion"] == MODEL_VERSION
    assert model.metadata["featureSchemaVersion"] == FEATURE_SCHEMA_VERSION
    assert model.metadata["featureOrder"] == list(FEATURE_ORDER)
    assert 0.0 <= result.probability <= 1.0
    assert result.level in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert len(result.top_factors) == 3
    assert all(item["explanationType"] == "global_model_importance" for item in result.top_factors)


def test_probability_thresholds_use_all_four_database_risk_levels() -> None:
    thresholds = {"medium": 0.10, "high": 0.20, "critical": 0.40}

    assert _classify_probability(0.09, thresholds) == ("LOW", 0.10)
    assert _classify_probability(0.10, thresholds) == ("MEDIUM", 0.10)
    assert _classify_probability(0.20, thresholds) == ("HIGH", 0.20)
    assert _classify_probability(0.40, thresholds) == ("CRITICAL", 0.40)


def test_risk_only_backfill_scores_all_open_pull_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = SimpleNamespace(
        id=uuid4(),
        workspace_id=uuid4(),
        owner_login="adept-industries",
        name="adept-engine",
    )
    pull_request_id = uuid4()
    pull_request = {
        "id": 100,
        "number": 42,
        "state": "open",
        "changed_files": 1,
        "additions": 5,
        "deletions": 1,
    }
    commits = [{"sha": "abc", "commit": {"message": "Change runtime"}}]
    files = [{"filename": "app/main.py", "additions": 5, "deletions": 1}]
    client = MagicMock()
    client.list_open_pull_requests.return_value = ProviderPage([{"number": 42}], None)
    client.get_pull_request.return_value = pull_request
    client.list_pull_request_commits.return_value = commits
    client.list_pull_request_files.return_value = files
    score = MagicMock()
    monkeypatch.setattr(
        backfill_repository,
        "upsert_pull_request",
        MagicMock(return_value=pull_request_id),
    )
    monkeypatch.setattr(backfill_repository, "calculate_and_persist_pull_request_risk", score)

    next_cursor, count = backfill_repository._process_open_pull_request_page(
        MagicMock(),
        client,
        repository,
        1,
        risk_only=True,
    )

    assert next_cursor is None
    assert count == 1
    score.assert_called_once()
    assert score.call_args.args[4:] == (pull_request, files, commits)


@dataclass(frozen=True, slots=True)
class RiskRows:
    user_id: UUID
    workspace_id: UUID
    repository_id: UUID
    pull_request_id: UUID


@pytest.fixture
def risk_rows(database_engine: Engine) -> Iterator[RiskRows]:
    user_id = uuid4()
    workspace_id = uuid4()
    integration_id = uuid4()
    repository_id = uuid4()
    now = datetime.now(UTC).replace(microsecond=0)
    with database_engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO users (id, email, password_hash, display_name)
                VALUES (:id, :email, 'hash', 'Risk Test')
                """
            ),
            {"id": user_id, "email": f"risk-{workspace_id.hex}@example.test"},
        )
        connection.execute(
            text(
                """
                INSERT INTO workspaces (id, name, slug, timezone)
                VALUES (:id, 'Risk Runtime', :slug, 'UTC')
                """
            ),
            {"id": workspace_id, "slug": f"risk-{workspace_id.hex}"},
        )
        connection.execute(
            text(
                """
                INSERT INTO github_integrations (
                    id, workspace_id, installation_id, account_external_id,
                    account_login, account_type, repository_selection, status
                ) VALUES (
                    :id, :workspace_id, 7100, 8100,
                    'adept-industries', 'ORGANIZATION', 'ALL', 'ACTIVE'
                )
                """
            ),
            {"id": integration_id, "workspace_id": workspace_id},
        )
        connection.execute(
            text(
                """
                INSERT INTO repositories (
                    id, workspace_id, github_integration_id, github_repo_id,
                    owner_login, name, full_name, default_branch, visibility,
                    tracking_enabled
                ) VALUES (
                    :id, :workspace_id, :integration_id, 9100,
                    'adept-industries', 'risk-runtime', 'adept-industries/risk-runtime',
                    'main', 'PRIVATE', true
                )
                """
            ),
            {
                "id": repository_id,
                "workspace_id": workspace_id,
                "integration_id": integration_id,
            },
        )

    pull_request_id = upsert_pull_request(
        database_engine,
        workspace_id,
        repository_id,
        {
            "id": 5100,
            "number": 51,
            "title": "Fix runtime",
            "state": "open",
            "merged": False,
            "draft": False,
            "user": {"login": "developer"},
            "base": {"ref": "main"},
            "head": {"ref": "feature/risk", "sha": "abc123"},
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "additions": 10,
            "deletions": 2,
            "changed_files": 1,
            "commits": 1,
        },
        "opened",
        [],
    )
    yield RiskRows(user_id, workspace_id, repository_id, pull_request_id)

    with database_engine.begin() as connection:
        connection.execute(text("DELETE FROM workspaces WHERE id = :id"), {"id": workspace_id})
        connection.execute(text("DELETE FROM users WHERE id = :id"), {"id": user_id})


class _FixedModel:
    def predict(self, _features: PullRequestRiskFeatures) -> RiskPredictionResult:
        return RiskPredictionResult(
            probability=0.25,
            level="HIGH",
            threshold_used=0.17,
            top_factors=[
                {
                    "feature": "la",
                    "value": 10,
                    "globalImportance": 0.2,
                    "explanationType": "global_model_importance",
                }
            ],
        )


@pytest.mark.integration
def test_feature_and_prediction_upserts_are_atomic_versioned_and_idempotent(
    database_engine: Engine,
    risk_rows: RiskRows,
) -> None:
    model = cast(JitFineRiskModel, _FixedModel())
    pull_request = {
        "changed_files": 1,
        "additions": 10,
        "deletions": 2,
        "title": "Fix runtime",
        "body": None,
    }
    files = [{"filename": "app/runtime.py", "additions": 10, "deletions": 2}]
    commits = [{"commit": {"message": "Fix runtime"}}]

    calculate_and_persist_pull_request_risk(
        database_engine,
        risk_rows.workspace_id,
        risk_rows.repository_id,
        risk_rows.pull_request_id,
        pull_request,
        files,
        commits,
        model=model,
    )
    pull_request["additions"] = 12
    files[0]["additions"] = 12
    calculate_and_persist_pull_request_risk(
        database_engine,
        risk_rows.workspace_id,
        risk_rows.repository_id,
        risk_rows.pull_request_id,
        pull_request,
        files,
        commits,
        model=model,
    )

    with database_engine.connect() as connection:
        feature = (
            connection.execute(
                text(
                    """
                    SELECT id, feature_schema_version, lines_added, lines_deleted,
                           files_changed, feature_payload, version
                    FROM pull_request_features
                    WHERE pull_request_id = :pull_request_id
                    """
                ),
                {"pull_request_id": risk_rows.pull_request_id},
            )
            .mappings()
            .one()
        )
        prediction = (
            connection.execute(
                text(
                    """
                    SELECT feature_id, model_name, model_version, risk_score,
                           risk_level, threshold_used, top_factors, version
                    FROM risk_predictions
                    WHERE pull_request_id = :pull_request_id
                    """
                ),
                {"pull_request_id": risk_rows.pull_request_id},
            )
            .mappings()
            .one()
        )

    assert feature["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert feature["lines_added"] == 12
    assert feature["lines_deleted"] == 2
    assert feature["files_changed"] == 1
    assert feature["feature_payload"]["featureOrder"] == list(FEATURE_ORDER)
    assert feature["feature_payload"]["fileScope"] == "all_changed_files_reported_by_github"
    assert feature["feature_payload"]["values"]["la"] == 12
    assert feature["version"] == 1
    assert prediction["feature_id"] == feature["id"]
    assert prediction["model_name"] == MODEL_NAME
    assert prediction["model_version"] == MODEL_VERSION
    assert prediction["risk_score"] == Decimal("0.250000")
    assert prediction["risk_level"] == "HIGH"
    assert prediction["threshold_used"] == Decimal("0.170000")
    assert prediction["top_factors"][0]["explanationType"] == "global_model_importance"
    assert prediction["version"] == 1
