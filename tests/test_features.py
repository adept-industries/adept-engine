"""Tests for PR risk feature schema, extraction, and leakage prevention."""

from datetime import UTC, datetime

import pytest

from app.risk.features import (
    FEATURE_SCHEMA_VERSION,
    RISK_FEATURES,
    classify_file,
    extract_features_from_pr_record,
    risk_level,
    validate_feature_dict,
)


def test_feature_schema_consistency() -> None:
    """Verifies that canonical RISK_FEATURES is consistent and matches expected schema."""
    assert len(RISK_FEATURES) == 16
    assert "lines_added" in RISK_FEATURES
    assert "lines_deleted" in RISK_FEATURES
    assert "files_changed" in RISK_FEATURES
    assert "commit_count" in RISK_FEATURES
    assert "source_files_changed" in RISK_FEATURES
    assert "test_files_changed" in RISK_FEATURES
    assert "dependency_files_changed" in RISK_FEATURES
    assert "hotspot_score" in RISK_FEATURES
    assert "recent_file_bugfix_rate" in RISK_FEATURES
    assert "recent_file_change_rate" in RISK_FEATURES
    assert "author_file_familiarity" in RISK_FEATURES
    assert "author_repo_experience" in RISK_FEATURES
    assert "ci_failures" in RISK_FEATURES
    assert "changes_requested" in RISK_FEATURES
    assert "review_comment_count" in RISK_FEATURES
    assert "review_rounds" in RISK_FEATURES
    assert FEATURE_SCHEMA_VERSION == "v1"


def test_validate_feature_dict_valid() -> None:
    valid_features = {f: 1.0 for f in RISK_FEATURES}
    validated = validate_feature_dict(valid_features)
    assert len(validated) == 16
    assert all(isinstance(v, float) for v in validated.values())


def test_validate_feature_dict_missing_feature() -> None:
    incomplete = {"lines_added": 10, "lines_deleted": 5}
    with pytest.raises(ValueError, match="Missing required feature"):
        validate_feature_dict(incomplete)


def test_validate_feature_dict_invalid_type() -> None:
    invalid = {f: 1.0 for f in RISK_FEATURES}
    invalid["hotspot_score"] = "not-a-number"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="invalid non-numeric value"):
        validate_feature_dict(invalid)


def test_file_classification() -> None:
    assert classify_file("app/main.py") == "source"
    assert classify_file("src/main/java/App.java") == "source"
    assert classify_file("tests/test_logic.py") == "test"
    assert classify_file("src/test/java/AppTest.java") == "test"
    assert classify_file("frontend/src/components/Button.test.tsx") == "test"
    assert classify_file("package.json") == "dependency"
    assert classify_file("pom.xml") == "dependency"
    assert classify_file("pyproject.toml") == "dependency"
    assert classify_file("Dockerfile") == "dependency"
    assert classify_file(".github/workflows/ci.yml") == "dependency"
    assert classify_file("README.md") == "other"


def test_risk_level_boundaries() -> None:
    assert risk_level(0.05, medium_threshold=0.15, high_threshold=0.30) == "LOW"
    assert risk_level(0.149, medium_threshold=0.15, high_threshold=0.30) == "LOW"
    assert risk_level(0.15, medium_threshold=0.15, high_threshold=0.30) == "MEDIUM"
    assert risk_level(0.299, medium_threshold=0.15, high_threshold=0.30) == "MEDIUM"
    assert risk_level(0.30, medium_threshold=0.15, high_threshold=0.30) == "HIGH"
    assert risk_level(0.85, medium_threshold=0.15, high_threshold=0.30) == "HIGH"


def test_future_data_leakage_guard() -> None:
    """CRITICAL ML TEST: Asserts events occurring AFTER snapshot_at are never included."""
    snapshot_time = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)

    pr_record = {
        "number": 42,
        "additions": 100,
        "deletions": 20,
        "changed_files": 2,
        "commit_count": 1,
        "opened_at": datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC),
        "raw_data": {
            "files": [
                {"filename": "app/service.py", "additions": 50, "deletions": 10},
                {"filename": "tests/test_service.py", "additions": 50, "deletions": 10},
            ],
            "commits": [
                {"sha": "c1", "committerDate": "2025-06-01T10:30:00Z"},  # Before snapshot
                {"sha": "c2", "committerDate": "2025-06-02T15:00:00Z"},  # Future commit!
            ],
            "reviews": [
                {
                    "state": "CHANGES_REQUESTED",
                    "submittedAt": "2025-06-01T11:00:00Z",
                    "user": {"login": "alice"},
                },  # Before snapshot
                {
                    "state": "CHANGES_REQUESTED",
                    "submittedAt": "2025-06-03T10:00:00Z",
                    "user": {"login": "bob"},
                },  # Future review!
            ],
            "comments": [
                {"id": 1, "createdAt": "2025-06-01T11:30:00Z"},  # Before snapshot
                {"id": 2, "createdAt": "2025-06-02T09:00:00Z"},  # Future comment!
                {"id": 3, "createdAt": "2025-06-04T12:00:00Z"},  # Future comment!
            ],
        },
    }

    features = extract_features_from_pr_record(pr_record, snapshot_at=snapshot_time)

    # Commits before snapshot = 1 (future commit c2 omitted)
    assert features["commit_count"] == 1
    # Changes requested before snapshot = 1 (future changes_requested by bob omitted)
    assert features["changes_requested"] == 1
    # Comments before snapshot = 1 (future comments omitted)
    assert features["review_comment_count"] == 1
    # Review rounds before snapshot = 1 (only alice)
    assert features["review_rounds"] == 1
    # Source / test files properly counted
    assert features["source_files_changed"] == 1
    assert features["test_files_changed"] == 1
