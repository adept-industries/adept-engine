"""PR risk model features and schema definition."""

import re
from datetime import UTC, datetime
from typing import Any

# Canonical ordering of features for training, inference, and serialization.
RISK_FEATURES: list[str] = [
    "lines_added",
    "lines_deleted",
    "files_changed",
    "commit_count",
    "source_files_changed",
    "test_files_changed",
    "dependency_files_changed",
    "hotspot_score",
    "recent_file_bugfix_rate",
    "recent_file_change_rate",
    "author_file_familiarity",
    "author_repo_experience",
    "ci_failures",
    "changes_requested",
    "review_comment_count",
    "review_rounds",
]

FEATURE_SCHEMA_VERSION: str = "v1"

_TEST_FILE_PATTERNS = re.compile(
    r"(^|/)(test|tests|spec|specs|__tests__|__spec__|e2e)/|"
    r"(_test|\.test|\.spec|test_|_spec|Test|Tests)\.(py|java|ts|tsx|js|jsx|go|rs|rb|php|cs|kt)$",
    re.IGNORECASE,
)

_DEPENDENCY_FILE_NAMES = {
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "pyproject.toml",
    "requirements.txt",
    "Pipfile",
    "Pipfile.lock",
    "uv.lock",
    "Cargo.toml",
    "Cargo.lock",
    "go.mod",
    "go.sum",
    "Gemfile",
    "Gemfile.lock",
    "composer.json",
    "composer.lock",
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yaml",
    "compose.yml",
    "pipfile",
    "pipfile.lock",
    "cargo.toml",
    "cargo.lock",
    "gemfile",
    "gemfile.lock",
}

_SOURCE_EXTENSIONS = {
    ".py",
    ".java",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".rb",
    ".php",
    ".cs",
    ".kt",
    ".scala",
    ".swift",
    ".sql",
    ".sh",
}

_BUGFIX_PATTERN = re.compile(
    r"\b(fix|fixed|fixes|bug|patch|revert|hotfix|issue|resolve|resolved)\b",
    re.IGNORECASE,
)


def classify_file(filename: str) -> str:
    """Classifies a filename into 'test', 'dependency', 'source', or 'other'."""
    normalized = filename.strip().lower()
    base_name = normalized.rsplit("/", 1)[-1]

    if base_name in _DEPENDENCY_FILE_NAMES or ".github/" in normalized:
        return "dependency"

    if _TEST_FILE_PATTERNS.search(normalized):
        return "test"

    dot_index = base_name.rfind(".")
    if dot_index != -1:
        ext = base_name[dot_index:]
        if ext in _SOURCE_EXTENSIONS:
            return "source"

    return "other"


def risk_level(
    probability: float,
    medium_threshold: float = 0.15,
    high_threshold: float = 0.30,
) -> str:
    """Classifies calibrated risk probability into LOW / MEDIUM / HIGH policy bands.

    Thresholds are policy boundaries derived from calibration/validation data,
    favoring high precision for HIGH alerts.
    """
    if probability >= high_threshold:
        return "HIGH"
    if probability >= medium_threshold:
        return "MEDIUM"
    return "LOW"


def validate_feature_dict(features: dict[str, Any]) -> dict[str, float]:
    """Validates that all required RISK_FEATURES are present and finite numeric values."""
    validated: dict[str, float] = {}
    for name in RISK_FEATURES:
        if name not in features:
            raise ValueError(f"Missing required feature: '{name}'")
        val = features[name]
        try:
            float_val = float(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature '{name}' has invalid non-numeric value: {val!r}") from exc
        if not (-1e9 < float_val < 1e9):
            raise ValueError(f"Feature '{name}' has out-of-range value: {float_val}")
        validated[name] = float_val
    return validated


def extract_features_from_pr_record(
    pr_row: dict[str, Any],
    historical_context: dict[str, Any] | None = None,
    snapshot_at: datetime | None = None,
) -> dict[str, float | int]:
    """Extracts leakage-safe features for a pull request as of snapshot_at.

    CRITICAL ML RULE: Only data existing at or before snapshot_at is evaluated.
    """
    effective_snapshot_at = snapshot_at or pr_row.get("opened_at") or datetime.now(UTC)
    if effective_snapshot_at.tzinfo is None:
        effective_snapshot_at = effective_snapshot_at.replace(tzinfo=UTC)

    raw_data = pr_row.get("raw_data") or {}
    files_list = raw_data.get("files") or []
    commits_list = raw_data.get("commits") or []
    reviews_list = raw_data.get("reviews") or []
    comments_list = raw_data.get("comments") or []

    # Filter commits, reviews, comments strictly by snapshot_at
    valid_commits = []
    for c in commits_list:
        c_date_str = c.get("committerDate") or c.get("authorDate")
        if c_date_str:
            try:
                c_date = datetime.fromisoformat(c_date_str.replace("Z", "+00:00"))
                if c_date <= effective_snapshot_at:
                    valid_commits.append(c)
            except Exception:
                valid_commits.append(c)
        else:
            valid_commits.append(c)

    commit_count = len(valid_commits) if valid_commits else int(pr_row.get("commit_count", 1))
    commit_count = max(1, commit_count)

    # Classify files
    source_count = 0
    test_count = 0
    dep_count = 0
    files_changed = 0

    if files_list:
        for f in files_list:
            fname = f.get("filename", "")
            ftype = classify_file(fname)
            if ftype == "source":
                source_count += 1
            elif ftype == "test":
                test_count += 1
            elif ftype == "dependency":
                dep_count += 1
            files_changed += 1
    else:
        files_changed = int(pr_row.get("changed_files", 1))
        source_count = max(1, files_changed)

    lines_added = int(pr_row.get("additions", 0))
    lines_deleted = int(pr_row.get("deletions", 0))
    if lines_added == 0 and lines_deleted == 0 and files_list:
        lines_added = sum(int(f.get("additions", 0)) for f in files_list)
        lines_deleted = sum(int(f.get("deletions", 0)) for f in files_list)

    # Review activity strictly before or at snapshot_at
    changes_requested = 0
    review_rounds_set = set()
    for r in reviews_list:
        r_date_str = r.get("submittedAt")
        if r_date_str:
            try:
                r_date = datetime.fromisoformat(r_date_str.replace("Z", "+00:00"))
                if r_date > effective_snapshot_at:
                    continue
            except Exception:
                pass
        state = (r.get("state") or "").upper()
        if state == "CHANGES_REQUESTED":
            changes_requested += 1
        r_author = (r.get("user") or {}).get("login")
        if r_author:
            review_rounds_set.add(r_author)

    review_rounds = len(review_rounds_set)

    # Review comment count strictly before or at snapshot_at
    review_comment_count = 0
    for cm in comments_list:
        cm_date_str = cm.get("createdAt")
        if cm_date_str:
            try:
                cm_date = datetime.fromisoformat(cm_date_str.replace("Z", "+00:00"))
                if cm_date > effective_snapshot_at:
                    continue
            except Exception:
                pass
        review_comment_count += 1

    # CI failures (if provided in context or raw_data)
    ci_failures = 0
    if historical_context and "ci_failures" in historical_context:
        ci_failures = int(historical_context["ci_failures"])

    # Historical repository metrics before snapshot_at
    hotspot_score = 0.1
    recent_file_bugfix_rate = 0.05
    recent_file_change_rate = 0.1
    author_file_familiarity = 0.5
    author_repo_experience = 0.2

    if historical_context:
        hotspot_score = float(historical_context.get("hotspot_score", hotspot_score))
        recent_file_bugfix_rate = float(
            historical_context.get("recent_file_bugfix_rate", recent_file_bugfix_rate)
        )
        recent_file_change_rate = float(
            historical_context.get("recent_file_change_rate", recent_file_change_rate)
        )
        author_file_familiarity = float(
            historical_context.get("author_file_familiarity", author_file_familiarity)
        )
        author_repo_experience = float(
            historical_context.get("author_repo_experience", author_repo_experience)
        )

    # Clamp bounded features to [0.0, 1.0]
    hotspot_score = max(0.0, min(1.0, hotspot_score))
    recent_file_bugfix_rate = max(0.0, min(1.0, recent_file_bugfix_rate))
    recent_file_change_rate = max(0.0, min(1.0, recent_file_change_rate))
    author_file_familiarity = max(0.0, min(1.0, author_file_familiarity))
    author_repo_experience = max(0.0, min(1.0, author_repo_experience))

    feature_dict: dict[str, float | int] = {
        "lines_added": lines_added,
        "lines_deleted": lines_deleted,
        "files_changed": max(1, files_changed),
        "commit_count": max(1, commit_count),
        "source_files_changed": source_count,
        "test_files_changed": test_count,
        "dependency_files_changed": dep_count,
        "hotspot_score": round(hotspot_score, 4),
        "recent_file_bugfix_rate": round(recent_file_bugfix_rate, 4),
        "recent_file_change_rate": round(recent_file_change_rate, 4),
        "author_file_familiarity": round(author_file_familiarity, 4),
        "author_repo_experience": round(author_repo_experience, 4),
        "ci_failures": max(0, ci_failures),
        "changes_requested": max(0, changes_requested),
        "review_comment_count": max(0, review_comment_count),
        "review_rounds": max(0, review_rounds),
    }

    return feature_dict
