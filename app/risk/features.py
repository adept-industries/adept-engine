"""Seven-feature runtime contract for the approved JIT-Fine-derived model."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

FEATURE_SCHEMA_VERSION = "jitfine-pr-features-v1"
FEATURE_ORDER = ("ns", "nd", "nf", "entropy", "la", "ld", "fix")
FILE_SCOPE = "all_changed_files_reported_by_github"

_FIX_PATTERN = re.compile(
    r"\b(?:fix|fixed|fixes|bug|defect|patch|revert|hotfix|issue|resolve|resolved)\b",
    re.IGNORECASE,
)


class RiskFeatureUnavailable(ValueError):
    """The provider cannot supply a complete, trustworthy feature vector."""


@dataclass(frozen=True, slots=True)
class PullRequestRiskFeatures:
    ns: int
    nd: int
    nf: int
    entropy: float
    la: int
    ld: int
    fix: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "ns": self.ns,
            "nd": self.nd,
            "nf": self.nf,
            "entropy": self.entropy,
            "la": self.la,
            "ld": self.ld,
            "fix": self.fix,
        }

    def as_ordered_values(self) -> tuple[float, ...]:
        values = self.as_dict()
        return tuple(float(values[name]) for name in FEATURE_ORDER)


def extract_pull_request_features(
    pull_request: dict[str, Any],
    files: list[dict[str, Any]],
    commits: list[dict[str, Any]],
) -> PullRequestRiskFeatures:
    """Aggregate one GitHub pull request into the frozen seven-feature order.

    Runtime deliberately uses every changed file returned by GitHub. The training
    package excludes some non-source files, so this scope decision is persisted
    with every snapshot and remains a documented MVP limitation.
    """
    expected_files = _non_negative_int(pull_request.get("changed_files"), "changed_files")
    if expected_files > 3_000:
        raise RiskFeatureUnavailable(
            "GitHub caps pull-request file results at 3,000; complete extraction is impossible"
        )
    if len(files) != expected_files:
        raise RiskFeatureUnavailable(
            f"GitHub returned {len(files)} files but the pull request reports {expected_files}"
        )

    directories: set[str] = set()
    subsystems: set[str] = set()
    churn_by_file: list[int] = []
    additions = 0
    deletions = 0

    for file_data in files:
        filename = str(file_data.get("filename") or "").strip().strip("/")
        if not filename:
            raise RiskFeatureUnavailable("GitHub returned a changed file without a filename")
        path = PurePosixPath(filename)
        parent = str(path.parent)
        directory = "<root>" if parent == "." else parent
        subsystem = path.parts[0] if len(path.parts) > 1 else "<root>"
        directories.add(directory)
        subsystems.add(subsystem)

        file_additions = _non_negative_int(file_data.get("additions", 0), "file additions")
        file_deletions = _non_negative_int(file_data.get("deletions", 0), "file deletions")
        additions += file_additions
        deletions += file_deletions
        churn_by_file.append(file_additions + file_deletions)

    provider_additions = _non_negative_int(pull_request.get("additions", 0), "additions")
    provider_deletions = _non_negative_int(pull_request.get("deletions", 0), "deletions")
    if additions != provider_additions or deletions != provider_deletions:
        raise RiskFeatureUnavailable(
            "GitHub pull-request totals changed while files were being collected"
        )

    return PullRequestRiskFeatures(
        ns=len(subsystems),
        nd=len(directories),
        nf=len(files),
        entropy=_churn_entropy(churn_by_file),
        la=additions,
        ld=deletions,
        fix=int(_looks_like_fix(pull_request, commits)),
    )


def _churn_entropy(churn_by_file: list[int]) -> float:
    total = sum(churn_by_file)
    if total == 0:
        return 0.0
    return -sum(
        proportion * math.log2(proportion)
        for churn in churn_by_file
        if churn > 0
        for proportion in (churn / total,)
    )


def _looks_like_fix(pull_request: dict[str, Any], commits: list[dict[str, Any]]) -> bool:
    candidates = [pull_request.get("title"), pull_request.get("body")]
    for commit in commits:
        details = commit.get("commit")
        if isinstance(details, dict):
            candidates.append(details.get("message"))
    return any(
        _FIX_PATTERN.search(candidate) is not None
        for candidate in candidates
        if isinstance(candidate, str)
    )


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise RiskFeatureUnavailable(f"GitHub returned an invalid {field_name}")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise RiskFeatureUnavailable(f"GitHub returned an invalid {field_name}") from exc
    if parsed < 0:
        raise RiskFeatureUnavailable(f"GitHub returned a negative {field_name}")
    return parsed
