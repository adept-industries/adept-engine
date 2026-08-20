"""PR outcome and strong adverse event labeling layer.

CRITICAL RULE:
A merged PR is labelled risky only using strong post-merge evidence within
a configurable observation window (default 14 days).
Never invent weak causal links.
"""

import re
from datetime import UTC, datetime, timedelta
from typing import Any

DEFAULT_OBSERVATION_WINDOW_DAYS = 14

_REVERT_PATTERN = re.compile(
    r"(^revert\b|reverts\s+(?:pr\s+)?#?(\d+)|reverting\b|rollback\b)",
    re.IGNORECASE,
)

_HOTFIX_PATTERN = re.compile(
    r"\b(?:hotfix|emergency\s+fix|patch)\s+(?:for|to|referencing)?\s*(?:pr\s*)?#?(\d+)\b",
    re.IGNORECASE,
)


def evaluate_pr_outcome(
    pr_row: dict[str, Any],
    later_prs: list[dict[str, Any]] | None = None,
    deployments: list[dict[str, Any]] | None = None,
    incidents: list[dict[str, Any]] | None = None,
    observation_window_days: int = DEFAULT_OBSERVATION_WINDOW_DAYS,
) -> dict[str, Any]:
    """Evaluates whether a merged PR experienced strong adverse post-merge events."""
    merged_at = pr_row.get("merged_at")
    pr_number = int(pr_row.get("number", 0))

    if not merged_at:
        # Not merged yet -> not an outcome record
        return {
            "is_risky": False,
            "reason": "NOT_MERGED",
            "merged_at": None,
            "observed_until": datetime.now(UTC),
            "evidence": {},
        }

    if isinstance(merged_at, str):
        merged_at_dt = datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
    else:
        merged_at_dt = merged_at

    if merged_at_dt.tzinfo is None:
        merged_at_dt = merged_at_dt.replace(tzinfo=UTC)

    observed_until = merged_at_dt + timedelta(days=observation_window_days)

    # 1. Check for explicit reverts in later PRs
    if later_prs:
        for later_pr in later_prs:
            later_opened = later_pr.get("opened_at")
            if isinstance(later_opened, str):
                later_opened = datetime.fromisoformat(later_opened.replace("Z", "+00:00"))
            if later_opened and later_opened.tzinfo is None:
                later_opened = later_opened.replace(tzinfo=UTC)

            if later_opened and merged_at_dt <= later_opened <= observed_until:
                title = later_pr.get("title", "")
                revert_match = _REVERT_PATTERN.search(title)
                if revert_match:
                    # Check if it specifically targets this PR or title
                    ref_group = revert_match.group(2)
                    matches_ref = bool(ref_group and int(ref_group) == pr_number)
                    if matches_ref or str(pr_number) in title:
                        return {
                            "is_risky": True,
                            "reason": "EXPLICIT_REVERT",
                            "merged_at": merged_at_dt,
                            "observed_until": observed_until,
                            "evidence": {
                                "revert_pr_number": later_pr.get("number"),
                                "revert_pr_title": title,
                                "revert_pr_opened_at": str(later_opened),
                            },
                        }

                # 2. Check for explicit hotfixes referencing this PR
                hotfix_match = _HOTFIX_PATTERN.search(title)
                if (
                    hotfix_match
                    and hotfix_match.group(1)
                    and int(hotfix_match.group(1)) == pr_number
                ):
                    return {
                        "is_risky": True,
                        "reason": "HOTFIX_REMEDY",
                        "merged_at": merged_at_dt,
                        "observed_until": observed_until,
                        "evidence": {
                            "hotfix_pr_number": later_pr.get("number"),
                            "hotfix_pr_title": title,
                            "hotfix_pr_opened_at": str(later_opened),
                        },
                    }

    # 3. Check for failed production deployments containing this PR
    if deployments:
        for dep in deployments:
            dep_finished = dep.get("finished_at")
            if isinstance(dep_finished, str):
                dep_finished = datetime.fromisoformat(dep_finished.replace("Z", "+00:00"))
            if dep_finished and dep_finished.tzinfo is None:
                dep_finished = dep_finished.replace(tzinfo=UTC)

            if dep_finished and merged_at_dt <= dep_finished <= observed_until:
                status = (dep.get("status") or "").upper()
                is_prod = dep.get("is_production", False)
                if is_prod and status == "FAILURE":
                    return {
                        "is_risky": True,
                        "reason": "FAILED_DEPLOYMENT",
                        "merged_at": merged_at_dt,
                        "observed_until": observed_until,
                        "evidence": {
                            "deployment_id": str(dep.get("id")),
                            "environment": dep.get("environment"),
                            "finished_at": str(dep_finished),
                        },
                    }

    # 4. Check for production incidents explicitly linked to this PR
    if incidents:
        for inc in incidents:
            inc_detected = inc.get("detected_at")
            if isinstance(inc_detected, str):
                inc_detected = datetime.fromisoformat(inc_detected.replace("Z", "+00:00"))
            if inc_detected and inc_detected.tzinfo is None:
                inc_detected = inc_detected.replace(tzinfo=UTC)

            if inc_detected and merged_at_dt <= inc_detected <= observed_until:
                return {
                    "is_risky": True,
                    "reason": "PRODUCTION_INCIDENT",
                    "merged_at": merged_at_dt,
                    "observed_until": observed_until,
                    "evidence": {
                        "incident_id": str(inc.get("id")),
                        "title": inc.get("title"),
                        "severity": inc.get("severity"),
                        "detected_at": str(inc_detected),
                    },
                }

    return {
        "is_risky": False,
        "reason": "NO_ADVERSE_EVENT_OBSERVED",
        "merged_at": merged_at_dt,
        "observed_until": observed_until,
        "evidence": {},
    }
