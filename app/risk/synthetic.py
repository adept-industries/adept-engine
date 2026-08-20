"""Isolated synthetic/demo PR dataset generator.

CRITICAL RULE:
Demo data is strictly isolated to prove training, calibration, inference,
and persistence pipelines.
Never combine synthetic and real training data.
Every output is labelled 'DEMO DATA — NOT REAL-WORLD MODEL ACCURACY'.
"""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app.risk.features import RISK_FEATURES

DEMO_DISCLAIMER = "DEMO DATA — NOT REAL-WORLD MODEL ACCURACY"


def generate_synthetic_pr_dataset(
    n_samples: int = 5000,
    seed: int = 42,
    output_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generates an isolated synthetic dataset for pipeline demonstration."""
    rng = np.random.default_rng(seed)

    lines_added = rng.lognormal(5.0, 1.0, n_samples).astype(int)
    lines_deleted = rng.lognormal(4.2, 1.0, n_samples).astype(int)
    files_changed = np.maximum(1, (lines_added + lines_deleted) // rng.integers(40, 180, n_samples))
    commit_count = np.maximum(1, rng.poisson(4, n_samples))
    source_files_changed = np.maximum(
        0, (files_changed * rng.uniform(0.45, 0.90, n_samples)).astype(int)
    )
    test_files_changed = np.maximum(
        0, (files_changed * rng.uniform(0.0, 0.35, n_samples)).astype(int)
    )
    dependency_files_changed = rng.binomial(3, 0.08, n_samples)
    hotspot_score = np.round(rng.beta(2, 3, n_samples), 4)
    recent_file_bugfix_rate = np.round(rng.beta(1.4, 6, n_samples), 4)
    recent_file_change_rate = np.round(rng.beta(2, 4, n_samples), 4)
    author_file_familiarity = np.round(rng.beta(2.5, 2, n_samples), 4)
    author_repo_experience = np.round(rng.beta(2, 2, n_samples), 4)
    ci_failures = rng.poisson(0.35, n_samples)
    changes_requested = rng.poisson(0.55, n_samples)
    review_comment_count = rng.poisson(3.0, n_samples)
    review_rounds = np.maximum(0, rng.poisson(1.2, n_samples))

    # Realistic synthetic risk latent function
    logit = (
        -4.8
        + 0.0014 * np.minimum(lines_added + lines_deleted, 2500)
        + 0.055 * np.minimum(files_changed, 40)
        + 2.2 * hotspot_score
        + 3.0 * recent_file_bugfix_rate
        + 1.4 * recent_file_change_rate
        - 1.0 * author_file_familiarity
        - 0.65 * author_repo_experience
        + 0.55 * ci_failures
        + 0.35 * changes_requested
        + 0.25 * dependency_files_changed
        - 0.12 * np.minimum(test_files_changed, 8)
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, np.clip(p, 0.01, 0.95))

    start_date = datetime(2024, 1, 1, tzinfo=UTC)
    dates = [start_date + pd.Timedelta(hours=i * 2) for i in range(n_samples)]

    df = pd.DataFrame(
        {
            "snapshot_at": dates,
            "lines_added": lines_added,
            "lines_deleted": lines_deleted,
            "files_changed": files_changed,
            "commit_count": commit_count,
            "source_files_changed": source_files_changed,
            "test_files_changed": test_files_changed,
            "dependency_files_changed": dependency_files_changed,
            "hotspot_score": hotspot_score,
            "recent_file_bugfix_rate": recent_file_bugfix_rate,
            "recent_file_change_rate": recent_file_change_rate,
            "author_file_familiarity": author_file_familiarity,
            "author_repo_experience": author_repo_experience,
            "ci_failures": ci_failures,
            "changes_requested": changes_requested,
            "review_comment_count": review_comment_count,
            "review_rounds": review_rounds,
            "is_risky": y,
        }
    )

    assert all(c in df.columns for c in RISK_FEATURES)

    if output_csv_path:
        out_p = Path(output_csv_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)

    return df
