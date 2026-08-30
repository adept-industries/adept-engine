from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest

from ml_training.src.constants import FEATURE_ORDER
from ml_training.src.dataset_loader import (
    DatasetIntegrityError,
    load_jitfine_dataset,
    message_matches_runtime_fix_rule,
    parse_fix,
)


def _write_split(
    data_dir: Path,
    split: str,
    commits: list[str],
    *,
    duplicate_feature: bool = False,
) -> None:
    labels = [index % 2 for index in range(len(commits))]
    messages = ["ordinary refactor", "Fix parser regression"] * (len(commits) // 2)
    code_changes: list[dict[str, set[str]]] = [{"added_code": set(), "removed_code": set()}] * len(
        commits
    )
    changes = [commits, labels, messages, code_changes]
    with (data_dir / f"changes_{split}.pkl").open("wb") as destination:
        pickle.dump(changes, destination)

    feature_commits = list(reversed(commits))
    if duplicate_feature:
        feature_commits[-1] = feature_commits[0]
    label_by_commit = dict(zip(commits, labels, strict=True))
    rows = []
    for index, commit in enumerate(feature_commits):
        nf = 1 if index % 2 == 0 else 2
        rows.append(
            {
                "commit_hash": commit,
                "is_buggy_commit": label_by_commit.get(commit, 0),
                "ns": 1,
                "nd": 1,
                "nf": nf,
                "entropy": 0.0 if nf == 1 else 1.0,
                "la": index + 1,
                "ld": index,
                "fix": "True" if label_by_commit.get(commit, 0) else "False",
            }
        )
    pd.DataFrame(rows).to_pickle(data_dir / f"features_{split}.pkl")


def _write_dataset(data_dir: Path, *, overlap: bool = False) -> None:
    with (data_dir / "dataset_dict.pkl").open("wb") as destination:
        pickle.dump(({}, {}), destination)
    _write_split(data_dir, "train", ["train-0", "train-1", "train-2", "train-3"])
    valid_commits = ["valid-0", "valid-1", "valid-2", "valid-3"]
    if overlap:
        valid_commits[0] = "train-0"
    _write_split(data_dir, "valid", valid_commits)
    _write_split(data_dir, "test", ["test-0", "test-1", "test-2", "test-3"])


def test_loads_by_commit_hash_and_preserves_frozen_feature_order(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    splits = load_jitfine_dataset(
        tmp_path,
        allow_unsafe_pickle=True,
        verify_source_checksums=False,
    )

    train = splits["train"]
    assert FEATURE_ORDER == ("ns", "nd", "nf", "entropy", "la", "ld", "fix")
    assert train.commit_hashes == ("train-0", "train-1", "train-2", "train-3")
    assert train.features.shape == (4, 7)
    assert train.features[:, -1].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert train.labels.tolist() == [0, 1, 0, 1]
    assert train.fix_message_comparison["agree"] == 4


def test_rejects_pickle_without_explicit_isolation_acknowledgement(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    with pytest.raises(DatasetIntegrityError, match="pickle loading is disabled"):
        load_jitfine_dataset(tmp_path, verify_source_checksums=False)


def test_rejects_duplicate_commit_hash(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    _write_split(
        tmp_path,
        "valid",
        ["valid-0", "valid-1", "valid-2", "valid-3"],
        duplicate_feature=True,
    )

    with pytest.raises(DatasetIntegrityError, match="duplicate commit hashes"):
        load_jitfine_dataset(
            tmp_path,
            allow_unsafe_pickle=True,
            verify_source_checksums=False,
        )


def test_rejects_cross_split_commit_leakage(tmp_path: Path) -> None:
    _write_dataset(tmp_path, overlap=True)

    with pytest.raises(DatasetIntegrityError, match="train/valid leakage"):
        load_jitfine_dataset(
            tmp_path,
            allow_unsafe_pickle=True,
            verify_source_checksums=False,
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(False, 0.0), (True, 1.0), ("False", 0.0), (" true ", 1.0), (0, 0.0), (1.0, 1.0)],
)
def test_fix_conversion_is_explicit(value: object, expected: float) -> None:
    assert parse_fix(value) == expected


@pytest.mark.parametrize("value", ["", "yes", 2, None])
def test_fix_conversion_rejects_unknown_values(value: object) -> None:
    with pytest.raises(DatasetIntegrityError, match="unsupported fix"):
        parse_fix(value)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Fix parser regression", 1),
        ("HOTFIX: avoid crash", 1),
        ("update fixture and prefix", 0),
        ("Refactor only", 0),
    ],
)
def test_runtime_fix_comparison_uses_token_boundaries(message: str, expected: int) -> None:
    assert message_matches_runtime_fix_rule(message) == expected
