from __future__ import annotations

import hashlib
import math
import pickle
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_training.src.constants import EXPECTED_SOURCE_SHA256, FEATURE_ORDER, SPLITS


class DatasetIntegrityError(ValueError):
    """The supplied research data does not satisfy the frozen training contract."""


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    commit_hashes: tuple[str, ...]
    features: np.ndarray
    labels: np.ndarray
    source_sha256: dict[str, str]
    fix_message_comparison: dict[str, int]

    @property
    def class_distribution(self) -> dict[str, int]:
        values, counts = np.unique(self.labels, return_counts=True)
        return {str(int(value)): int(count) for value, count in zip(values, counts, strict=True)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_fix(value: object) -> float:
    """Convert the prepared FIX value without treating the string ``False`` as true."""
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "true":
            return 1.0
        if normalized == "false":
            return 0.0
        raise DatasetIntegrityError(f"unsupported fix string: {value!r}")
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if numeric in (0.0, 1.0):
            return numeric
    raise DatasetIntegrityError(f"unsupported fix value: {value!r}")


FIX_PATTERN = re.compile(
    r"(?<![a-z0-9])(bug|bugfix|defect|fix|fixed|fixes|fixing|hotfix|patch|regression)"
    r"(?![a-z0-9])"
)


def message_matches_runtime_fix_rule(value: object) -> int:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return int(FIX_PATTERN.search(normalized) is not None)


def _validate_hash(path: Path, *, verify_source_checksums: bool) -> str:
    if not path.is_file():
        raise DatasetIntegrityError(f"missing dataset file: {path}")
    actual = sha256_file(path)
    if verify_source_checksums:
        expected = EXPECTED_SOURCE_SHA256[path.name]
        if actual != expected:
            raise DatasetIntegrityError(
                f"checksum mismatch for {path.name}: expected {expected}, got {actual}"
            )
    return actual


def _load_pickle(path: Path, *, allow_unsafe_pickle: bool) -> Any:
    if not allow_unsafe_pickle:
        raise DatasetIntegrityError(
            "pickle loading is disabled; use an isolated, no-network research environment "
            "and pass allow_unsafe_pickle=True explicitly"
        )
    with path.open("rb") as source:
        return pickle.load(source)  # noqa: S301 - explicit isolated-research opt-in above


def _normalize_commit_hash(value: object) -> str:
    commit_hash = str(value).strip().lower()
    if not commit_hash:
        raise DatasetIntegrityError("empty commit hash")
    return commit_hash


def _load_split(
    data_dir: Path,
    split: str,
    *,
    allow_unsafe_pickle: bool,
    verify_source_checksums: bool,
) -> DatasetSplit:
    changes_path = data_dir / f"changes_{split}.pkl"
    features_path = data_dir / f"features_{split}.pkl"
    source_sha256 = {
        changes_path.name: _validate_hash(
            changes_path, verify_source_checksums=verify_source_checksums
        ),
        features_path.name: _validate_hash(
            features_path, verify_source_checksums=verify_source_checksums
        ),
    }

    changes = _load_pickle(changes_path, allow_unsafe_pickle=allow_unsafe_pickle)
    features = _load_pickle(features_path, allow_unsafe_pickle=allow_unsafe_pickle)
    if not isinstance(changes, (list, tuple)) or len(changes) != 4:
        raise DatasetIntegrityError(f"{changes_path.name} must contain four parallel sequences")
    if not isinstance(features, pd.DataFrame):
        raise DatasetIntegrityError(f"{features_path.name} must contain a pandas DataFrame")

    commit_ids, labels, messages = changes[0], changes[1], changes[2]
    if len(commit_ids) != len(labels) or len(commit_ids) != len(messages):
        raise DatasetIntegrityError(f"parallel-sequence length mismatch in {changes_path.name}")
    required_columns = {"commit_hash", "is_buggy_commit", *FEATURE_ORDER}
    missing_columns = required_columns.difference(features.columns)
    if missing_columns:
        raise DatasetIntegrityError(
            f"{features_path.name} is missing columns: {sorted(missing_columns)}"
        )

    label_frame = pd.DataFrame(
        {
            "commit_hash": [_normalize_commit_hash(value) for value in commit_ids],
            "change_label": labels,
            "commit_message": messages,
        }
    )
    feature_frame = features.loc[:, ["commit_hash", "is_buggy_commit", *FEATURE_ORDER]].copy()
    feature_frame["commit_hash"] = feature_frame["commit_hash"].map(_normalize_commit_hash)

    duplicate_labels = label_frame["commit_hash"].duplicated(keep=False)
    duplicate_features = feature_frame["commit_hash"].duplicated(keep=False)
    if duplicate_labels.any() or duplicate_features.any():
        raise DatasetIntegrityError(f"duplicate commit hashes in {split} split")

    label_hashes = set(label_frame["commit_hash"])
    feature_hashes = set(feature_frame["commit_hash"])
    if label_hashes != feature_hashes:
        missing = sorted(label_hashes - feature_hashes)[:5]
        extra = sorted(feature_hashes - label_hashes)[:5]
        raise DatasetIntegrityError(
            f"commit-hash join mismatch in {split}: missing={missing}, extra={extra}"
        )

    joined = label_frame.merge(
        feature_frame,
        on="commit_hash",
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    try:
        joined["change_label"] = joined["change_label"].astype("int8")
        joined["is_buggy_commit"] = joined["is_buggy_commit"].astype("int8")
    except (TypeError, ValueError) as exc:
        raise DatasetIntegrityError(f"non-binary labels in {split}") from exc
    for label_column in ("change_label", "is_buggy_commit"):
        if not joined[label_column].isin((0, 1)).all():
            raise DatasetIntegrityError(f"non-binary {label_column} in {split}")
    if not joined["change_label"].equals(joined["is_buggy_commit"]):
        raise DatasetIntegrityError(f"label disagreement between files in {split}")

    joined["fix"] = joined["fix"].map(parse_fix)
    message_fix = joined["commit_message"].map(message_matches_runtime_fix_rule)
    prepared_fix = joined["fix"].astype("int8")
    fix_message_comparison = {
        "rows": len(joined),
        "preparedFixPositive": int(prepared_fix.sum()),
        "messageRulePositive": int(message_fix.sum()),
        "agree": int((prepared_fix == message_fix).sum()),
        "preparedPositiveMessageNegative": int(((prepared_fix == 1) & (message_fix == 0)).sum()),
        "preparedNegativeMessagePositive": int(((prepared_fix == 0) & (message_fix == 1)).sum()),
    }
    try:
        numeric_features = joined.loc[:, FEATURE_ORDER].astype("float64")
    except (TypeError, ValueError) as exc:
        raise DatasetIntegrityError(f"non-numeric feature in {split}") from exc
    feature_values = numeric_features.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(feature_values).all():
        raise DatasetIntegrityError(f"missing or non-finite feature in {split}")
    if (feature_values < 0).any():
        raise DatasetIntegrityError(f"negative feature in {split}")

    entropy = numeric_features["entropy"].to_numpy()
    nf = numeric_features["nf"].to_numpy()
    entropy_ceiling = np.where(nf > 1, np.log2(nf), 0.0)
    if np.any(entropy > entropy_ceiling + 1e-6):
        raise DatasetIntegrityError(
            f"entropy exceeds unnormalized base-2 Shannon ceiling in {split}"
        )
    if np.any((nf <= 1) & (np.abs(entropy) > 1e-6)):
        raise DatasetIntegrityError(f"single-file change has non-zero entropy in {split}")

    return DatasetSplit(
        name=split,
        commit_hashes=tuple(joined["commit_hash"]),
        features=feature_values,
        labels=joined["change_label"].to_numpy(dtype=np.int8, copy=True),
        source_sha256=source_sha256,
        fix_message_comparison=fix_message_comparison,
    )


def load_jitfine_dataset(
    data_dir: Path,
    *,
    allow_unsafe_pickle: bool = False,
    verify_source_checksums: bool = True,
) -> dict[str, DatasetSplit]:
    resolved = data_dir.resolve()
    _validate_hash(
        resolved / "dataset_dict.pkl",
        verify_source_checksums=verify_source_checksums,
    )
    splits = {
        split: _load_split(
            resolved,
            split,
            allow_unsafe_pickle=allow_unsafe_pickle,
            verify_source_checksums=verify_source_checksums,
        )
        for split in SPLITS
    }
    commit_sets = {name: set(split.commit_hashes) for name, split in splits.items()}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = commit_sets[left].intersection(commit_sets[right])
            if overlap:
                raise DatasetIntegrityError(f"{left}/{right} leakage: {sorted(overlap)[:5]}")
    for split in splits.values():
        if set(split.class_distribution) != {"0", "1"}:
            raise DatasetIntegrityError(f"{split.name} split does not contain both classes")
        if split.features.shape[1] != len(FEATURE_ORDER):
            raise DatasetIntegrityError(f"wrong feature width in {split.name}")
        if any(not math.isfinite(float(value)) for value in split.features.flat):
            raise DatasetIntegrityError(f"non-finite value in {split.name}")
    return splits
