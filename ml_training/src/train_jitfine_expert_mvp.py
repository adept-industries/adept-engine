from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas
import sklearn
from sklearn.ensemble import RandomForestClassifier

from ml_training.src.artifact_export import LIMITATIONS, export_artifact
from ml_training.src.constants import (
    DATASET_ARCHIVE_SHA256,
    EXPECTED_SOURCE_SHA256,
    FEATURE_ORDER,
    FEATURE_SCHEMA_VERSION,
    MODEL_NAME,
    MODEL_VERSION,
)
from ml_training.src.dataset_loader import load_jitfine_dataset
from ml_training.src.evaluation import evaluate_probabilities, select_thresholds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the seven-feature JIT-Fine-derived MVP")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help="Required acknowledgement; run only in a no-network isolated research container.",
    )
    return parser


def train(data_dir: Path, output_dir: Path, *, allow_unsafe_pickle: bool) -> dict[str, object]:
    splits = load_jitfine_dataset(
        data_dir,
        allow_unsafe_pickle=allow_unsafe_pickle,
        verify_source_checksums=True,
    )
    train_split = splits["train"]
    valid_split = splits["valid"]
    test_split = splits["test"]

    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(train_split.features, train_split.labels)
    positive_positions = np.flatnonzero(model.classes_ == 1)
    if positive_positions.size != 1:
        raise RuntimeError(f"cannot identify positive class from {model.classes_.tolist()}")
    positive_position = int(positive_positions[0])
    validation_probabilities = model.predict_proba(valid_split.features)[:, positive_position]
    thresholds = select_thresholds(valid_split.labels, validation_probabilities)
    test_probabilities = model.predict_proba(test_split.features)[:, positive_position]

    la_index = FEATURE_ORDER.index("la")
    ld_index = FEATURE_ORDER.index("ld")
    validation_metrics = evaluate_probabilities(
        valid_split.labels,
        validation_probabilities,
        classification_threshold=thresholds.high,
        lines_changed=valid_split.features[:, la_index] + valid_split.features[:, ld_index],
    )
    test_metrics = evaluate_probabilities(
        test_split.labels,
        test_probabilities,
        classification_threshold=thresholds.high,
        lines_changed=test_split.features[:, la_index] + test_split.features[:, ld_index],
    )

    report: dict[str, object] = {
        "reportVersion": 1,
        "generatedAt": datetime.now(UTC).isoformat(),
        "model": {
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "classifier": "sklearn.ensemble.RandomForestClassifier",
            "parameters": {"n_estimators": 300, "random_state": 42, "n_jobs": -1},
            "classImbalanceHandling": "None; no validation or test resampling was performed.",
        },
        "featureContract": {
            "schemaVersion": FEATURE_SCHEMA_VERSION,
            "order": list(FEATURE_ORDER),
            "entropy": (
                "Unnormalized base-2 Shannon entropy over each changed file's share of total "
                "added plus deleted lines; zero for zero total changed lines. Verified exactly "
                "against three public Apache ant-ivy commits listed in entropyEvidence."
            ),
            "fixTrainingConversion": (
                "Exact booleans/0/1 or case-insensitive 'true'/'false' strings; unlike the "
                "source helper, the non-empty string 'False' is not converted with bool()."
            ),
            "entropyEvidence": [
                {
                    "commit": "2a5a07fcb1a24fded957d260f9a9df988323db19",
                    "includedFileChurn": [10, 30],
                    "expected": 0.8112781244591328,
                },
                {
                    "commit": "4642f1df280891c2238e74935239096fba508ba0",
                    "includedFileChurn": [33, 22, 2, 4],
                    "expected": 1.4295487875817467,
                },
                {
                    "commit": "36a1784d52126394afc5963d8139f8933bdd60c9",
                    "includedFileChurn": [20, 9, 35, 5, 73],
                    "expected": 1.8120032760083444,
                },
            ],
            "fileScopeFinding": (
                "The prepared metrics exclude some non-source files visible in public commit "
                "metadata. This is recorded as a limitation and must be resolved before the "
                "runtime extractor claims exact feature parity."
            ),
        },
        "dataset": {
            "name": "JIT-Fine replication package data.zip",
            "source": "https://github.com/jacknichao/JIT-Fine",
            "archiveSha256": DATASET_ARCHIVE_SHA256,
            "sourceFileSha256": EXPECTED_SOURCE_SHA256,
            "licenseReview": (
                "No licence file or GitHub licence declaration was present in the source "
                "repository when reviewed on 2026-08-30; raw data is not redistributed here."
            ),
            "joinKey": "commit_hash",
            "splits": {
                name: {
                    "rows": len(split.labels),
                    "classDistribution": split.class_distribution,
                    "sourceSha256": split.source_sha256,
                    "fixMessageComparison": split.fix_message_comparison,
                }
                for name, split in splits.items()
            },
            "integrityChecks": {
                "commitHashOneToOneJoin": "passed",
                "featureAndChangeLabelsAgree": "passed",
                "noDuplicateCommitHashes": "passed",
                "splitsDisjoint": "passed",
                "finiteSevenFeatureRows": "passed",
                "entropyWithinBase2ShannonCeiling": "passed",
            },
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pandas.__version__,
            "scikitLearn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "thresholdSelection": {"method": thresholds.method, "values": thresholds.as_dict()},
        "validationMetrics": validation_metrics,
        "untouchedTestMetrics": test_metrics,
        "limitations": list(LIMITATIONS),
        "researchBaselineSeparation": (
            "The JIT-Fine repository's published/reproduced 14-feature JITLine baseline is "
            "separate evidence and is not reported as this seven-feature model's result."
        ),
        "researchReproduction": {
            "command": "python -m baselines.JITLine.jitline_rq2 -style manual",
            "scope": "Original 14-feature JITLine manual baseline; not this model.",
            "observed": {
                "f1": 0.1525,
                "rocAuc": 0.6701,
                "pciAt20PercentLoc": 0.6147,
                "effortAt20PercentRecall": 0.0244,
                "pOpt": 0.8177,
            },
        },
    }
    metadata = export_artifact(
        model=model,
        report=report,
        thresholds=thresholds,
        output_dir=output_dir,
    )
    return {"metadata": metadata, "report": report}


def main() -> None:
    args = build_parser().parse_args()
    result = train(
        args.data_dir,
        args.output_dir,
        allow_unsafe_pickle=args.allow_unsafe_pickle,
    )
    print(json.dumps(result["metadata"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
