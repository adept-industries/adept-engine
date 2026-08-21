"""Local CLI interface for PR Risk Analytics training, evaluation, and demoing."""

import argparse
import contextlib
import json
from typing import Any

from app.db.session import get_database_engine
from app.risk.dx import compute_dx_score
from app.risk.model import risk_model
from app.risk.service import predict_and_persist
from app.risk.stale import check_stale
from app.risk.synthetic import generate_synthetic_pr_dataset
from app.risk.trainer import train_risk_model


def run_train_demo() -> dict[str, Any]:
    """Generates synthetic data, trains model, calibrates, tunes thresholds, and registers."""
    print("\n" + "=" * 70)
    print("  PR RISK ANALYTICS — TRAINING PIPELINE (DEMO MODE)")
    print("=" * 70)
    print(">> Generating 5,000 isolated chronological synthetic PR snapshots...")
    df = generate_synthetic_pr_dataset(n_samples=5000, seed=42)
    print(f">> Dataset generated: {len(df)} records. Positive rate: {df['is_risky'].mean():.2%}")

    print(">> Splitting chronologically: 60% Train, 20% Calibration, 20% Untouched Test...")
    print(">> Fitting XGBClassifier (hist, binary:logistic, scale_pos_weight)...")
    print(">> Fitting LogisticRegression Calibrator on separate calibration split...")
    print(">> Tuning HIGH and MEDIUM alert thresholds on validation data (favoring precision)...")
    print(">> Evaluating final performance on untouched test set...")

    db_engine = None
    with contextlib.suppress(Exception):
        db_engine = get_database_engine()

    result = train_risk_model(df, is_demo=True, database_engine=db_engine)

    print("\n" + "-" * 70)
    print("  MODEL EVALUATION METRICS")
    print("  [DEMO DATA — NOT REAL-WORLD MODEL ACCURACY]")
    print("-" * 70)
    metrics = result["metrics"]
    print(f"Model Version        : {result['model_version']}")
    print(f"ROC-AUC              : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC (Avg Prec)    : {metrics['pr_auc']:.4f}")
    print(f"Brier Score          : {metrics['brier_score']:.4f}")
    print(f"HIGH Alert Threshold : {result['high_threshold']:.2f}")
    print(f"MEDIUM Alert Thresh  : {result['medium_threshold']:.2f}")
    print(f"Precision @ HIGH     : {metrics['precision_at_high']:.2%}")
    print(f"Recall @ HIGH        : {metrics['recall_at_high']:.2%}")
    print(f"F1 @ HIGH            : {metrics['f1_at_high']:.4f}")
    print(f"Test Positive Rate   : {metrics['test_positive_rate']:.2%}")
    print(f"Confusion Matrix     : {metrics['confusion_matrix']}")
    print("-" * 70)

    return result


def run_demo() -> None:
    """Executes full end-to-end demo proving training, inference, SHAP, stale check, DX, DB."""
    # 1. Train and save model
    run_train_demo()

    # 2. Load model
    risk_model.load()

    # 3. Sample PR prediction
    print("\n" + "=" * 70)
    print("  1. REAL-TIME RISK PREDICTION & SHAP EXPLANATION")
    print("=" * 70)
    sample_pr_features = {
        "lines_added": 850,
        "lines_deleted": 420,
        "files_changed": 28,
        "commit_count": 12,
        "source_files_changed": 22,
        "test_files_changed": 1,
        "dependency_files_changed": 2,
        "hotspot_score": 0.85,
        "recent_file_bugfix_rate": 0.45,
        "recent_file_change_rate": 0.70,
        "author_file_familiarity": 0.15,
        "author_repo_experience": 0.20,
        "ci_failures": 2,
        "changes_requested": 1,
        "review_comment_count": 8,
        "review_rounds": 3,
    }

    prediction = risk_model.predict(sample_pr_features)
    added = sample_pr_features["lines_added"]
    deleted = sample_pr_features["lines_deleted"]
    files = sample_pr_features["files_changed"]
    print(f"Input Changes        : +{added} / -{deleted} across {files} files")
    print(f"Calibrated Risk Prob : {prediction['probability']:.2%}")
    print(f"Alert Risk Level     : {prediction['risk_level']}")
    print(f"Model Version        : {prediction['model_version']}")
    print("\nTop 5 Contributing Factors (SHAP Explanations):")
    for factor in prediction["top_factors"]:
        direction_symbol = "▲" if factor["direction"] == "raises_risk" else "▼"
        fname = factor["feature"]
        fval = factor["value"]
        fdir = factor["direction"]
        fimp = factor["impact"]
        print(f"  {direction_symbol} {fname:<26} = {fval:<8} ({fdir}, impact={fimp:+.4f})")

    # 4. Stale PR Check
    print("\n" + "=" * 70)
    print("  2. DETERMINISTIC STALE PR DETECTION (NON-ML)")
    print("=" * 70)
    stale_result = check_stale(is_open=True, hours_since_last_activity=148.5, threshold_hours=120.0)
    print("PR Status            : OPEN")
    hrs = stale_result["hours_since_last_activity"]
    thresh = stale_result["threshold_hours"]
    print(f"Inactivity Hours     : {hrs}h (Threshold: {thresh}h)")
    print(f"Is Stale             : {stale_result['is_stale']}")
    print(f"Reason               : {stale_result['reason']}")

    # 5. Developer Experience Score
    print("\n" + "=" * 70)
    print("  3. DEVELOPER EXPERIENCE (DX) WORKFLOW SCORE")
    print("=" * 70)
    dx = compute_dx_score(
        median_first_review_hours=6.5,
        median_pr_cycle_hours=38.0,
        stale_pr_rate=0.06,
        change_failure_rate=0.05,
        ci_success_rate=0.96,
    )
    print(f"Repository DX Score  : {dx['score']} / 100.0")
    print("Component Breakdown:")
    for comp, score in dx["components"].items():
        weight = dx["weights"][comp]
        print(f"  - {comp:<20} : {score:>5.1f}/100 (weight: {weight:.2f})")

    # 6. Database persistence test
    print("\n" + "=" * 70)
    print("  4. DATABASE PERSISTENCE VERIFICATION")
    print("=" * 70)
    try:
        db_engine = get_database_engine()
        # Find a repository and PR in DB to persist demo prediction
        from sqlalchemy import text

        with db_engine.connect() as conn:
            row = (
                conn.execute(text("SELECT repository_id, number FROM pull_requests LIMIT 1"))
                .mappings()
                .one_or_none()
            )
        if row:
            db_pred = predict_and_persist(
                database_engine=db_engine,
                repo_identifier=str(row["repository_id"]),
                pr_number=int(row["number"]),
                custom_features=sample_pr_features,
                stage="live",
            )
            print(">> Successfully persisted live prediction to PostgreSQL!")
            print(f">> Repository UUID : {db_pred['repository_id']}")
            print(f">> PR Number       : #{db_pred['pr_number']}")
            print(
                f">> Risk Score      : {db_pred['risk_probability']:.4f} ({db_pred['risk_level']})"
            )
        else:
            print(">> PostgreSQL connected (no existing PR rows to attach sample prediction to).")
    except Exception as exc:
        print(f">> Database check note: {exc}")

    print("\n" + "=" * 70)
    print("  PR RISK ANALYTICS DEMO COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="PR Risk Analytics CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("train-demo", help="Generate demo data and train model")
    subparsers.add_parser("demo", help="Run full end-to-end demo")
    subparsers.add_parser("demo-predict", help="Run sample prediction with SHAP explanation")
    subparsers.add_parser("safe", help="Simulate a Safe Pull Request prediction")
    subparsers.add_parser("risky", help="Simulate a Risky Pull Request prediction")

    args = parser.parse_args()

    if args.command == "train-demo":
        run_train_demo()
    elif args.command == "demo":
        run_demo()
    elif args.command == "safe":
        if not risk_model.load():
            run_train_demo()
            risk_model.load()
        safe_features = {
            "lines_added": 12,
            "lines_deleted": 2,
            "files_changed": 1,
            "commit_count": 1,
            "source_files_changed": 0,
            "test_files_changed": 1,
            "dependency_files_changed": 0,
            "hotspot_score": 0.05,
            "recent_file_bugfix_rate": 0.02,
            "recent_file_change_rate": 0.05,
            "author_file_familiarity": 0.95,
            "author_repo_experience": 0.90,
            "ci_failures": 0,
            "changes_requested": 0,
            "review_comment_count": 1,
            "review_rounds": 1,
        }
        res = risk_model.predict(safe_features)
        print("========================================")
        print("           SAFE PR PREDICTION           ")
        print("========================================")
        print(f"Risk Probability : {res['probability'] * 100:.1f}%")
        print(f"Risk Level       : {res['risk_level']}")
        print("Top Contributing SHAP Factors:")
        for f in res["top_factors"]:
            sym = "▲ Raises" if f["direction"] == "raises_risk" else "▼ Lowers"
            print(
                f"  {sym} Risk: {f['feature']} (val={f['value']}) | SHAP impact: {f['impact']:+.4f}"
            )
        print("========================================")
    elif args.command == "risky":
        if not risk_model.load():
            run_train_demo()
            risk_model.load()
        risky_features = {
            "lines_added": 950,
            "lines_deleted": 480,
            "files_changed": 25,
            "commit_count": 8,
            "source_files_changed": 20,
            "test_files_changed": 2,
            "dependency_files_changed": 3,
            "hotspot_score": 0.85,
            "recent_file_bugfix_rate": 0.45,
            "recent_file_change_rate": 0.75,
            "author_file_familiarity": 0.05,
            "author_repo_experience": 0.10,
            "ci_failures": 2,
            "changes_requested": 1,
            "review_comment_count": 18,
            "review_rounds": 3,
        }
        res = risk_model.predict(risky_features)
        print("========================================")
        print("          RISKY PR PREDICTION           ")
        print("========================================")
        print(f"Risk Probability : {res['probability'] * 100:.1f}%")
        print(f"Risk Level       : {res['risk_level']}")
        print("Top Contributing SHAP Factors:")
        for f in res["top_factors"]:
            sym = "▲ Raises" if f["direction"] == "raises_risk" else "▼ Lowers"
            print(
                f"  {sym} Risk: {f['feature']} (val={f['value']}) | SHAP impact: {f['impact']:+.4f}"
            )
        print("========================================")
    elif args.command == "demo-predict":
        if not risk_model.load():
            run_train_demo()
            risk_model.load()
        sample_pr_features = {
            "lines_added": 500,
            "lines_deleted": 120,
            "files_changed": 15,
            "commit_count": 6,
            "source_files_changed": 10,
            "test_files_changed": 4,
            "dependency_files_changed": 1,
            "hotspot_score": 0.45,
            "recent_file_bugfix_rate": 0.20,
            "recent_file_change_rate": 0.35,
            "author_file_familiarity": 0.60,
            "author_repo_experience": 0.75,
            "ci_failures": 0,
            "changes_requested": 0,
            "review_comment_count": 4,
            "review_rounds": 1,
        }
        res = risk_model.predict(sample_pr_features)
        print(json.dumps(res, indent=2, default=str))
    else:
        # Default to running demo
        run_demo()


if __name__ == "__main__":
    main()
