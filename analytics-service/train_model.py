import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score

FEATURES = [
    "la",
    "ld",
    "nf",
    "ns",
    "nd",
    "entropy",
    "ndev",
    "lt",
    "nuc",
    "age",
    "exp",
    "rexp",
    "sexp",
    "fix",
]


def main():
    service_dir = Path(__file__).resolve().parent
    train_path = service_dir / "train.csv"
    test_path = service_dir / "test.csv"
    model_path = service_dir / "pr_risk_model.joblib"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Train/Test CSVs missing. Expected at:\n  {train_path}\n  {test_path}\n"
            "Please run extract_data.py first."
        )

    print(f"Loading data from {service_dir}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[FEATURES]
    y_train = train_df["label"]

    X_test = test_df[FEATURES]
    y_test = test_df["label"]

    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set:     {X_test.shape[0]} samples, {X_test.shape[1]} features")

    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Predictions and probabilities
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 40)
    print("        EVALUATION METRICS")
    print("=" * 40)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("=" * 40 + "\n")

    # Save model artifact
    joblib.dump(clf, model_path)
    print(f"Model saved successfully to {model_path}")


if __name__ == "__main__":
    main()
