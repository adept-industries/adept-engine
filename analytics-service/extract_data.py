from pathlib import Path

import pandas as pd

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


def find_dataset_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "JIT-Fine" / "data" / "jitfine",
        Path("/home/teshaaan/Projects/JIT-Fine/data/jitfine"),
        Path.cwd() / "data" / "jitfine",
    ]
    for c in candidates:
        if (c / "features_train.pkl").exists() and (c / "features_test.pkl").exists():
            return c
    locs = [str(c) for c in candidates]
    raise FileNotFoundError(f"Could not locate JIT-Fine dataset in: {locs}")


def parse_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fix"] = df["fix"].apply(lambda x: float(bool(x)))
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["label"] = pd.to_numeric(df["is_buggy_commit"], errors="coerce").astype(int)
    return df[FEATURES + ["label"]]


def main():
    service_dir = Path(__file__).resolve().parent
    data_dir = find_dataset_dir()
    print(f"Loading datasets from: {data_dir}")

    train_raw = pd.read_pickle(data_dir / "features_train.pkl")
    test_raw = pd.read_pickle(data_dir / "features_test.pkl")

    train_df = parse_features_dataframe(train_raw)
    test_df = parse_features_dataframe(test_raw)

    train_csv_path = service_dir / "train.csv"
    test_csv_path = service_dir / "test.csv"

    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print(f"Extracted Train dataset: {train_df.shape} -> {train_csv_path}")
    print(f"  Defect distribution: {dict(train_df['label'].value_counts())}")
    print(f"Extracted Test dataset:  {test_df.shape} -> {test_csv_path}")
    print(f"  Defect distribution: {dict(test_df['label'].value_counts())}")
    print("Data extraction completed successfully.")


if __name__ == "__main__":
    main()
