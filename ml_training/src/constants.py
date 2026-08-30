from __future__ import annotations

MODEL_NAME = "jitfine-expert-pr-risk-mvp"
MODEL_VERSION = "jitfine-expert-pr-risk-mvp-v1"
FEATURE_SCHEMA_VERSION = "jitfine-pr-features-v1"

FEATURE_ORDER = ("ns", "nd", "nf", "entropy", "la", "ld", "fix")
SPLITS = ("train", "valid", "test")

DATASET_ARCHIVE_SHA256 = "9e5ca1a393b70ee7e87c410b162005958775f3f3732f9f83da9dd24a7dfe2b47"
EXPECTED_SOURCE_SHA256 = {
    "changes_train.pkl": "088898f4c87a59dfeaaabdbc48488b7e224ea7109ee453d0e8d244146b715f85",
    "changes_valid.pkl": "dc9bb8f3390b74b5329a2186332466c22d78e71047521c56f1d8b598147c723b",
    "changes_test.pkl": "23107219e63648b6516d74ff73b1e696c5e8eb368f6766c10e88d61dadb0809d",
    "features_train.pkl": "0b9992d77cd07f45a71bd327a3d408ed3f0ab7a907ffdc10f9916a8f226dc42a",
    "features_valid.pkl": "e2ee48d5cd9e7f59bea726d4f6a71ef58712ae58d3388d8195a798369cf0d2cf",
    "features_test.pkl": "136eebbb1c1719bffbfed8f0d686e3c015c5756a7b3c43d6f700a07687ab58fa",
    "dataset_dict.pkl": "9006b36e814bbafcab2b7f82c443b4d117ec865836e706c3c9378737b688f5eb",
}
