"""Subject-level feature aggregation.

The MODMA/Lanzhou dataset has ~29 correlated recordings per subject (see
backend/notebooks/DATA_CONTEXT.md). Modeling at the file level without grouping
leaks a subject's recordings across train/test; with only 52 subjects total, the
simplest leakage-safe unit is the subject itself. This module builds that
subject-level feature table from the per-file `features.csv` produced by
FeatureExtractor.
"""
from pathlib import Path

import pandas as pd

from .data_loader import classify_task


def build_subject_level_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-file features to one row per subject.

    Args:
        features_df: DataFrame as produced by FeatureExtractor.extract_features_batch
            / load_features — one row per audio file, with a 'label' column
            (1=MDD, 0=HC) and a 'file_path' column, plus numeric feature columns.

    Returns:
        DataFrame with one row per subject: subject_id, label, type ('MDD'/'HC'),
        and mean/std of every input feature across that subject's files
        (columns suffixed '_mean' / '_std').
    """
    df = features_df.copy()
    df["subject_id"] = df["file_path"].apply(lambda p: Path(p).parent.name)
    df["file_index"] = df["file_path"].apply(lambda p: int(Path(p).stem))
    df["task_type"] = df["file_index"].apply(classify_task)

    feature_cols = [
        c for c in df.columns
        if c not in ("label", "file_path", "subject_id", "file_index", "task_type")
    ]

    agg = df.groupby("subject_id")[feature_cols].agg(["mean", "std"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]

    subject_label = df.groupby("subject_id")["label"].first()
    subject_df = agg.join(subject_label).reset_index()
    subject_df["type"] = subject_df["label"].map({1: "MDD", 0: "HC"})

    return subject_df
