from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MISSING_TOKEN = "__MISSING__"


@dataclass(frozen=True)
class DataPaths:
    data_dir: Path

    @property
    def train_features(self) -> Path:
        return self.data_dir / "train_hh_features.csv"

    @property
    def train_gt(self) -> Path:
        return self.data_dir / "train_hh_gt.csv"

    @property
    def train_rates(self) -> Path:
        return self.data_dir / "train_rates_gt.csv"

    @property
    def test_features(self) -> Path:
        return self.data_dir / "test_hh_features.csv"


ID_COLS = ["survey_id", "hhid"]


def _normalize_object_columns(df: pd.DataFrame, missing_token: str = MISSING_TOKEN) -> pd.DataFrame:
    """
    Normalize string columns to be consistent across train/test.
    - strip leading/trailing whitespace
    - coerce empty strings to missing token
    - fill NaN with missing token

    This neutralizes the observed test-only category with leading whitespace in `sector1d`.
    """
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        s = out[c].astype("string")
        s = s.str.strip()  # critical for test-only whitespace issues
        s = s.fillna(missing_token)
        s = s.replace("", missing_token)
        out[c] = s.astype("string")
    return out


def load_train_features(paths: DataPaths) -> pd.DataFrame:
    df = pd.read_csv(paths.train_features)
    df = _normalize_object_columns(df)
    _assert_required_columns(df, required=set(ID_COLS + ["weight"]))
    return df


def load_train_gt(paths: DataPaths) -> pd.DataFrame:
    df = pd.read_csv(paths.train_gt)
    _assert_required_columns(df, required=set(ID_COLS + ["cons_ppp17"]))
    return df


def load_train_rates(paths: DataPaths) -> pd.DataFrame:
    df = pd.read_csv(paths.train_rates)
    _assert_required_columns(df, required={"survey_id"})
    return df.set_index("survey_id").sort_index()


def load_test_features(paths: DataPaths) -> pd.DataFrame:
    df = pd.read_csv(paths.test_features)
    df = _normalize_object_columns(df)
    _assert_required_columns(df, required=set(ID_COLS + ["weight"]))
    return df


def build_train_frame(paths: DataPaths) -> pd.DataFrame:
    """
    Returns a single training frame with features + ground-truth consumption.
    Enforces one-to-one merge on (survey_id, hhid).
    """
    feat = load_train_features(paths)
    gt = load_train_gt(paths)
    df = feat.merge(gt, on=ID_COLS, how="inner", validate="one_to_one")
    if len(df) != len(feat):
        raise ValueError(f"Train merge dropped rows: features={len(feat)} merged={len(df)}")

    # Drop constant column if present
    if "com" in df.columns and df["com"].nunique(dropna=False) == 1:
        df = df.drop(columns=["com"])

    return df


def feature_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    """
    Returns columns intended for modeling.
    We exclude IDs and the label. We also exclude `survey_id` from predictors by default
    because test surveys contain unseen IDs and it can destabilize generalization.
    """
    exclude_set = set(exclude) | {"cons_ppp17"} | set(ID_COLS)
    cols = [c for c in df.columns if c not in exclude_set]
    return cols


def _assert_required_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def check_keys_unique(df: pd.DataFrame) -> None:
    if df.duplicated(ID_COLS).any():
        dups = df[df.duplicated(ID_COLS, keep=False)][ID_COLS].head(10)
        raise ValueError(f"Duplicate keys found for (survey_id, hhid). Examples:\n{dups}")


def basic_schema_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Convenience function for logging/debugging.
    """
    return {
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "train_surveys": sorted(train_df["survey_id"].unique().tolist()),
        "test_surveys": sorted(test_df["survey_id"].unique().tolist()),
        "n_train_object_cols": int(train_df.select_dtypes(include=["object", "string"]).shape[1]),
        "n_test_object_cols": int(test_df.select_dtypes(include=["object", "string"]).shape[1]),
    }
