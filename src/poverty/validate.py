from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from poverty.data_io import (
    DataPaths,
    build_train_frame,
    load_test_features,
    check_keys_unique,
    basic_schema_report,
)
from poverty.metrics import score_all_surveys


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted median used for a constant baseline predictor.
    This is a sanity-check model to verify the LOSO + scorer plumbing.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    order = np.argsort(v)
    v_sorted = v[order]
    w_sorted = w[order]

    cdf = np.cumsum(w_sorted) / np.sum(w_sorted)
    idx = int(np.searchsorted(cdf, 0.5, side="left"))
    idx = min(max(idx, 0), len(v_sorted) - 1)
    return float(v_sorted[idx])


def loso_constant_baseline(train_df: pd.DataFrame, rates_gt: pd.DataFrame) -> pd.DataFrame:
    """
    Leave-one-survey-out (LOSO) evaluation with a constant predictor.
    For each held-out survey, fit a weighted median on the other surveys and predict that constant.
    """
    out_rows = []
    surveys = sorted(train_df["survey_id"].unique().tolist())

    for holdout_sid in surveys:
        tr = train_df[train_df["survey_id"] != holdout_sid].copy()
        va = train_df[train_df["survey_id"] == holdout_sid].copy()

        const_pred = weighted_median(
            tr["cons_ppp17"].to_numpy(),
            tr["weight"].to_numpy(),
        )
        va["y_pred"] = const_pred

        blended, scores = score_all_surveys(
            va,
            rates_gt,
            y_true_col="cons_ppp17",
            y_pred_col="y_pred",
            weight_col="weight",
        )

        # There is only one survey in this validation slice, so scores has length 1.
        s = scores[0]
        out_rows.append(
            {
                "holdout_survey_id": holdout_sid,
                "poverty_wmape": s.poverty_wmape,
                "household_mape": s.household_mape,
                "blended": s.blended,
                "const_pred": const_pred,
            }
        )

    return pd.DataFrame(out_rows).sort_values("holdout_survey_id")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to data/raw directory containing the CSVs.",
    )
    args = parser.parse_args()

    paths = DataPaths(data_dir=Path(args.data_dir))

    train_df = build_train_frame(paths)
    test_df = load_test_features(paths)

    check_keys_unique(train_df)
    check_keys_unique(test_df)

    # Quick schema report (helps verify train/test look as expected)
    rep = basic_schema_report(train_df, test_df)
    print("Schema report:", rep)

    rates_gt = pd.read_csv(paths.train_rates).set_index("survey_id").sort_index()

    print("\nRunning LOSO constant baseline (sanity check)...")
    loso = loso_constant_baseline(train_df, rates_gt)
    print(loso.to_string(index=False))

    print("\nDone. Next step is to replace the constant baseline with real models.")


if __name__ == "__main__":
    main()
