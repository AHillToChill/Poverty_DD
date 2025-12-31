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
    feature_columns,
)
from poverty.metrics import score_all_surveys
from poverty.models import (
    ElasticNetConfig,
    HGBConfig,
    WeightMode,
    fit_predict_elasticnet_log,
    fit_predict_hgb_log,
)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
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
    out_rows = []
    surveys = sorted(train_df["survey_id"].unique().tolist())

    for holdout_sid in surveys:
        tr = train_df[train_df["survey_id"] != holdout_sid].copy()
        va = train_df[train_df["survey_id"] == holdout_sid].copy()

        const_pred = weighted_median(tr["cons_ppp17"].to_numpy(), tr["weight"].to_numpy())
        va["y_pred"] = const_pred

        _, scores = score_all_surveys(
            va,
            rates_gt,
            y_true_col="cons_ppp17",
            y_pred_col="y_pred",
            weight_col="weight",
        )

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


def loso_model_baseline(
    train_df: pd.DataFrame,
    rates_gt: pd.DataFrame,
    *,
    model_name: str,
    weight_mode: WeightMode,
    fit_predict_fn,
) -> pd.DataFrame:
    out_rows = []
    surveys = sorted(train_df["survey_id"].unique().tolist())
    feat_cols = feature_columns(train_df)

    for holdout_sid in surveys:
        tr = train_df[train_df["survey_id"] != holdout_sid].copy()
        va = train_df[train_df["survey_id"] == holdout_sid].copy()

        y_hat = fit_predict_fn(tr, va, feat_cols, weight_mode=weight_mode)

        va = va.copy()
        va["y_pred"] = y_hat

        _, scores = score_all_surveys(
            va,
            rates_gt,
            y_true_col="cons_ppp17",
            y_pred_col="y_pred",
            weight_col="weight",
        )

        s = scores[0]
        out_rows.append(
            {
                "holdout_survey_id": holdout_sid,
                "poverty_wmape": s.poverty_wmape,
                "household_mape": s.household_mape,
                "blended": s.blended,
                "model": f"{model_name}({weight_mode})",
            }
        )

    return pd.DataFrame(out_rows).sort_values("holdout_survey_id")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    args = parser.parse_args()

    paths = DataPaths(data_dir=Path(args.data_dir))

    train_df = build_train_frame(paths)
    test_df = load_test_features(paths)

    check_keys_unique(train_df)
    check_keys_unique(test_df)

    rep = basic_schema_report(train_df, test_df)
    print("Schema report:", rep)

    rates_gt = pd.read_csv(paths.train_rates).set_index("survey_id").sort_index()

    print("\nRunning LOSO constant baseline (sanity check)...")
    loso = loso_constant_baseline(train_df, rates_gt)
    print(loso.to_string(index=False))

    print("\nRunning LOSO ElasticNet baseline...")
    for wm in ["none", "weight", "sqrt_weight"]:
        print(f"\nElasticNet weight_mode={wm}")
        cfg = ElasticNetConfig(alpha=0.01, l1_ratio=0.2, max_iter=5000, random_state=42)

        def _fn(tr, va, feat_cols, *, weight_mode: WeightMode):
            return fit_predict_elasticnet_log(tr, va, feat_cols, weight_mode=weight_mode, cfg=cfg)

        df = loso_model_baseline(train_df, rates_gt, model_name="elasticnet_log", weight_mode=wm, fit_predict_fn=_fn)
        print(df.to_string(index=False))

    print("\nRunning LOSO Boosted Trees baseline (HistGradientBoostingRegressor)...")
    hgb_cfg = HGBConfig(max_iter=600, learning_rate=0.05, max_depth=6, min_samples_leaf=20, l2_regularization=0.0)

    def _hgb(tr, va, feat_cols, *, weight_mode: WeightMode):
        return fit_predict_hgb_log(tr, va, feat_cols, weight_mode=weight_mode, cfg=hgb_cfg)

    df = loso_model_baseline(train_df, rates_gt, model_name="hgb_log", weight_mode="none", fit_predict_fn=_hgb)
    print(df.to_string(index=False))

    print("\nDone. Next step: iterate HGB hyperparameters and compare via LOSO + viz.")


if __name__ == "__main__":
    main()
