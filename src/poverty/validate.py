from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from poverty.data_io import (
    DataPaths,
    basic_schema_report,
    build_train_frame,
    check_keys_unique,
    load_test_features,
    feature_columns,
)
from poverty.metrics import score_all_surveys
from poverty.models import (
    ElasticNetConfig,
    HGBConfig,
    fit_predict_elasticnet_log,
    fit_predict_hgb_log,
    WeightMode,
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

        _blended, scores = score_all_surveys(
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


def loso_model_eval(
    train_df: pd.DataFrame,
    rates_gt: pd.DataFrame,
    *,
    model_name: str,
    predict_fn,
) -> pd.DataFrame:
    out_rows = []
    surveys = sorted(train_df["survey_id"].unique().tolist())

    for holdout_sid in surveys:
        tr = train_df[train_df["survey_id"] != holdout_sid].copy()
        va = train_df[train_df["survey_id"] == holdout_sid].copy()

        y_hat = predict_fn(tr, va)
        va["y_pred"] = y_hat

        _blended, scores = score_all_surveys(
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
                "model": model_name,
            }
        )

    return pd.DataFrame(out_rows).sort_values("holdout_survey_id")


def main() -> None:
    parser = argparse.ArgumentParser(prog="poverty.validate")
    parser.add_argument("--data-dir", type=str, required=True)

    parser.add_argument("--run-elasticnet", action="store_true")
    parser.add_argument("--run-hgb", action="store_true")

    parser.add_argument("--weight-mode", type=str, default="none", choices=["none", "weight", "sqrt_weight"])

    # ElasticNet args
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--l1-ratio", type=float, default=0.2)

    # HGB args
    parser.add_argument("--hgb-max-iter", type=int, default=400)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--hgb-max-depth", type=int, default=6)
    parser.add_argument("--hgb-min-samples-leaf", type=int, default=20)
    parser.add_argument("--hgb-l2", type=float, default=0.0)
    parser.add_argument("--hgb-early-stopping", action="store_true")

    args = parser.parse_args()

    # Default behavior: if neither flag set, run both
    if not args.run_elasticnet and not args.run_hgb:
        args.run_elasticnet = True
        args.run_hgb = True

    paths = DataPaths(data_dir=Path(args.data_dir))
    train_df = build_train_frame(paths)
    _test_df = load_test_features(paths)

    check_keys_unique(train_df)
    check_keys_unique(_test_df)

    rep = basic_schema_report(train_df, _test_df)
    print("Schema report:", rep)

    rates_gt = pd.read_csv(paths.train_rates).set_index("survey_id").sort_index()
    feat_cols = feature_columns(train_df)

    print("\nRunning LOSO constant baseline (sanity check)...")
    loso = loso_constant_baseline(train_df, rates_gt)
    print(loso.to_string(index=False))

    weight_mode: WeightMode = args.weight_mode  # type: ignore[assignment]

    if args.run_elasticnet:
        print("\nRunning LOSO ElasticNet baseline...")
        en_cfg = ElasticNetConfig(alpha=args.alpha, l1_ratio=args.l1_ratio)
        df_en = loso_model_eval(
            train_df,
            rates_gt,
            model_name=f"elasticnet_log({weight_mode})",
            predict_fn=lambda tr, va: fit_predict_elasticnet_log(
                tr, va, feat_cols, weight_mode=weight_mode, cfg=en_cfg
            ),
        )
        print(df_en.to_string(index=False))

    if args.run_hgb:
        print("\nRunning LOSO Boosted Trees baseline (HistGradientBoostingRegressor)...")
        hgb_cfg = HGBConfig(
            max_iter=args.hgb_max_iter,
            learning_rate=args.hgb_learning_rate,
            max_depth=args.hgb_max_depth,
            min_samples_leaf=args.hgb_min_samples_leaf,
            l2_regularization=args.hgb_l2,
            early_stopping=args.hgb_early_stopping,
        )
        df_hgb = loso_model_eval(
            train_df,
            rates_gt,
            model_name=f"hgb_log({weight_mode})",
            predict_fn=lambda tr, va: fit_predict_hgb_log(
                tr, va, feat_cols, cfg=hgb_cfg, weight_mode=weight_mode
            ),
        )
        print(df_hgb.to_string(index=False))

    print("\nDone. Next step: iterate HGB hyperparameters and compare against ElasticNet via LOSO + viz.")


if __name__ == "__main__":
    main()
