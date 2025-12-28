from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poverty.data_io import DataPaths, build_train_frame
from poverty.metrics import THRESHOLDS, poverty_rate_columns, poverty_rates_from_consumption


@pytest.mark.parametrize("data_dir", ["data/raw"])
def test_recomputed_poverty_rates_match_ground_truth(data_dir: str) -> None:
    paths = DataPaths(data_dir=Path(data_dir))

    # Skip gracefully if the user hasn't placed data yet
    if not paths.train_features.exists():
        pytest.skip(f"Missing {paths.train_features}. Place CSVs in {paths.data_dir}.")
    if not paths.train_gt.exists():
        pytest.skip(f"Missing {paths.train_gt}. Place CSVs in {paths.data_dir}.")
    if not paths.train_rates.exists():
        pytest.skip(f"Missing {paths.train_rates}. Place CSVs in {paths.data_dir}.")

    train_df = build_train_frame(paths)
    rates_gt = pd.read_csv(paths.train_rates).set_index("survey_id").sort_index()

    rate_cols = poverty_rate_columns()
    assert all(c in rates_gt.columns for c in rate_cols)

    # Recompute poverty rates from *true* consumption
    recomputed = {}
    for sid, g in train_df.groupby("survey_id", sort=True):
        r = poverty_rates_from_consumption(
            cons=g["cons_ppp17"].to_numpy(),
            weights=g["weight"].to_numpy(),
            thresholds=THRESHOLDS,
        )
        recomputed[int(sid)] = r

    recomputed_df = pd.DataFrame.from_dict(
        recomputed,
        orient="index",
        columns=rate_cols,
    ).sort_index()

    gt_df = rates_gt[rate_cols].sort_index()

    # Should match to floating point tolerance
    max_abs = float((recomputed_df - gt_df).abs().to_numpy().max())
    assert max_abs < 1e-12, f"Max abs diff too large: {max_abs}"
