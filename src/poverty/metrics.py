from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


THRESHOLDS: list[float] = [
    3.17, 3.94, 4.60, 5.26, 5.88, 6.47, 7.06, 7.70, 8.40,
    9.13, 9.87, 10.70, 11.62, 12.69, 14.03, 15.64, 17.76, 20.99, 27.37
]

# These thresholds correspond to ventiles => percentile ranks 0.05..0.95
PERCENTILE_RANKS = np.arange(0.05, 1.00, 0.05)
THRESHOLD_WEIGHTS = 1.0 - np.abs(0.4 - PERCENTILE_RANKS)  # official weighting definition


def poverty_rate_column_name(threshold: float) -> str:
    # Matches train_rates_gt.csv: pct_hh_below_3.17 etc (two decimals)
    return f"pct_hh_below_{threshold:.2f}"


def poverty_rate_columns(thresholds: Iterable[float] = THRESHOLDS) -> list[str]:
    return [poverty_rate_column_name(t) for t in thresholds]


def poverty_rates_from_consumption(
    cons: np.ndarray,
    weights: np.ndarray,
    thresholds: Iterable[float] = THRESHOLDS,
) -> np.ndarray:
    """
    Weighted poverty rates using strict inequality:
      pct = sum_i w_i * 1(cons_i < threshold) / sum_i w_i
    """
    cons = np.asarray(cons, dtype=float)
    w = np.asarray(weights, dtype=float)
    ws = float(w.sum())
    if not np.isfinite(ws) or ws <= 0:
        raise ValueError(f"Invalid weight sum: {ws}")

    rates = []
    for t in thresholds:
        rates.append(float(w[cons < t].sum() / ws))
    return np.array(rates, dtype=float)


def poverty_wmape(r_pred: np.ndarray, r_true: np.ndarray) -> float:
    """
    Weighted mean absolute percentage error across thresholds.
    """
    r_pred = np.asarray(r_pred, dtype=float)
    r_true = np.asarray(r_true, dtype=float)

    if r_pred.shape != r_true.shape:
        raise ValueError(f"Shape mismatch: r_pred={r_pred.shape} r_true={r_true.shape}")

    # Avoid division by zero (shouldn't happen here, but defensive)
    if np.any(r_true <= 0):
        raise ValueError("r_true contains non-positive values; cannot compute MAPE.")

    num = (THRESHOLD_WEIGHTS * np.abs(r_pred - r_true) / r_true).sum()
    den = THRESHOLD_WEIGHTS.sum()
    return float(num / den)


def household_mape(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if np.any(y_true <= 0):
        raise ValueError("y_true contains non-positive values; cannot compute MAPE.")
    return float(np.mean(np.abs(y_pred - y_true) / y_true))


def blended_competition_metric(
    poverty_wmape_value: float,
    household_mape_value: float,
) -> float:
    # Official blend: 90% poverty + 10% household
    return float(90.0 * poverty_wmape_value + 10.0 * household_mape_value)


@dataclass(frozen=True)
class SurveyScore:
    survey_id: int
    poverty_wmape: float
    household_mape: float
    blended: float


def score_one_survey(
    survey_id: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    poverty_rates_true: np.ndarray,
) -> SurveyScore:
    r_pred = poverty_rates_from_consumption(y_pred, weights)
    pw = poverty_wmape(r_pred, poverty_rates_true)
    hm = household_mape(y_pred, y_true)
    blended = blended_competition_metric(pw, hm)
    return SurveyScore(survey_id=survey_id, poverty_wmape=pw, household_mape=hm, blended=blended)


def score_all_surveys(
    df: pd.DataFrame,
    rates_gt: pd.DataFrame,
    y_true_col: str = "cons_ppp17",
    y_pred_col: str = "y_pred",
    weight_col: str = "weight",
) -> tuple[float, list[SurveyScore]]:
    """
    Scores a dataframe containing multiple surveys.
    `rates_gt` must be indexed by survey_id with poverty-rate columns.
    """
    required = {"survey_id", y_true_col, y_pred_col, weight_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required df columns: {sorted(missing)}")

    rate_cols = poverty_rate_columns()
    if any(c not in rates_gt.columns for c in rate_cols):
        missing_cols = [c for c in rate_cols if c not in rates_gt.columns]
        raise ValueError(f"rates_gt missing poverty columns: {missing_cols}")

    scores: list[SurveyScore] = []
    for sid, g in df.groupby("survey_id", sort=True):
        poverty_true = rates_gt.loc[int(sid), rate_cols].to_numpy(dtype=float)
        s = score_one_survey(
            survey_id=int(sid),
            y_true=g[y_true_col].to_numpy(dtype=float),
            y_pred=g[y_pred_col].to_numpy(dtype=float),
            weights=g[weight_col].to_numpy(dtype=float),
            poverty_rates_true=poverty_true,
        )
        scores.append(s)

    blended_mean = float(np.mean([s.blended for s in scores])) if scores else float("nan")
    return blended_mean, scores
