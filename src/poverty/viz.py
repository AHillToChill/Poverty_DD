from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # write PNGs without needing a GUI backend
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from poverty.data_io import DataPaths, build_train_frame, feature_columns
from poverty.metrics import THRESHOLDS, poverty_rates_from_consumption, score_all_surveys
from poverty.models import fit_predict_elasticnet_log, ElasticNetConfig


# Thresholds are approximately ventiles (5%, 10%, ..., 95%) from the reference survey.
# Used only for visualization aids (metric emphasis band near 40%).
PERCENTILE_RANKS = np.asarray([i / 100 for i in range(5, 100, 5)], dtype=float)


@dataclass(frozen=True)
class OOFResult:
    df: pd.DataFrame  # survey_id, hhid, weight, cons_ppp17, y_pred
    model_name: str


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sample_df(df: pd.DataFrame, n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def _load_rates_gt(paths: DataPaths) -> pd.DataFrame:
    return pd.read_csv(paths.train_rates).set_index("survey_id").sort_index()


def make_oof_predictions_elasticnet(
    train_df: pd.DataFrame,
    weight_mode: str,
    cfg: ElasticNetConfig,
) -> OOFResult:
    surveys = sorted(train_df["survey_id"].unique().tolist())
    feat_cols = feature_columns(train_df)

    parts: list[pd.DataFrame] = []
    for holdout_sid in surveys:
        tr = train_df[train_df["survey_id"] != holdout_sid].copy()
        va = train_df[train_df["survey_id"] == holdout_sid].copy()

        y_hat = fit_predict_elasticnet_log(
            train_df=tr,
            valid_df=va,
            feature_cols=feat_cols,
            weight_mode=weight_mode,
            cfg=cfg,
        )

        oof = va[["survey_id", "hhid", "weight", "cons_ppp17"]].copy()
        oof["y_pred"] = y_hat
        parts.append(oof)

    out = pd.concat(parts, axis=0, ignore_index=True)
    model_name = f"elasticnet_log({weight_mode})"
    return OOFResult(df=out, model_name=model_name)


def _poverty_rates_by_survey(oof: pd.DataFrame, thresholds: Iterable[float], col: str) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for sid, g in oof.groupby("survey_id", sort=True):
        r = poverty_rates_from_consumption(
            cons=g[col].to_numpy(dtype=float),
            weights=g["weight"].to_numpy(dtype=float),
            thresholds=thresholds,
        )
        out[int(sid)] = r
    return out


def _metric_emphasis_band(thresholds: np.ndarray) -> tuple[float, float, float]:
    """
    Returns (x_left, x_right, x_center) for the ~35%-45% region, centered near 40%,
    to visually indicate where the competition weights peak.
    """
    if len(PERCENTILE_RANKS) != len(thresholds):
        # Fallback: choose center by closest threshold to ~$7.70 if list sizes mismatch.
        x_center = float(thresholds[np.argmin(np.abs(thresholds - 7.70))])
        return x_center, x_center, x_center

    i35 = int(np.argmin(np.abs(PERCENTILE_RANKS - 0.35)))
    i45 = int(np.argmin(np.abs(PERCENTILE_RANKS - 0.45)))
    i40 = int(np.argmin(np.abs(PERCENTILE_RANKS - 0.40)))

    x_left = float(min(thresholds[i35], thresholds[i45]))
    x_right = float(max(thresholds[i35], thresholds[i45]))
    x_center = float(thresholds[i40])
    return x_left, x_right, x_center


def _metrics_by_survey(rep: pd.DataFrame) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for _, r in rep.iterrows():
        sid = r["survey_id"]
        if isinstance(sid, str):
            continue
        out[int(sid)] = {
            "poverty_wmape": float(r["poverty_wmape"]),
            "household_mape": float(r["household_mape"]),
            "blended": float(r["blended"]),
        }
    return out


def plot_consumption_pred_vs_true_grid(
    oof: pd.DataFrame,
    out_dir: Path,
    model_name: str,
    metrics: dict[int, dict[str, float]] | None = None,
) -> None:
    _ensure_dir(out_dir)

    # Shared axis limits in original units (log scales), to enable easy comparison.
    y_all = np.concatenate(
        [
            oof["cons_ppp17"].to_numpy(dtype=float),
            oof["y_pred"].to_numpy(dtype=float),
        ]
    )
    y_all = y_all[np.isfinite(y_all) & (y_all > 0)]
    lo = float(np.quantile(y_all, 0.005))
    hi = float(np.quantile(y_all, 0.995))
    lo = max(lo, 1e-6)

    surveys = sorted(oof["survey_id"].unique().tolist())
    fig, axes = plt.subplots(1, len(surveys), figsize=(15, 4.8), sharex=True, sharey=True)

    if len(surveys) == 1:
        axes = [axes]

    for ax, sid in zip(axes, surveys):
        g = oof[oof["survey_id"] == sid]
        gg = _sample_df(g, n=12_000, seed=42)

        y_true = gg["cons_ppp17"].to_numpy(dtype=float)
        y_pred = gg["y_pred"].to_numpy(dtype=float)

        ax.scatter(y_true, y_pred, s=6, alpha=0.20)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        # Identity line
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, label="Perfect")

        # Simple calibration-by-bins curve (bin by true, plot mean pred)
        bins = np.logspace(np.log10(lo), np.log10(hi), num=14)
        idx = np.digitize(y_true, bins) - 1
        xs, ys = [], []
        for b in range(len(bins) - 1):
            m = idx == b
            if np.sum(m) < 50:
                continue
            xs.append(float(np.mean(y_true[m])))
            ys.append(float(np.mean(y_pred[m])))
        if len(xs) >= 2:
            ax.plot(xs, ys, linewidth=1.5, label="Binned mean")

        ax.set_title(f"Survey {sid}", fontsize=12)
        ax.set_xlabel("Actual consumption (USD PPP 2017)")
        if ax is axes[0]:
            ax.set_ylabel("Predicted consumption (USD PPP 2017)")

        if metrics and int(sid) in metrics:
            m = metrics[int(sid)]
            txt = (
                f"poverty_wmape={m['poverty_wmape']:.3f}\n"
                f"hh_mape={m['household_mape']:.3f}\n"
                f"blended={m['blended']:.2f}"
            )
            ax.text(
                0.03, 0.97, txt,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", alpha=0.10, linewidth=0.0),
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Consumption: Predicted vs Actual (log scale) — {model_name}", y=1.02, fontsize=13)
    fig.tight_layout()

    fig.savefig(out_dir / "cons_pred_vs_true_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_ratio_hist_grid(oof: pd.DataFrame, out_dir: Path, model_name: str) -> None:
    _ensure_dir(out_dir)

    surveys = sorted(oof["survey_id"].unique().tolist())
    fig, axes = plt.subplots(1, len(surveys), figsize=(15, 4.2), sharey=True)

    if len(surveys) == 1:
        axes = [axes]

    for ax, sid in zip(axes, surveys):
        g = oof[oof["survey_id"] == sid]
        y_true = g["cons_ppp17"].to_numpy(dtype=float)
        y_pred = g["y_pred"].to_numpy(dtype=float)

        m = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
        ratio = y_pred[m] / y_true[m]

        # Clip extreme ratios for readability, but annotate what we did.
        q_lo, q_hi = np.quantile(ratio, [0.005, 0.995])
        ratio_clip = np.clip(ratio, q_lo, q_hi)
        clipped_frac = float(np.mean((ratio < q_lo) | (ratio > q_hi)))

        # Log-spaced bins for ratio on log axis, centered around 1.
        lo = max(float(np.min(ratio_clip)), 1e-3)
        hi = float(np.max(ratio_clip))
        bins = np.logspace(np.log10(lo), np.log10(hi), 50)

        ax.hist(ratio_clip, bins=bins)
        ax.set_xscale("log")
        ax.axvline(1.0, linestyle="--", linewidth=1.2)
        ax.set_title(f"Survey {sid}", fontsize=12)
        ax.set_xlabel("Pred / True (log axis)")
        if ax is axes[0]:
            ax.set_ylabel("Count")

        med = float(np.median(ratio))
        p10, p90 = np.quantile(ratio, [0.10, 0.90])
        txt = f"median={med:.2f}\n10–90%=[{p10:.2f}, {p90:.2f}]\nclipped={clipped_frac:.1%}"
        ax.text(
            0.03, 0.97, txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", alpha=0.10, linewidth=0.0),
        )

    fig.suptitle(f"Multiplicative Error Distribution — {model_name}", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "error_ratio_hist_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_poverty_curve_grid(oof: pd.DataFrame, out_dir: Path, model_name: str) -> None:
    _ensure_dir(out_dir)

    thresholds = np.asarray(list(THRESHOLDS), dtype=float)
    pred_rates = _poverty_rates_by_survey(oof, thresholds, col="y_pred")
    true_rates = _poverty_rates_by_survey(oof, thresholds, col="cons_ppp17")

    x_left, x_right, x_center = _metric_emphasis_band(thresholds)

    surveys = sorted(pred_rates.keys())
    fig, axes = plt.subplots(1, len(surveys), figsize=(15, 4.6), sharex=True, sharey=True)

    if len(surveys) == 1:
        axes = [axes]

    for ax, sid in zip(axes, surveys):
        ax.axvspan(x_left, x_right, alpha=0.08)
        ax.axvline(x_center, linestyle=":", linewidth=1.0)

        ax.plot(thresholds, true_rates[sid], marker="o", linewidth=1.6, label="True")
        ax.plot(thresholds, pred_rates[sid], marker="o", linewidth=1.6, linestyle="--", label="Pred")

        ax.set_title(f"Survey {sid}", fontsize=12)
        ax.set_xlabel("Poverty threshold (USD PPP 2017)")
        if ax is axes[0]:
            ax.set_ylabel("Population below threshold (%)")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Poverty Curves (weighted) — {model_name}\nShaded band marks ~35–45% region where metric weights peak",
        y=1.05,
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "poverty_curve_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_poverty_error_pp_grid(oof: pd.DataFrame, out_dir: Path, model_name: str) -> None:
    _ensure_dir(out_dir)

    thresholds = np.asarray(list(THRESHOLDS), dtype=float)
    pred_rates = _poverty_rates_by_survey(oof, thresholds, col="y_pred")
    true_rates = _poverty_rates_by_survey(oof, thresholds, col="cons_ppp17")

    x_left, x_right, x_center = _metric_emphasis_band(thresholds)

    surveys = sorted(pred_rates.keys())
    fig, axes = plt.subplots(1, len(surveys), figsize=(15, 4.2), sharex=True, sharey=True)

    if len(surveys) == 1:
        axes = [axes]

    for ax, sid in zip(axes, surveys):
        ax.axvspan(x_left, x_right, alpha=0.08)
        ax.axvline(x_center, linestyle=":", linewidth=1.0)
        ax.axhline(0.0, linestyle="--", linewidth=1.0)

        err_pp = (pred_rates[sid] - true_rates[sid]) * 100.0  # percentage points
        ax.plot(thresholds, err_pp, marker="o", linewidth=1.6)

        ax.set_title(f"Survey {sid}", fontsize=12)
        ax.set_xlabel("Poverty threshold (USD PPP 2017)")
        if ax is axes[0]:
            ax.set_ylabel("Pred − True (percentage points)")

    fig.suptitle(
        f"Poverty Error by Threshold — {model_name}\n(percentage points; shaded band marks ~35–45% region)",
        y=1.05,
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "poverty_error_pp_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def report_metrics(paths: DataPaths, oof: pd.DataFrame) -> pd.DataFrame:
    rates_gt = _load_rates_gt(paths)
    blended_mean, scores = score_all_surveys(
        df=oof,
        rates_gt=rates_gt,
        y_true_col="cons_ppp17",
        y_pred_col="y_pred",
        weight_col="weight",
    )

    rows = []
    for s in scores:
        rows.append(
            {
                "survey_id": s.survey_id,
                "poverty_wmape": s.poverty_wmape,
                "household_mape": s.household_mape,
                "blended": s.blended,
            }
        )
    rows.append(
        {
            "survey_id": "MEAN",
            "poverty_wmape": float(np.mean([s.poverty_wmape for s in scores])),
            "household_mape": float(np.mean([s.household_mape for s in scores])),
            "blended": blended_mean,
        }
    )

    return pd.DataFrame(rows)


def cmd_oof(args: argparse.Namespace) -> None:
    paths = DataPaths(data_dir=Path(args.data_dir))
    train_df = build_train_frame(paths)

    if args.model == "elasticnet":
        cfg = ElasticNetConfig(alpha=args.alpha, l1_ratio=args.l1_ratio, max_iter=5000, random_state=42)
        res = make_oof_predictions_elasticnet(train_df, weight_mode=args.weight_mode, cfg=cfg)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    out_path = Path(args.out)
    _ensure_dir(out_path.parent)
    res.df.to_csv(out_path, index=False)
    print(f"Wrote OOF predictions: {out_path}")
    print(f"Model: {res.model_name}")


def cmd_plots(args: argparse.Namespace) -> None:
    paths = DataPaths(data_dir=Path(args.data_dir))
    oof = pd.read_csv(args.oof)

    model_name = args.model_name if args.model_name else Path(args.oof).stem
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    rep = report_metrics(paths, oof)
    rep.to_csv(out_dir / "metrics_report.csv", index=False)
    print(rep.to_string(index=False))

    m_by_sid = _metrics_by_survey(rep)

    # Refactored, lower-cognitive-load plot set (4 files total)
    plot_consumption_pred_vs_true_grid(oof, out_dir, model_name=model_name, metrics=m_by_sid)
    plot_error_ratio_hist_grid(oof, out_dir, model_name=model_name)
    plot_poverty_curve_grid(oof, out_dir, model_name=model_name)
    plot_poverty_error_pp_grid(oof, out_dir, model_name=model_name)

    print(f"Wrote plots to: {out_dir}")


def cmd_compare(args: argparse.Namespace) -> None:
    paths = DataPaths(data_dir=Path(args.data_dir))
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    frames = []
    for f in args.oof:
        o = pd.read_csv(f)
        name = Path(f).stem
        rep = report_metrics(paths, o)
        rep["model"] = name
        frames.append(rep)

    rep_all = pd.concat(frames, axis=0, ignore_index=True)
    rep_all.to_csv(out_dir / "compare_metrics.csv", index=False)
    print(rep_all.to_string(index=False))

    thresholds = np.asarray(list(THRESHOLDS), dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for f in args.oof:
        o = pd.read_csv(f)
        name = Path(f).stem
        pr = _poverty_rates_by_survey(o, thresholds, col="y_pred")
        mean_curve = np.mean(np.vstack([pr[sid] for sid in sorted(pr.keys())]), axis=0)
        ax.plot(thresholds, mean_curve, linestyle="-", marker="o", label=name)

    x_left, x_right, x_center = _metric_emphasis_band(thresholds)
    ax.axvspan(x_left, x_right, alpha=0.08)
    ax.axvline(x_center, linestyle=":", linewidth=1.0)

    ax.set_title("Mean predicted poverty curve (train surveys)")
    ax.set_xlabel("Poverty threshold (USD PPP 2017)")
    ax.set_ylabel("Population below threshold (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "compare_mean_poverty_curve.png", dpi=150)
    plt.close(fig)

    print(f"Wrote comparison outputs to: {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(prog="poverty.viz")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_oof = sub.add_parser("oof", help="Generate out-of-fold (LOSO) predictions for a model.")
    p_oof.add_argument("--data-dir", type=str, required=True)
    p_oof.add_argument("--model", type=str, default="elasticnet", choices=["elasticnet"])
    p_oof.add_argument("--weight-mode", type=str, default="none", choices=["none", "weight", "sqrt_weight"])
    p_oof.add_argument("--alpha", type=float, default=0.01)
    p_oof.add_argument("--l1-ratio", type=float, default=0.2)
    p_oof.add_argument("--out", type=str, required=True)
    p_oof.set_defaults(func=cmd_oof)

    p_plots = sub.add_parser("plots", help="Create plots + metrics report from an OOF file.")
    p_plots.add_argument("--data-dir", type=str, required=True)
    p_plots.add_argument("--oof", type=str, required=True)
    p_plots.add_argument("--out-dir", type=str, required=True)
    p_plots.add_argument("--model-name", type=str, default="")
    p_plots.set_defaults(func=cmd_plots)

    p_cmp = sub.add_parser("compare", help="Compare multiple OOF files (metrics + overlay plots).")
    p_cmp.add_argument("--data-dir", type=str, required=True)
    p_cmp.add_argument("--oof", type=str, nargs="+", required=True)
    p_cmp.add_argument("--out-dir", type=str, required=True)
    p_cmp.set_defaults(func=cmd_compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
