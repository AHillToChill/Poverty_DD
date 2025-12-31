from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

WeightMode = Literal["none", "weight", "sqrt_weight"]


def sample_weight(df: pd.DataFrame, *, weight_col: str = "weight", mode: WeightMode = "none") -> Optional[np.ndarray]:
    """Return sample weights for model fitting.

    Metric evaluation always uses `weight`. This controls only *training*.

    - mode="none": unweighted fit
    - mode="weight": weights proportional to population-expanded weights
    - mode="sqrt_weight": softened weights (often good bias/variance compromise)
    """
    if mode == "none":
        return None

    w = df[weight_col].to_numpy(dtype=float)
    if mode == "weight":
        return w
    if mode == "sqrt_weight":
        return np.sqrt(w)

    raise ValueError(f"Unknown WeightMode: {mode!r}")


# =============================================================================
# ElasticNet on log(consumption) with one-hot encoding
# =============================================================================

@dataclass(frozen=True)
class ElasticNetConfig:
    alpha: float = 0.01
    l1_ratio: float = 0.2
    max_iter: int = 5000
    random_state: int = 42

    def tag(self, weight_mode: WeightMode) -> str:
        a = f"{self.alpha:g}".replace(".", "p")
        l1 = f"{self.l1_ratio:g}".replace(".", "p")
        return f"elasticnet_{weight_mode}_a{a}_l1{l1}"


def build_elasticnet_pipeline(X: pd.DataFrame, cfg: ElasticNetConfig) -> Pipeline:
    """Preprocess (impute + one-hot) then ElasticNet on log(consumption)."""
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = ElasticNet(
        alpha=cfg.alpha,
        l1_ratio=cfg.l1_ratio,
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
    )

    return Pipeline(steps=[("pre", pre), ("model", model)])


def fit_predict_elasticnet_log(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    weight_mode: WeightMode = "none",
    cfg: ElasticNetConfig = ElasticNetConfig(),
) -> np.ndarray:
    """Fit ElasticNet on log(cons_ppp17) and predict cons_ppp17 on valid_df."""
    X_tr = train_df[feature_cols]
    y_tr = np.log(np.clip(train_df["cons_ppp17"].to_numpy(dtype=float), 1e-6, None))
    X_va = valid_df[feature_cols]

    sw = sample_weight(train_df, mode=weight_mode)

    pipe = build_elasticnet_pipeline(X_tr, cfg)
    pipe.fit(X_tr, y_tr, model__sample_weight=sw)

    z_hat = pipe.predict(X_va)
    return np.exp(z_hat)


# =============================================================================
# Boosted trees (HistGradientBoostingRegressor) on log(consumption)
# =============================================================================

@dataclass(frozen=True)
class HGBConfig:
    """Hyperparameters for HistGradientBoostingRegressor."""
    max_iter: int = 600
    learning_rate: float = 0.05
    max_depth: Optional[int] = 6
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    random_state: int = 42
    early_stopping: bool = False

    def tag(self, weight_mode: WeightMode) -> str:
        lr = f"{self.learning_rate:g}".replace(".", "p")
        l2 = f"{self.l2_regularization:g}".replace(".", "p")
        d = "None" if self.max_depth is None else str(self.max_depth)
        return f"hgb_{weight_mode}_i{self.max_iter}_lr{lr}_d{d}_leaf{self.min_samples_leaf}_l2{l2}"


def build_hgb_preprocessor(df: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    """Tree-friendly preprocessing: numeric impute + ordinal encode categoricals."""
    X = df[feature_cols]
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def fit_predict_hgb_log(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    weight_mode: WeightMode = "none",
    cfg: HGBConfig = HGBConfig(),
) -> np.ndarray:
    """Fit HGB on log(cons_ppp17) and predict cons_ppp17 on valid_df."""
    pre = build_hgb_preprocessor(train_df, feature_cols)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=cfg.max_iter,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=cfg.l2_regularization,
        early_stopping=cfg.early_stopping,
        random_state=cfg.random_state,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])

    y_tr = np.log(np.clip(train_df["cons_ppp17"].to_numpy(dtype=float), 1e-6, None))
    sw = sample_weight(train_df, mode=weight_mode)

    pipe.fit(train_df[feature_cols], y_tr, model__sample_weight=sw)
    z_hat = pipe.predict(valid_df[feature_cols])
    return np.exp(z_hat)
