from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, TypeAlias

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

WeightMode: TypeAlias = Literal["none", "weight", "sqrt_weight"]


def _sample_weight(df: pd.DataFrame, weight_col: str, mode: WeightMode) -> Optional[np.ndarray]:
    if mode == "none":
        return None
    w = df[weight_col].to_numpy(dtype=float)
    if mode == "weight":
        return w
    if mode == "sqrt_weight":
        return np.sqrt(w)
    raise ValueError(f"Unknown weight mode: {mode}")


# =============================================================================
# ElasticNet on log(consumption)
# =============================================================================

@dataclass(frozen=True)
class ElasticNetConfig:
    alpha: float = 0.01
    l1_ratio: float = 0.2
    max_iter: int = 5000
    random_state: int = 42

    def tag(self) -> str:
        return f"elasticnet_a{self.alpha:g}_l1{self.l1_ratio:g}"


def build_elasticnet_pipeline(
    X: pd.DataFrame,
    cfg: ElasticNetConfig = ElasticNetConfig(),
) -> Pipeline:
    """
    Returns a sklearn Pipeline:
      - numeric: median impute
      - categorical: most_frequent impute + one-hot
      - model: ElasticNet on log(consumption)
    """
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
    weight_mode: WeightMode = "none",
    cfg: ElasticNetConfig = ElasticNetConfig(),
    *,
    y_col: str = "cons_ppp17",
    weight_col: str = "weight",
) -> np.ndarray:
    """Fit ElasticNet on log(y) and predict y (back-transformed) for valid_df."""
    X_tr = train_df[feature_cols]
    y_tr = np.log(np.clip(train_df[y_col].to_numpy(dtype=float), 1e-6, None))

    pipe = build_elasticnet_pipeline(X_tr, cfg)
    sw = _sample_weight(train_df, weight_col=weight_col, mode=weight_mode)
    pipe.fit(X_tr, y_tr, model__sample_weight=sw)

    z_hat = pipe.predict(valid_df[feature_cols])
    return np.exp(z_hat)


# =============================================================================
# Boosted trees (HistGradientBoostingRegressor) on log(consumption)
# =============================================================================

@dataclass(frozen=True)
class HGBConfig:
    """Hyperparameters for HistGradientBoostingRegressor."""
    max_iter: int = 400
    learning_rate: float = 0.05
    max_depth: Optional[int] = 6
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    early_stopping: bool = False

    def tag(self) -> str:
        md = "None" if self.max_depth is None else str(self.max_depth)
        es = "es1" if self.early_stopping else "es0"
        return (
            f"hgb_i{self.max_iter}_lr{self.learning_rate:g}_d{md}"
            f"_leaf{self.min_samples_leaf}_l2{self.l2_regularization:g}_{es}"
        )


def build_hgb_preprocessor(df: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    """
    Preprocessor for tree models:
      - numeric: median impute
      - categorical: constant impute + ordinal encode (unknown -> -1)

    Ordinal encoding imposes an arbitrary order, but in practice boosted trees often
    handle it acceptably for large tabular problems while avoiding huge one-hot matrices.
    """
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

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
        sparse_threshold=0.0,  # keep dense for HGB
    )


def fit_predict_hgb_log(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: HGBConfig = HGBConfig(),
    weight_mode: WeightMode = "none",
    *,
    y_col: str = "cons_ppp17",
    weight_col: str = "weight",
    random_state: int = 42,
) -> np.ndarray:
    """Fit boosted trees on log(y) and predict y (back-transformed) for valid_df."""
    pre = build_hgb_preprocessor(train_df, feature_cols)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=cfg.max_iter,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=cfg.l2_regularization,
        early_stopping=cfg.early_stopping,
        random_state=random_state,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])

    y = train_df[y_col].to_numpy(dtype=float)
    y_log = np.log(np.clip(y, 1e-6, None))

    sw = _sample_weight(train_df, weight_col=weight_col, mode=weight_mode)
    pipe.fit(train_df[feature_cols], y_log, model__sample_weight=sw)

    z_hat = pipe.predict(valid_df[feature_cols])
    return np.exp(z_hat)
