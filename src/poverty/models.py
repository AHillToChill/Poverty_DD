from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class ElasticNetConfig:
    alpha: float = 0.01
    l1_ratio: float = 0.2
    max_iter: int = 5000
    random_state: int = 42


def build_elasticnet_pipeline(
    X: pd.DataFrame,
    cfg: ElasticNetConfig = ElasticNetConfig(),
) -> Pipeline:
    """
    Returns a sklearn Pipeline:
      - numeric: median impute
      - categorical: impute missing token already set upstream, then one-hot encode
      - model: ElasticNet on log(consumption)
    """
    # Infer column types from the training frame
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical = Pipeline(steps=[
        # Strings already normalized; still impute defensively:
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
    weight_mode: str = "none",  # "none" | "weight" | "sqrt_weight"
    cfg: ElasticNetConfig = ElasticNetConfig(),
) -> np.ndarray:
    """
    Fit ElasticNet on log(cons_ppp17) and predict cons_ppp17 on valid_df.
    """
    X_tr = train_df[feature_cols]
    y_tr = np.log(train_df["cons_ppp17"].to_numpy(dtype=float))

    X_va = valid_df[feature_cols]

    if weight_mode == "none":
        w = None
    elif weight_mode == "weight":
        w = train_df["weight"].to_numpy(dtype=float)
    elif weight_mode == "sqrt_weight":
        w = np.sqrt(train_df["weight"].to_numpy(dtype=float))
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    pipe = build_elasticnet_pipeline(X_tr, cfg)
    pipe.fit(X_tr, y_tr, model__sample_weight=w)

    z_hat = pipe.predict(X_va)
    y_hat = np.exp(z_hat)  # back-transform
    return y_hat
