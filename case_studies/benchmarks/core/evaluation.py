"""Three-way model comparison: Global time cutoff + user holdout.

Design:
  1. Global T_cutoff (70th percentile of all timestamps per dataset)
  2. Features from events BEFORE T_cutoff
  3. Targets from events AFTER T_cutoff
  4. 80/20 random user holdout — train on 80%, evaluate on 20%
  5. LightGBM defaults + minimal regularization
  6. Permutation importance on test set
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
import pandas as pd

from case_studies.benchmarks.config import SEED


# LightGBM defaults + minimal universal regularization
LGBM_PARAMS = {
    "random_state": SEED,
    "verbose": -1,
    "colsample_bytree": 0.8,
    "min_child_samples": 50,
}


@dataclass
class BenchmarkResult:
    dataset: str
    target: str
    task_type: str
    n_users: int
    n_train: int
    n_test: int
    n_rp_features: int
    n_base_features: int
    positive_rate: float = 0.0

    # Test set metrics (20% holdout users, future targets)
    auc_rp: float = 0.0
    auc_base: float = 0.0
    auc_combined: float = 0.0
    ap_rp: float = 0.0
    ap_base: float = 0.0
    ap_combined: float = 0.0
    r2_rp: float = 0.0
    r2_base: float = 0.0
    r2_combined: float = 0.0

    top_features: list | None = None  # Permutation importance on test set
    wall_time_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_lift_pct(combined: float, base: float) -> float:
    if base > 0.5:
        return (combined - base) / base * 100
    return 0.0


def run_three_way(
    X_rp: pd.DataFrame,
    X_base: pd.DataFrame,
    y: np.ndarray,
    dataset: str,
    target: str,
    task_type: Literal["classification", "regression"] = "classification",
) -> BenchmarkResult:
    """Global time cutoff already applied by caller. This does user holdout + eval."""
    import time as _time
    _t0 = _time.time()

    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, r2_score,
    )
    from sklearn.inspection import permutation_importance

    # Align indices
    common_idx = X_rp.index.intersection(X_base.index)
    X_rp = X_rp.loc[common_idx]
    X_base = X_base.loc[common_idx]

    X_combined = pd.concat([X_base, X_rp], axis=1)
    X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]

    feature_sets = {
        "rp": X_rp,
        "base": X_base,
        "combined": X_combined,
    }

    n_users = len(y)
    pos_rate = float(np.mean(y)) if task_type == "classification" else 0.0

    # 80/20 user holdout
    stratify = y if task_type == "classification" else None
    idx = np.arange(n_users)
    tr_idx, te_idx = train_test_split(
        idx, test_size=0.2, random_state=SEED, stratify=stratify
    )
    y_train, y_test = y[tr_idx], y[te_idx]

    metrics = {}
    models = {}
    for name, X_df in feature_sets.items():
        X_arr = X_df.values
        # Impute NaN/inf using TRAIN medians only
        train_df = pd.DataFrame(X_arr[tr_idx])
        medians = train_df.median()
        X_tr = train_df.fillna(medians).replace([np.inf, -np.inf], 0).values
        X_te = pd.DataFrame(X_arr[te_idx]).fillna(medians).replace([np.inf, -np.inf], 0).values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if task_type == "classification":
                model = lgb.LGBMClassifier(**LGBM_PARAMS)
                model.fit(X_tr, y_train)
                y_prob = model.predict_proba(X_te)[:, 1]
                try:
                    auc = roc_auc_score(y_test, y_prob)
                except ValueError:
                    auc = 0.5
                try:
                    ap = average_precision_score(y_test, y_prob)
                except ValueError:
                    ap = 0.0
                metrics[name] = {"auc": auc, "ap": ap}
            else:
                model = lgb.LGBMRegressor(**LGBM_PARAMS)
                model.fit(X_tr, y_train)
                y_pred = model.predict(X_te)
                metrics[name] = {"r2": float(r2_score(y_test, y_pred))}
            models[name] = (model, X_te, y_test)

    # Permutation importance on TEST set (combined model)
    top_features = None
    model_c, X_te_c, y_te_c = models["combined"]
    scoring = "roc_auc" if task_type == "classification" else "r2"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perm = permutation_importance(
            model_c, X_te_c, y_te_c,
            scoring=scoring, n_repeats=5, random_state=SEED,
        )
    feat_names = list(X_combined.columns)
    imp_mean = perm.importances_mean
    top_features = sorted(
        zip(feat_names, imp_mean.tolist()),
        key=lambda x: x[1], reverse=True,
    )[:15]

    result = BenchmarkResult(
        dataset=dataset, target=target, task_type=task_type,
        n_users=n_users, n_train=len(tr_idx), n_test=len(te_idx),
        n_rp_features=X_rp.shape[1], n_base_features=X_base.shape[1],
        positive_rate=pos_rate, top_features=top_features,
    )

    if task_type == "classification":
        result.auc_rp = metrics["rp"]["auc"]
        result.auc_base = metrics["base"]["auc"]
        result.auc_combined = metrics["combined"]["auc"]
        result.ap_rp = metrics["rp"]["ap"]
        result.ap_base = metrics["base"]["ap"]
        result.ap_combined = metrics["combined"]["ap"]
    else:
        result.r2_rp = metrics["rp"]["r2"]
        result.r2_base = metrics["base"]["r2"]
        result.r2_combined = metrics["combined"]["r2"]

    result.wall_time_s = _time.time() - _t0
    return result
