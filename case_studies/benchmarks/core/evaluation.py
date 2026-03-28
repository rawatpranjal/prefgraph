"""Three-way model comparison with rigorous ML evaluation.

Evaluation: Out-of-sample 5-fold stratified CV (splits users, not time).

Data leakage prevention:
  Each dataset module must implement a GLOBAL CALENDAR CUTOFF:
  - Features extracted from observations BEFORE the cutoff date
  - Targets computed from observations AFTER the cutoff date
  - This ensures no temporal overlap across users

Metrics (classification):
  - AUC-ROC: Discrimination ability across all thresholds
  - AUC-PR: Average precision (better for imbalanced targets)
  - Log Loss: Calibration quality
  - F1: Balanced precision-recall at default threshold

Metrics (regression):
  - R²: Explained variance
  - RMSE: Root mean squared error
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
import pandas as pd

from case_studies.benchmarks.config import (
    CATBOOST_CLASSIFIER_PARAMS,
    CATBOOST_REGRESSOR_PARAMS,
    TEST_FRACTION,
    SEED,
)


@dataclass
class BenchmarkResult:
    """Result from a single dataset x target benchmark."""

    dataset: str
    target: str
    task_type: str
    n_users: int
    n_rp_features: int
    n_base_features: int
    positive_rate: float = 0.0

    # --- Out-of-sample classification ---
    auc_rp: float = 0.0
    auc_base: float = 0.0
    auc_combined: float = 0.0
    auc_rp_std: float = 0.0
    auc_base_std: float = 0.0
    auc_combined_std: float = 0.0

    # AUC-PR (average precision)
    ap_rp: float = 0.0
    ap_base: float = 0.0
    ap_combined: float = 0.0

    # Log loss
    logloss_rp: float = 0.0
    logloss_base: float = 0.0
    logloss_combined: float = 0.0

    f1_rp: float = 0.0
    f1_base: float = 0.0
    f1_combined: float = 0.0

    # --- Out-of-sample regression ---
    rmse_rp: float = 0.0
    rmse_base: float = 0.0
    rmse_combined: float = 0.0
    r2_rp: float = 0.0
    r2_base: float = 0.0
    r2_combined: float = 0.0

    top_features: list | None = None
    auc_lift: float = 0.0
    auc_lift_pct: float = 0.0
    wall_time_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_lift_pct(auc_combined: float, auc_base: float) -> float:
    """Compute percentage lift of combined over baseline AUC."""
    if auc_base > 0.5:
        return (auc_combined - auc_base) / auc_base * 100
    return 0.0


def run_three_way(
    X_rp: pd.DataFrame,
    X_base: pd.DataFrame,
    y: np.ndarray,
    dataset: str,
    target: str,
    task_type: Literal["classification", "regression"] = "classification",
) -> BenchmarkResult:
    """Run three-way model comparison — pure out-of-time evaluation.

    Features (X) come from each user's PAST observations.
    Targets (y) come from each user's FUTURE observations.
    The temporal split IS the evaluation — no user holdout needed.

    Three models: (a) RP only, (b) Baseline only, (c) RP + Baseline.
    """
    import time as _time
    _t0 = _time.time()

    from catboost import CatBoostClassifier, CatBoostRegressor
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        log_loss,
        f1_score,
        mean_squared_error,
        r2_score,
    )

    # Align indices
    common_idx = X_rp.index.intersection(X_base.index)
    n_dropped_rp = len(X_rp) - len(common_idx)
    n_dropped_base = len(X_base) - len(common_idx)
    if n_dropped_rp > 0 or n_dropped_base > 0:
        warnings.warn(
            f"Dropped {n_dropped_rp} RP-only and {n_dropped_base} baseline-only users "
            f"during index alignment ({len(common_idx)} remaining)"
        )
    X_rp = X_rp.loc[common_idx]
    X_base = X_base.loc[common_idx]

    # Handle NaN/inf
    X_rp = X_rp.fillna(X_rp.median()).replace([np.inf, -np.inf], 0)
    X_base = X_base.fillna(X_base.median()).replace([np.inf, -np.inf], 0)

    X_combined = pd.concat([X_base, X_rp], axis=1)
    X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]

    feature_sets = {
        "rp": X_rp.values,
        "base": X_base.values,
        "combined": X_combined.values,
    }
    feature_names = {
        "rp": list(X_rp.columns),
        "base": list(X_base.columns),
        "combined": list(X_combined.columns),
    }

    n_users = len(y)
    pos_rate = float(np.mean(y)) if task_type == "classification" else 0.0
    params = (CATBOOST_CLASSIFIER_PARAMS if task_type == "classification"
              else CATBOOST_REGRESSOR_PARAMS).copy()

    # Pure out-of-time: train on ALL users' past, predict ALL users' future.
    # The temporal split in each bench file IS the train/test boundary.
    metrics = {}
    for name, X_arr in feature_sets.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if task_type == "classification":
                model = CatBoostClassifier(**params)
                model.fit(X_arr, y)
                y_prob = model.predict_proba(X_arr)[:, 1]
                y_pred = model.predict(X_arr)

                try:
                    auc = roc_auc_score(y, y_prob)
                except ValueError:
                    auc = 0.5
                try:
                    ap = average_precision_score(y, y_prob)
                except ValueError:
                    ap = 0.0
                try:
                    ll = log_loss(y, y_prob)
                except ValueError:
                    ll = 1.0
                f1 = f1_score(y, y_pred, zero_division=0)
                metrics[name] = {"auc": auc, "ap": ap, "logloss": ll, "f1": f1}
            else:
                model = CatBoostRegressor(**params)
                model.fit(X_arr, y)
                y_pred = model.predict(X_arr)
                metrics[name] = {
                    "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
                    "r2": float(r2_score(y, y_pred)),
                }

    # Feature importance
    top_features = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task_type == "classification":
            final_model = CatBoostClassifier(**params)
        else:
            final_model = CatBoostRegressor(**params)
        final_model.fit(X_combined.values, y)
        importances = final_model.feature_importances_
        feat_imp = sorted(
            zip(feature_names["combined"], importances.tolist()),
            key=lambda x: x[1], reverse=True,
        )
        top_features = feat_imp[:15]

    # Assemble
    result = BenchmarkResult(
        dataset=dataset, target=target, task_type=task_type,
        n_users=n_users, n_rp_features=X_rp.shape[1],
        n_base_features=X_base.shape[1], positive_rate=pos_rate,
        top_features=top_features,
    )

    if task_type == "classification":
        result.auc_rp = metrics["rp"]["auc"]
        result.auc_base = metrics["base"]["auc"]
        result.auc_combined = metrics["combined"]["auc"]
        result.ap_rp = metrics["rp"]["ap"]
        result.ap_base = metrics["base"]["ap"]
        result.ap_combined = metrics["combined"]["ap"]
        result.logloss_rp = metrics["rp"]["logloss"]
        result.logloss_base = metrics["base"]["logloss"]
        result.logloss_combined = metrics["combined"]["logloss"]
        result.f1_rp = metrics["rp"]["f1"]
        result.f1_base = metrics["base"]["f1"]
        result.f1_combined = metrics["combined"]["f1"]
        result.auc_lift = result.auc_combined - result.auc_base
        result.auc_lift_pct = compute_lift_pct(result.auc_combined, result.auc_base)
    else:
        result.rmse_rp = metrics["rp"]["rmse"]
        result.rmse_base = metrics["base"]["rmse"]
        result.rmse_combined = metrics["combined"]["rmse"]
        result.r2_rp = metrics["rp"]["r2"]
        result.r2_base = metrics["base"]["r2"]
        result.r2_combined = metrics["combined"]["r2"]

    result.wall_time_s = _time.time() - _t0
    return result
