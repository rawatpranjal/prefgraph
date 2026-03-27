"""Three-way model comparison with rigorous ML evaluation.

Reports:
  - In-sample (train) metrics
  - Out-of-sample (5-fold stratified CV) metrics with confidence intervals
  - Out-of-time: guaranteed by temporal feature/target split in each dataset module

The temporal split happens at the dataset level: first 70% of each user's
observations → features, last 30% → targets. This ensures no future leakage.
K-fold CV then splits users (not time) for model evaluation.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict
from typing import Literal

import numpy as np
import pandas as pd

from case_studies.benchmarks.config import (
    LGBM_CLASSIFIER_PARAMS,
    LGBM_REGRESSOR_PARAMS,
    N_FOLDS,
    SEED,
)


@dataclass
class BenchmarkResult:
    """Result from a single dataset x target benchmark."""

    dataset: str
    target: str
    task_type: str  # "classification" or "regression"
    n_users: int
    n_rp_features: int
    n_base_features: int
    positive_rate: float = 0.0  # For classification: fraction of positive labels

    # --- Out-of-sample (CV) classification metrics ---
    auc_rp: float = 0.0
    auc_base: float = 0.0
    auc_combined: float = 0.0
    auc_rp_std: float = 0.0
    auc_base_std: float = 0.0
    auc_combined_std: float = 0.0

    acc_rp: float = 0.0
    acc_base: float = 0.0
    acc_combined: float = 0.0

    f1_rp: float = 0.0
    f1_base: float = 0.0
    f1_combined: float = 0.0

    # --- In-sample classification metrics ---
    auc_rp_train: float = 0.0
    auc_base_train: float = 0.0
    auc_combined_train: float = 0.0

    # --- Out-of-sample (CV) regression metrics ---
    rmse_rp: float = 0.0
    rmse_base: float = 0.0
    rmse_combined: float = 0.0
    r2_rp: float = 0.0
    r2_base: float = 0.0
    r2_combined: float = 0.0

    # --- In-sample regression metrics ---
    r2_rp_train: float = 0.0
    r2_base_train: float = 0.0
    r2_combined_train: float = 0.0

    # Feature importance from combined model (top features)
    top_features: list | None = None

    # Lift
    auc_lift: float = 0.0  # auc_combined - auc_base

    def to_dict(self) -> dict:
        return asdict(self)


def run_three_way(
    X_rp: pd.DataFrame,
    X_base: pd.DataFrame,
    y: np.ndarray,
    dataset: str,
    target: str,
    task_type: Literal["classification", "regression"] = "classification",
) -> BenchmarkResult:
    """Run three-way model comparison with cross-validation.

    Trains three LightGBM models:
      (a) RP features only
      (b) Baseline features only
      (c) RP + Baseline combined

    Evaluation:
      - In-sample: Train on all data, predict on all data (measures ceiling / overfitting)
      - Out-of-sample: Stratified K-fold CV (measures generalization)
      - Out-of-time: Guaranteed by dataset-level temporal split (features from
        period 1, targets from period 2)
    """
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.metrics import (
        roc_auc_score,
        accuracy_score,
        f1_score,
        mean_squared_error,
        r2_score,
    )

    # Align indices
    common_idx = X_rp.index.intersection(X_base.index)
    X_rp = X_rp.loc[common_idx]
    X_base = X_base.loc[common_idx]

    # Handle NaN: fill with median
    X_rp = X_rp.fillna(X_rp.median())
    X_base = X_base.fillna(X_base.median())

    # Handle infinite values
    X_rp = X_rp.replace([np.inf, -np.inf], 0)
    X_base = X_base.replace([np.inf, -np.inf], 0)

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

    if task_type == "classification":
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(kf.split(np.zeros(n_users), y))
        params = LGBM_CLASSIFIER_PARAMS.copy()
    else:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(kf.split(np.zeros(n_users)))
        params = LGBM_REGRESSOR_PARAMS.copy()

    # --- Out-of-sample: K-fold CV ---
    fold_metrics = {
        name: {"auc": [], "acc": [], "f1": [], "rmse": [], "r2": []}
        for name in feature_sets
    }

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        y_train, y_test = y[train_idx], y[test_idx]

        for name, X_arr in feature_sets.items():
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if task_type == "classification":
                    model = lgb.LGBMClassifier(**params)
                    model.fit(X_train, y_train)
                    y_prob = model.predict_proba(X_test)[:, 1]
                    y_pred = model.predict(X_test)

                    try:
                        auc = roc_auc_score(y_test, y_prob)
                    except ValueError:
                        auc = 0.5
                    fold_metrics[name]["auc"].append(auc)
                    fold_metrics[name]["acc"].append(accuracy_score(y_test, y_pred))
                    fold_metrics[name]["f1"].append(f1_score(y_test, y_pred, zero_division=0))
                else:
                    model = lgb.LGBMRegressor(**params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    fold_metrics[name]["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
                    fold_metrics[name]["r2"].append(r2_score(y_test, y_pred))

    # --- In-sample: Train and evaluate on full data ---
    in_sample = {}
    for name, X_arr in feature_sets.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if task_type == "classification":
                model = lgb.LGBMClassifier(**params)
                model.fit(X_arr, y)
                y_prob_train = model.predict_proba(X_arr)[:, 1]
                try:
                    in_sample[name] = roc_auc_score(y, y_prob_train)
                except ValueError:
                    in_sample[name] = 0.5
            else:
                model = lgb.LGBMRegressor(**params)
                model.fit(X_arr, y)
                y_pred_train = model.predict(X_arr)
                in_sample[name] = r2_score(y, y_pred_train)

    # --- Feature importance from combined model ---
    top_features = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task_type == "classification":
            final_model = lgb.LGBMClassifier(**params)
        else:
            final_model = lgb.LGBMRegressor(**params)
        final_model.fit(X_combined.values, y)
        importances = final_model.feature_importances_
        feat_imp = sorted(
            zip(feature_names["combined"], importances.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        top_features = feat_imp[:15]

    # --- Assemble result ---
    result = BenchmarkResult(
        dataset=dataset,
        target=target,
        task_type=task_type,
        n_users=n_users,
        n_rp_features=X_rp.shape[1],
        n_base_features=X_base.shape[1],
        positive_rate=pos_rate,
        top_features=top_features,
    )

    if task_type == "classification":
        result.auc_rp = float(np.mean(fold_metrics["rp"]["auc"]))
        result.auc_base = float(np.mean(fold_metrics["base"]["auc"]))
        result.auc_combined = float(np.mean(fold_metrics["combined"]["auc"]))
        result.auc_rp_std = float(np.std(fold_metrics["rp"]["auc"]))
        result.auc_base_std = float(np.std(fold_metrics["base"]["auc"]))
        result.auc_combined_std = float(np.std(fold_metrics["combined"]["auc"]))
        result.acc_rp = float(np.mean(fold_metrics["rp"]["acc"]))
        result.acc_base = float(np.mean(fold_metrics["base"]["acc"]))
        result.acc_combined = float(np.mean(fold_metrics["combined"]["acc"]))
        result.f1_rp = float(np.mean(fold_metrics["rp"]["f1"]))
        result.f1_base = float(np.mean(fold_metrics["base"]["f1"]))
        result.f1_combined = float(np.mean(fold_metrics["combined"]["f1"]))
        result.auc_lift = result.auc_combined - result.auc_base

        # In-sample
        result.auc_rp_train = in_sample["rp"]
        result.auc_base_train = in_sample["base"]
        result.auc_combined_train = in_sample["combined"]
    else:
        result.rmse_rp = float(np.mean(fold_metrics["rp"]["rmse"]))
        result.rmse_base = float(np.mean(fold_metrics["base"]["rmse"]))
        result.rmse_combined = float(np.mean(fold_metrics["combined"]["rmse"]))
        result.r2_rp = float(np.mean(fold_metrics["rp"]["r2"]))
        result.r2_base = float(np.mean(fold_metrics["base"]["r2"]))
        result.r2_combined = float(np.mean(fold_metrics["combined"]["r2"]))

        # In-sample
        result.r2_rp_train = in_sample["rp"]
        result.r2_base_train = in_sample["base"]
        result.r2_combined_train = in_sample["combined"]

    return result
