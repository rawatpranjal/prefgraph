#!/usr/bin/env python3
"""Cross-validated LGBM benchmark to verify no overfitting.

5-fold stratified CV on the combined feature set. Reports mean and std
of AUC-ROC and AUC-PR across folds for Base vs Base+RP.

Usage:
    PYREVEALED_DATA_DIR=/Volumes/Expansion/datasets \
      python case_studies/benchmarks/cv_benchmark.py --datasets validated
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from case_studies.benchmarks.config import SEED
from case_studies.benchmarks.runner import (
    AVAILABLE_DATASETS, VALIDATED_DATASETS, DATASET_DISPLAY_NAMES,
)

MODEL_PARAMS = {
    "random_state": SEED,
    "verbose": -1,
    "n_jobs": -1,
    "max_depth": 3,
    "num_leaves": 8,
    "min_child_samples": 30,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 2.0,
    "reg_lambda": 2.0,
    "n_estimators": 200,
}


def run_cv(X_base, X_rp, y, task_type, n_folds=5):
    """Run 5-fold CV on Base and Base+RP. Return per-fold metrics."""
    X_combined = pd.concat([X_base, X_rp], axis=1)
    X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]

    X_base_arr = X_base.values
    X_comb_arr = X_combined.values

    if task_type == "classification":
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    base_auc, base_ap, comb_auc, comb_ap = [], [], [], []
    base_r2, comb_r2 = [], []

    for tr, te in kf.split(X_base_arr, y):
        for label, X_arr, auc_list, ap_list, r2_list in [
            ("base", X_base_arr, base_auc, base_ap, base_r2),
            ("comb", X_comb_arr, comb_auc, comb_ap, comb_r2),
        ]:
            train_df = pd.DataFrame(X_arr[tr])
            med = train_df.median()
            X_tr = train_df.fillna(med).replace([np.inf, -np.inf], 0).values
            X_te = pd.DataFrame(X_arr[te]).fillna(med).replace([np.inf, -np.inf], 0).values

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if task_type == "classification":
                    m = LGBMClassifier(**MODEL_PARAMS)
                    m.fit(X_tr, y[tr])
                    prob = m.predict_proba(X_te)[:, 1]
                    try:
                        auc_list.append(roc_auc_score(y[te], prob))
                    except ValueError:
                        auc_list.append(0.5)
                    try:
                        ap_list.append(average_precision_score(y[te], prob))
                    except ValueError:
                        ap_list.append(0.0)
                else:
                    m = LGBMRegressor(**MODEL_PARAMS)
                    m.fit(X_tr, y[tr])
                    pred = m.predict(X_te)
                    r2_list.append(float(r2_score(y[te], pred)))

    if task_type == "classification":
        return {
            "base_auc_mean": np.mean(base_auc), "base_auc_std": np.std(base_auc),
            "base_ap_mean": np.mean(base_ap), "base_ap_std": np.std(base_ap),
            "comb_auc_mean": np.mean(comb_auc), "comb_auc_std": np.std(comb_auc),
            "comb_ap_mean": np.mean(comb_ap), "comb_ap_std": np.std(comb_ap),
        }
    else:
        return {
            "base_r2_mean": np.mean(base_r2), "base_r2_std": np.std(base_r2),
            "comb_r2_mean": np.mean(comb_r2), "comb_r2_std": np.std(comb_r2),
        }


def main():
    parser = argparse.ArgumentParser(description="5-Fold CV LGBM Benchmark")
    parser.add_argument("--datasets", type=str, default="validated")
    parser.add_argument("--max-users", type=int, default=None)
    args = parser.parse_args()

    if args.datasets == "validated":
        dataset_names = VALIDATED_DATASETS
    elif args.datasets == "all":
        dataset_names = list(AVAILABLE_DATASETS.keys())
    else:
        dataset_names = [d.strip() for d in args.datasets.split(",")]

    print("=" * 120)
    print(" 5-FOLD CV LGBM BENCHMARK (regularized)")
    print("=" * 120)

    try:
        from tqdm import tqdm
        ds_iter = tqdm(dataset_names, desc="CV", unit="ds")
    except ImportError:
        ds_iter = dataset_names

    results = []

    for name in ds_iter:
        mod_path = AVAILABLE_DATASETS.get(name)
        if not mod_path:
            continue
        mod = importlib.import_module(mod_path)

        kwargs = {}
        if name == "dunnhumby":
            if args.max_users: kwargs["n_households"] = args.max_users
        elif name == "open_ecommerce":
            if args.max_users: kwargs["n_users"] = args.max_users
        elif name == "mind":
            effective = args.max_users or 50000
            if args.max_users and args.max_users < 2000:
                effective = max(args.max_users * 20, 5000)
            kwargs["max_users"] = effective
        elif name == "kuairec":
            if args.max_users: kwargs["max_users"] = args.max_users
        elif name == "taobao_buy_window":
            kwargs["max_users"] = args.max_users or 50000
        else:
            kwargs["max_users"] = args.max_users or 50000

        try:
            raw = mod.load_and_prepare(**kwargs)
            if len(raw) == 5:
                X_rp, X_base, _, targets_dict, user_ids = raw
            else:
                X_rp, X_base, targets_dict, user_ids = raw
        except FileNotFoundError as e:
            print(f"  [SKIP] {name}: {e}")
            continue
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            continue

        if X_rp is None:
            continue

        common_idx = X_rp.index.intersection(X_base.index)
        X_rp = X_rp.loc[common_idx]
        X_base = X_base.loc[common_idx]

        display = DATASET_DISPLAY_NAMES.get(name, name)

        for tgt_name, tgt_tuple in targets_dict.items():
            if len(tgt_tuple) == 4:
                y, task_type, y_cont, pctl = tgt_tuple
                if y_cont is not None:
                    y_cont = np.asarray(y_cont)
            else:
                y, task_type = tgt_tuple
                y_cont, pctl = None, None

            y = np.asarray(y)
            if task_type == "classification":
                pr = float(np.mean(y))
                if pr < 0.02 or pr > 0.98:
                    continue

            # For threshold targets, binarize on the full dataset
            # (CV handles leakage within folds via the fold split)
            if y_cont is not None and pctl is not None:
                threshold = np.percentile(y_cont, pctl)
                y = (y_cont > threshold).astype(int)

            t0 = time.time()
            cv = run_cv(X_base, X_rp, y, task_type)
            elapsed = time.time() - t0

            cv["dataset"] = display
            cv["target"] = tgt_name
            cv["task_type"] = task_type
            cv["n_users"] = len(y)
            cv["time_s"] = elapsed
            results.append(cv)

            if task_type == "classification":
                lift_auc = cv["comb_auc_mean"] - cv["base_auc_mean"]
                lift_ap = cv["comb_ap_mean"] - cv["base_ap_mean"]
                print(f"  {display:<18} {tgt_name:<20} N={len(y):>6}  "
                      f"Base AUC={cv['base_auc_mean']:.3f}({cv['base_auc_std']:.3f})  "
                      f"+RP AUC={cv['comb_auc_mean']:.3f}({cv['comb_auc_std']:.3f})  Δ={lift_auc:+.4f}  "
                      f"Base AP={cv['base_ap_mean']:.3f}  +RP AP={cv['comb_ap_mean']:.3f}  Δ={lift_ap:+.4f}  "
                      f"[{elapsed:.0f}s]")
            else:
                lift_r2 = cv["comb_r2_mean"] - cv["base_r2_mean"]
                print(f"  {display:<18} {tgt_name:<20} N={len(y):>6}  "
                      f"Base R2={cv['base_r2_mean']:.3f}({cv['base_r2_std']:.3f})  "
                      f"+RP R2={cv['comb_r2_mean']:.3f}({cv['comb_r2_std']:.3f})  Δ={lift_r2:+.4f}  "
                      f"[{elapsed:.0f}s]")

    # Save
    import json
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "cv_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_dir}/cv_results.json")


if __name__ == "__main__":
    main()
