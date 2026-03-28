"""Generate summary tables from benchmark results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from case_studies.benchmarks.core.evaluation import BenchmarkResult, compute_lift_pct


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print benchmark results: global time cutoff + 80/20 user holdout."""
    cls = [r for r in results if r.task_type == "classification"]
    reg = [r for r in results if r.task_type == "regression"]

    print("\n" + "=" * 95)
    print(" ML BENCHMARK: RP Features as Predictive Signals")
    print(" Global time cutoff | 80/20 user holdout | LightGBM defaults")
    print("=" * 95)
    n_datasets = len(set(r.dataset for r in results))
    n_users = sum(r.n_users for r in results)
    print(f"\n  {len(results)} tasks, {n_datasets} datasets, {n_users:,} users.")

    if cls:
        print("\n  Classification — Test Set (20% holdout users, future targets)")
        print("  " + "-" * 91)
        print(f"  {'Dataset':<16} {'Target':<16} {'Train':>6} {'Test':>5} {'%pos':>5}  "
              f"{'Base':>7} {'+RP':>7} {'Lift%':>7}  {'AP Base':>8} {'AP +RP':>8}")
        print("  " + "-" * 91)
        for r in cls:
            lift = compute_lift_pct(r.auc_combined, r.auc_base)
            print(f"  {r.dataset:<16} {r.target:<16} {r.n_train:>6} {r.n_test:>5} {r.positive_rate:>5.1%}  "
                  f"{r.auc_base:>7.3f} {r.auc_combined:>7.3f} {lift:>+6.1f}%  "
                  f"{r.ap_base:>8.3f} {r.ap_combined:>8.3f}")
        print("  " + "-" * 91)

    if reg:
        print("\n  Regression — Test Set")
        print("  " + "-" * 70)
        print(f"  {'Dataset':<16} {'Target':<18} {'Train':>6} {'Test':>5}  "
              f"{'Base R²':>8} {'+RP R²':>8}")
        print("  " + "-" * 70)
        for r in reg:
            print(f"  {r.dataset:<16} {r.target:<18} {r.n_train:>6} {r.n_test:>5}  "
                  f"{r.r2_base:>8.3f} {r.r2_combined:>8.3f}")
        print("  " + "-" * 70)

    # Timing
    total_time = sum(r.wall_time_s for r in results)
    print(f"\n  Wall time: {total_time:.0f}s")

    # Permutation importance (test set)
    feat_results = [r for r in cls if r.top_features]
    if feat_results:
        print("\n  Permutation Importance — Test Set (Combined model)")
        print("  " + "-" * 55)
        feat_scores: dict[str, list[float]] = {}
        for r in feat_results:
            for fname, score in r.top_features:
                feat_scores.setdefault(fname, []).append(score)
        avg = {k: np.mean(v) for k, v in feat_scores.items()}
        top = sorted(avg.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (fname, score) in enumerate(top, 1):
            marker = " *" if score > 0.001 else ""
            print(f"    {i:2}. {fname:<30} {score:>10.4f}{marker}")
        print("  " + "-" * 55)
        print("  * = permutation importance > 0.001 on holdout set")

    print()


def save_results(results: list[BenchmarkResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"  Results saved to {output_dir}/")


def generate_plots(results: list[BenchmarkResult], output_dir: Path) -> None:
    pass
