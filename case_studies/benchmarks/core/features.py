"""Feature extraction for ML benchmarks.

Two feature sets per user:
  - Baseline: RFM + spending/engagement stats (no RP library needed)
  - RP: Revealed preference metrics from Engine batch API
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pyrevealed.engine import Engine, EngineResult, MenuResult, results_to_dataframe


# ---------------------------------------------------------------------------
# Budget baseline features (from raw prices/quantities)
# ---------------------------------------------------------------------------

def extract_budget_baseline(
    users: list[tuple[np.ndarray, np.ndarray]],
    user_ids: list[str],
) -> pd.DataFrame:
    """Extract standard RFM + spending features from budget data.

    Args:
        users: List of (prices T×K, quantities T×K) tuples.
        user_ids: Matching user ID strings.

    Returns:
        DataFrame indexed by user_id with baseline feature columns.
    """
    records = []
    for (prices, quantities), uid in zip(users, user_ids):
        T, K = quantities.shape
        spend_per_obs = np.sum(prices * quantities, axis=1)  # (T,)
        total_qty_per_obs = np.sum(quantities, axis=1)  # (T,)

        # Spending statistics
        total_spend = float(np.sum(spend_per_obs))
        mean_spend = float(np.mean(spend_per_obs))
        std_spend = float(np.std(spend_per_obs)) if T > 1 else 0.0
        max_spend = float(np.max(spend_per_obs))
        min_spend = float(np.min(spend_per_obs))

        # Frequency / volume
        n_obs = T
        mean_basket_size = float(np.mean(total_qty_per_obs))

        # Category concentration
        total_qty_per_cat = np.sum(quantities, axis=0)  # (K,)
        total_all = np.sum(total_qty_per_cat)
        if total_all > 0:
            shares = total_qty_per_cat / total_all
            herfindahl = float(np.sum(shares ** 2))
            top_share = float(np.max(shares))
            n_active_cats = int(np.sum(shares > 0))
        else:
            herfindahl = 1.0
            top_share = 1.0
            n_active_cats = 0

        # Temporal trend (spend slope via least-squares)
        if T >= 3:
            x = np.arange(T, dtype=float)
            x_centered = x - x.mean()
            denom = np.sum(x_centered ** 2)
            spend_slope = float(np.sum(x_centered * (spend_per_obs - spend_per_obs.mean())) / denom) if denom > 0 else 0.0
        else:
            spend_slope = 0.0

        # Coefficient of variation
        cv = std_spend / mean_spend if mean_spend > 0 else 0.0

        records.append({
            "user_id": uid,
            "n_obs": n_obs,
            "total_spend": total_spend,
            "mean_spend": mean_spend,
            "std_spend": std_spend,
            "max_spend": max_spend,
            "min_spend": min_spend,
            "mean_basket_size": mean_basket_size,
            "herfindahl": herfindahl,
            "top_category_share": top_share,
            "n_active_categories": n_active_cats,
            "spend_slope": spend_slope,
            "spend_cv": cv,
        })

    df = pd.DataFrame(records).set_index("user_id")
    return df


# ---------------------------------------------------------------------------
# Budget RP features (from Engine)
# ---------------------------------------------------------------------------

def extract_budget_rp(
    users: list[tuple[np.ndarray, np.ndarray]],
    user_ids: list[str],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Extract revealed preference features via Engine batch API.

    Args:
        users: List of (prices T×K, quantities T×K) tuples.
        user_ids: Matching user ID strings.
        metrics: Engine metrics to compute (default: garp, ccei, mpi, harp, hm, vei).

    Returns:
        DataFrame indexed by user_id with RP feature columns.
    """
    if metrics is None:
        metrics = ["garp", "ccei", "mpi", "harp", "hm", "vei"]

    engine = Engine(metrics=metrics)
    results = engine.analyze_arrays(users)
    df = results_to_dataframe(results, user_ids=user_ids)

    # Add derived features
    df["hm_ratio"] = df["hm_consistent"] / df["hm_total"].replace(0, 1)
    n_obs = np.array([u[0].shape[0] for u in users])
    df["violation_density"] = df["n_violations"] / np.maximum(n_obs * (n_obs - 1), 1)
    df["scc_ratio"] = df["max_scc"] / np.maximum(n_obs, 1)

    # Drop raw counts that are redundant with ratios
    df = df.drop(columns=["hm_consistent", "hm_total", "compute_time_us"], errors="ignore")

    # Convert bools to int for ML
    for col in ["is_garp", "is_harp", "utility_success"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df


# ---------------------------------------------------------------------------
# Menu baseline features (from raw menu-choice data)
# ---------------------------------------------------------------------------

def extract_menu_baseline(
    user_logs: dict[str, object],  # str -> MenuChoiceLog
) -> pd.DataFrame:
    """Extract standard engagement features from menu-choice data.

    Args:
        user_logs: Dict mapping user_id -> MenuChoiceLog.

    Returns:
        DataFrame indexed by user_id with baseline feature columns.
    """
    records = []
    for uid, log in user_logs.items():
        n_sessions = len(log.choices)
        menu_sizes = [len(m) for m in log.menus]
        n_unique_items = len(log.all_items)

        records.append({
            "user_id": uid,
            "n_sessions": n_sessions,
            "mean_menu_size": float(np.mean(menu_sizes)),
            "std_menu_size": float(np.std(menu_sizes)) if n_sessions > 1 else 0.0,
            "max_menu_size": max(menu_sizes),
            "min_menu_size": min(menu_sizes),
            "n_unique_items": n_unique_items,
            "items_per_session": n_unique_items / max(n_sessions, 1),
        })

    df = pd.DataFrame(records).set_index("user_id")
    return df


# ---------------------------------------------------------------------------
# Menu RP features (from Engine)
# ---------------------------------------------------------------------------

def extract_menu_rp(
    user_logs: dict[str, object],  # str -> MenuChoiceLog
) -> pd.DataFrame:
    """Extract revealed preference features for menu-choice data via Engine.

    Args:
        user_logs: Dict mapping user_id -> MenuChoiceLog.

    Returns:
        DataFrame indexed by user_id with RP feature columns.
    """
    user_ids = list(user_logs.keys())
    engine_tuples = []
    for uid in user_ids:
        log = user_logs[uid]
        engine_tuples.append(log.to_engine_tuple())

    engine = Engine()
    results = engine.analyze_menus(engine_tuples)
    df = results_to_dataframe(results, user_ids=user_ids)

    # Add derived features
    df["hm_ratio"] = df["hm_consistent"] / df["hm_total"].replace(0, 1)
    n_obs = np.array([len(user_logs[uid].choices) for uid in user_ids])
    df["sarp_violation_density"] = df["n_sarp_violations"] / np.maximum(n_obs * (n_obs - 1) / 2, 1)
    df["warp_violation_density"] = df["n_warp_violations"] / np.maximum(n_obs * (n_obs - 1) / 2, 1)
    df["scc_ratio"] = df["max_scc"] / np.maximum(
        np.array([user_logs[uid].num_items for uid in user_ids]), 1
    )

    # Drop raw counts, keep ratios
    df = df.drop(columns=["hm_consistent", "hm_total", "compute_time_us"], errors="ignore")

    # Convert bools to int
    for col in ["is_sarp", "is_warp", "is_warp_la"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df
