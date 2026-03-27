"""H&M Fashion dataset loader.

Loads the H&M Personalized Fashion Recommendations dataset of ~1.36M
customers purchasing clothing articles over 2 years (2018-09 to 2020-09),
returning a BehaviorPanel.

Articles are aggregated into product groups (first 2 digits of article_id).
Transactions are aggregated to monthly periods: for each customer-month,
quantity per product group. A price oracle provides median price per product
group per month.

Data must be downloaded separately from Kaggle.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from pyrevealed.core.panel import BehaviorPanel
from pyrevealed.core.session import BehaviorLog

# --- Constants ---

MAX_PRODUCT_GROUPS = 20
DEFAULT_MAX_USERS = 50_000
DEFAULT_MIN_MONTHS = 6
CHUNKSIZE = 500_000
CUTOFF_DATE = "2020-06-01"


def _find_data_dir(data_dir: str | Path | None) -> Path:
    """Find H&M data directory via cascade."""
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))

    env = os.environ.get("PYREVEALED_DATA_DIR")
    if env:
        candidates.append(Path(env) / "hm")

    candidates.extend([
        Path.home() / ".pyrevealed" / "data" / "hm",
        Path(__file__).resolve().parents[3] / "hm" / "data",
    ])

    for d in candidates:
        if d.is_dir() and (d / "transactions_train.csv").exists():
            return d

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"H&M data not found. Searched:\n  {searched}\n\n"
        "Download from Kaggle: https://www.kaggle.com/competitions/"
        "h-and-m-personalized-fashion-recommendations/data\n"
        "Place transactions_train.csv in the data directory.\n"
        "Then pass data_dir= or set PYREVEALED_DATA_DIR environment variable."
    )


def load_hm(
    data_dir: str | Path | None = None,
    max_users: int = DEFAULT_MAX_USERS,
    min_months: int = DEFAULT_MIN_MONTHS,
    top_k_groups: int = MAX_PRODUCT_GROUPS,
    cutoff_date: str = CUTOFF_DATE,
) -> BehaviorPanel:
    """Load H&M Fashion dataset as a BehaviorPanel.

    Reads transactions_train.csv in chunks for memory efficiency.
    Maps article_id to product groups (first 2 digits), aggregates
    to monthly price-quantity panels per customer.

    Args:
        data_dir: Path to directory containing transactions_train.csv.
            If None, searches standard locations.
        max_users: Maximum number of customers to include. Selects the
            most active users by total transaction count (default 50000).
        min_months: Minimum active months per customer (default 6).
        top_k_groups: Number of top product groups by frequency to keep
            (default 20).
        cutoff_date: ISO date string for train/test split boundary.
            Stored in metadata for out-of-time evaluation (default
            '2020-06-01').

    Returns:
        BehaviorPanel with one BehaviorLog per customer (rows = months,
        cols = product groups).

    Raises:
        FileNotFoundError: If data files cannot be found.
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for dataset loaders. "
            "Install with: pip install 'pyrevealed[datasets]'"
        ) from None

    data_path = _find_data_dir(data_dir)
    csv_path = data_path / "transactions_train.csv"

    # --- Pass 1: chunked scan to find top product groups and active users ---
    group_counts: dict[str, int] = {}
    user_counts: dict[str, int] = {}

    for chunk in pd.read_csv(
        csv_path,
        usecols=["customer_id", "article_id"],
        dtype={"customer_id": str, "article_id": str},
        chunksize=CHUNKSIZE,
    ):
        # Product group = first 2 digits of zero-padded article_id
        chunk["product_group"] = chunk["article_id"].str[:2]

        for grp, cnt in chunk["product_group"].value_counts().items():
            group_counts[grp] = group_counts.get(grp, 0) + cnt

        for uid, cnt in chunk["customer_id"].value_counts().items():
            user_counts[uid] = user_counts.get(uid, 0) + cnt

    # Select top product groups by total frequency
    sorted_groups = sorted(group_counts, key=group_counts.get, reverse=True)
    top_groups = sorted_groups[:top_k_groups]

    # Select most active users
    sorted_users = sorted(user_counts, key=user_counts.get, reverse=True)
    target_users = set(sorted_users[:max_users])

    # --- Pass 2: chunked load of filtered data ---
    frames = []

    for chunk in pd.read_csv(
        csv_path,
        dtype={"customer_id": str, "article_id": str, "sales_channel_id": int},
        parse_dates=["t_dat"],
        chunksize=CHUNKSIZE,
    ):
        chunk["product_group"] = chunk["article_id"].str[:2]
        mask = (
            chunk["customer_id"].isin(target_users)
            & chunk["product_group"].isin(top_groups)
        )
        if mask.any():
            frames.append(chunk.loc[mask, [
                "t_dat", "customer_id", "product_group", "price",
            ]])

    df = pd.concat(frames, ignore_index=True)

    # --- Month period key (YYYY-MM) ---
    df["month"] = df["t_dat"].dt.to_period("M")
    months_sorted = sorted(df["month"].unique())
    month_to_idx = {m: i for i, m in enumerate(months_sorted)}
    month_labels = [str(m) for m in months_sorted]

    # --- Price oracle: median price per product group per month ---
    price_oracle = df.pivot_table(
        values="price",
        index="month",
        columns="product_group",
        aggfunc="median",
    ).reindex(index=months_sorted, columns=top_groups)
    price_oracle = price_oracle.ffill().bfill()

    # Fill any remaining NaN with global median per group
    for col in price_oracle.columns:
        if price_oracle[col].isna().any():
            price_oracle[col].fillna(price_oracle[col].median(), inplace=True)
    # Last resort: fill with 0.01 if an entire column is NaN
    price_oracle = price_oracle.fillna(0.01)
    price_grid = price_oracle.values.astype(np.float64)  # (n_months, n_groups)

    # --- Build per-customer BehaviorLogs ---
    # Aggregate quantity: count of transactions per customer-month-group
    agg = df.groupby(["customer_id", "month", "product_group"]).size()
    agg = agg.reset_index(name="quantity")

    logs: dict[str, BehaviorLog] = {}

    for cid, cust_data in agg.groupby("customer_id"):
        # Pivot: rows = months, cols = product groups
        qty_pivot = cust_data.pivot_table(
            values="quantity",
            index="month",
            columns="product_group",
            aggfunc="sum",
        ).reindex(index=months_sorted, columns=top_groups).fillna(0)

        # Active months: at least one purchase in any group
        active_mask = qty_pivot.sum(axis=1) > 0
        active_months = qty_pivot.index[active_mask].tolist()

        if len(active_months) < min_months:
            continue

        active_indices = [month_to_idx[m] for m in active_months]

        qty_matrix = qty_pivot.loc[active_months].values.astype(np.float64)
        price_matrix = price_grid[active_indices]

        uid = f"customer_{cid[:12]}"
        logs[uid] = BehaviorLog(
            cost_vectors=price_matrix,
            action_vectors=qty_matrix,
            user_id=uid,
        )

    return BehaviorPanel(
        _logs=logs,
        metadata={
            "dataset": "hm",
            "goods": top_groups,
            "goods_labels": [f"group_{g}" for g in top_groups],
            "months": month_labels,
            "min_months": min_months,
            "max_users": max_users,
            "top_k_groups": top_k_groups,
            "cutoff_date": cutoff_date,
            "num_months_available": len(months_sorted),
        },
    )
