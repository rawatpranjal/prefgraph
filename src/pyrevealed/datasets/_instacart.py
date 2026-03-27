"""Instacart Market Basket dataset loader.

Loads the Instacart "Market Basket Analysis" dataset and aggregates
orders at the department level (21 departments). Since individual
product prices are not available, uniform prices ($1/unit) are used,
reducing the RP analysis to quantity-consistency checks.

Data must be downloaded separately from Kaggle:
  kaggle datasets download -d instacart/market-basket-analysis
  unzip market-basket-analysis.zip -d ~/.pyrevealed/data/instacart/

Source: https://www.kaggle.com/c/instacart-market-basket-analysis
License: Competition-specific (research use)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from pyrevealed.core.panel import BehaviorPanel
from pyrevealed.core.session import BehaviorLog


def _find_data_dir(data_dir: str | Path | None) -> Path:
    """Find Instacart data directory via cascade."""
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))

    env = os.environ.get("PYREVEALED_DATA_DIR")
    if env:
        candidates.append(Path(env) / "instacart")

    candidates.extend([
        Path.home() / ".pyrevealed" / "data" / "instacart",
        Path(__file__).resolve().parents[3] / "datasets" / "instacart" / "data",
    ])

    for d in candidates:
        if d.is_dir() and (d / "orders.csv").exists():
            return d

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Instacart data not found. Searched:\n  {searched}\n\n"
        "Download from Kaggle:\n"
        "  kaggle datasets download -d instacart/market-basket-analysis\n"
        "  unzip market-basket-analysis.zip -d ~/.pyrevealed/data/instacart/\n\n"
        "Required files: orders.csv, order_products__prior.csv, products.csv, departments.csv"
    )


def load_instacart(
    data_dir: str | Path | None = None,
    max_users: int | None = None,
    min_orders: int = 10,
) -> BehaviorPanel:
    """Load Instacart dataset as a BehaviorPanel.

    Aggregates products at the department level (21 departments).
    Uses uniform prices ($1/unit) since individual prices are unavailable.

    Args:
        data_dir: Path to directory containing Instacart CSV files.
        max_users: Maximum number of users (None = all, default 5000 for speed).
        min_orders: Minimum orders per user (default 10).

    Returns:
        BehaviorPanel with one BehaviorLog per user.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for dataset loaders. "
            "Install with: pip install 'pyrevealed[datasets]'"
        ) from None

    data_path = _find_data_dir(data_dir)

    print(f"  Loading Instacart data from {data_path}...")

    # Load orders
    orders = pd.read_csv(data_path / "orders.csv")
    # Use only "prior" orders (the main historical data)
    prior_orders = orders[orders["eval_set"] == "prior"].copy()
    prior_orders = prior_orders.sort_values(["user_id", "order_number"])

    # Load order-product details
    order_products = pd.read_csv(data_path / "order_products__prior.csv")

    # Load product -> department mapping
    products = pd.read_csv(data_path / "products.csv")
    departments = pd.read_csv(data_path / "departments.csv")
    products = products.merge(departments, on="department_id")

    # Merge to get department per order-product
    order_products = order_products.merge(
        products[["product_id", "department_id"]],
        on="product_id",
    )

    # Count items per department per order
    dept_counts = (
        order_products
        .groupby(["order_id", "department_id"])
        .size()
        .reset_index(name="quantity")
    )

    # Merge with order info to get user_id and order_number
    dept_counts = dept_counts.merge(
        prior_orders[["order_id", "user_id", "order_number"]],
        on="order_id",
    )

    # Filter users with enough orders
    user_order_counts = prior_orders.groupby("user_id")["order_id"].nunique()
    qualifying_users = user_order_counts[user_order_counts >= min_orders].index

    if max_users is not None:
        qualifying_users = qualifying_users[:max_users]

    dept_counts = dept_counts[dept_counts["user_id"].isin(qualifying_users)]

    n_departments = departments["department_id"].max()
    dept_ids = sorted(dept_counts["department_id"].unique())
    dept_map = {d: i for i, d in enumerate(dept_ids)}
    n_cols = len(dept_ids)

    # Build per-user BehaviorLogs
    logs: dict[str, BehaviorLog] = {}

    for user_id, user_data in dept_counts.groupby("user_id"):
        # Pivot: rows = orders, columns = departments
        order_nums = sorted(user_data["order_number"].unique())
        T = len(order_nums)
        if T < min_orders:
            continue

        qty_matrix = np.zeros((T, n_cols))
        for _, row in user_data.iterrows():
            t_idx = order_nums.index(row["order_number"])
            d_idx = dept_map[row["department_id"]]
            qty_matrix[t_idx, d_idx] += row["quantity"]

        # Uniform prices
        price_matrix = np.ones_like(qty_matrix)

        uid = f"user_{user_id}"
        logs[uid] = BehaviorLog(
            cost_vectors=price_matrix,
            action_vectors=qty_matrix,
            user_id=uid,
        )

    print(f"  Built {len(logs)} BehaviorLog objects ({n_cols} departments)")

    return BehaviorPanel(
        _logs=logs,
        metadata={
            "dataset": "instacart",
            "goods": dept_ids,
            "n_departments": n_cols,
            "price_type": "uniform",
        },
    )
