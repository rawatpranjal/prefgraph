"""UCI Online Retail dataset loader.

Loads the UCI Online Retail dataset of ~1,800 UK B2B customers,
returning a BehaviorPanel.

Data must be downloaded separately from UCI ML Repository.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from prefgraph.core.panel import BehaviorPanel
from prefgraph.core.session import BehaviorLog

# --- Constants ---

MIN_UNIT_PRICE = 0.01
MAX_UNIT_PRICE = 500.0
TOP_N_PRODUCTS = 50
MIN_TRANSACTIONS = 5


def _find_data_dir(data_dir: str | Path | None) -> Path:
    """Find UCI retail data directory via cascade."""
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))

    env = os.environ.get("PYREVEALED_DATA_DIR")
    if env:
        candidates.append(Path(env) / "uci_retail")

    candidates.extend([
        Path.home() / ".prefgraph" / "data" / "uci_retail",
        Path(__file__).resolve().parents[3] / "datasets" / "uci_retail" / "data",
    ])

    for d in candidates:
        if d.is_dir():
            for fname in ["online_retail.xlsx", "Online Retail.xlsx", "online_retail.csv"]:
                if (d / fname).exists():
                    return d

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"UCI Online Retail data not found. Searched:\n  {searched}\n\n"
        "Download from: https://archive.ics.uci.edu/ml/datasets/Online+Retail\n"
        "Then pass data_dir= or set PYREVEALED_DATA_DIR environment variable."
    )


def _load_raw(data_path: Path) -> "pd.DataFrame":
    """Load raw data, trying xlsx then csv."""
    import pandas as pd

    for fname in ["online_retail.xlsx", "Online Retail.xlsx"]:
        fpath = data_path / fname
        if fpath.exists():
            return pd.read_excel(fpath)

    csv_path = data_path / "online_retail.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(f"No online_retail file found in {data_path}")


def load_uci_retail(
    data_dir: str | Path | None = None,
    n_customers: int | None = None,
    min_transactions: int = MIN_TRANSACTIONS,
    top_n_products: int = TOP_N_PRODUCTS,
) -> BehaviorPanel:
    """Load UCI Online Retail dataset as a BehaviorPanel.

    Args:
        data_dir: Path to directory containing online_retail.xlsx.
        n_customers: Max number of customers to include (None = all).
        min_transactions: Minimum active months per customer (default 5).
        top_n_products: Number of top products to include (default 50).

    Returns:
        BehaviorPanel with one BehaviorLog per customer.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for dataset loaders. "
            "Install with: pip install 'prefgraph[datasets]'"
        ) from None

    data_path = _find_data_dir(data_dir)
    df = _load_raw(data_path)

    # Filter cancelled orders and missing customers
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Filter prices and quantities
    df = df[
        (df["UnitPrice"].between(MIN_UNIT_PRICE, MAX_UNIT_PRICE)) &
        (df["Quantity"] > 0)
    ]

    # Create monthly period
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df["period"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    # Select top products by transaction count
    top_products = df["StockCode"].value_counts().head(top_n_products).index.tolist()
    df = df[df["StockCode"].isin(top_products)]
    products = sorted(str(p) for p in top_products)

    # Ensure StockCode is string for consistent handling
    df["StockCode"] = df["StockCode"].astype(str)

    # Build price oracle
    periods = sorted(df["period"].unique())
    price_pivot = df.pivot_table(
        values="UnitPrice", index="period", columns="StockCode",
        aggfunc="median",
    ).reindex(index=periods, columns=products)
    price_pivot = price_pivot.ffill().bfill().fillna(price_pivot.median())
    price_grid = price_pivot.values
    period_to_idx = {p: i for i, p in enumerate(periods)}

    # Build per-customer sessions
    logs: dict[str, BehaviorLog] = {}
    grouped = df.groupby("CustomerID")
    customer_ids = list(grouped.groups.keys())

    if n_customers is not None:
        customer_ids = customer_ids[:n_customers]

    for cid in customer_ids:
        cust_data = grouped.get_group(cid)

        qty_pivot = cust_data.pivot_table(
            values="Quantity", index="period", columns="StockCode",
            aggfunc="sum",
        ).reindex(columns=products).fillna(0)

        active_periods = qty_pivot[qty_pivot.sum(axis=1) > 0].index.tolist()
        if len(active_periods) < min_transactions:
            continue

        qty_matrix = qty_pivot.loc[active_periods].values
        price_indices = [period_to_idx[p] for p in active_periods if p in period_to_idx]
        if len(price_indices) != len(active_periods):
            continue
        price_matrix = price_grid[price_indices]

        uid = f"customer_{cid}"
        logs[uid] = BehaviorLog(
            cost_vectors=price_matrix,
            action_vectors=qty_matrix,
            user_id=uid,
        )

    return BehaviorPanel(
        _logs=logs,
        metadata={
            "dataset": "uci_retail",
            "goods": products,
            "min_transactions": min_transactions,
        },
    )
