"""Online Retail II dataset loader.

Loads the Online Retail II dataset of ~5,942 UK e-commerce customers
over Dec 2009 to Dec 2011, returning a BehaviorPanel.

Budget-based with real prices. Monthly aggregation, top-N products
by transaction frequency.

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
TOP_N_CATEGORIES = 30
MIN_MONTHS = 4
CUTOFF_DATE = "2011-06-01"


def _find_data_dir(data_dir: str | Path | None) -> Path:
    """Find Online Retail II data directory via cascade."""
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))

    env = os.environ.get("PYREVEALED_DATA_DIR")
    if env:
        candidates.append(Path(env) / "online_retail_ii")

    candidates.extend([
        Path.home() / ".prefgraph" / "data" / "online_retail_ii",
        Path(__file__).resolve().parents[3] / "datasets" / "online_retail_ii" / "data",
    ])

    for d in candidates:
        if d.is_dir():
            for fname in [
                "online_retail_II.csv",
                "online_retail_ii.csv",
                "Online Retail II.csv",
                "online_retail_II.xlsx",
            ]:
                if (d / fname).exists():
                    return d

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Online Retail II data not found. Searched:\n  {searched}\n\n"
        "Download from: https://archive.ics.uci.edu/dataset/502/online+retail+ii\n"
        "Place online_retail_II.csv in ~/.prefgraph/data/online_retail_ii/\n"
        "Or pass data_dir= or set PYREVEALED_DATA_DIR environment variable."
    )


def _load_raw(data_path: Path) -> "pd.DataFrame":
    """Load raw data, trying csv then xlsx."""
    import pandas as pd

    for fname in [
        "online_retail_II.csv",
        "online_retail_ii.csv",
        "Online Retail II.csv",
    ]:
        fpath = data_path / fname
        if fpath.exists():
            return pd.read_csv(fpath)

    xlsx_path = data_path / "online_retail_II.xlsx"
    if xlsx_path.exists():
        return pd.read_excel(xlsx_path)

    raise FileNotFoundError(f"No Online Retail II file found in {data_path}")


def load_online_retail_ii(
    data_dir: str | Path | None = None,
    n_customers: int | None = None,
    min_months: int = MIN_MONTHS,
    top_n_categories: int = TOP_N_CATEGORIES,
) -> BehaviorPanel:
    """Load Online Retail II dataset as a BehaviorPanel.

    Args:
        data_dir: Path to directory containing online_retail_II.csv.
            If None, searches standard locations.
        n_customers: Max number of customers to include (None = all).
        min_months: Minimum active months per customer (default 4).
        top_n_categories: Number of top products by frequency (default 30).

    Returns:
        BehaviorPanel with one BehaviorLog per customer.
        Metadata includes 'cutoff_date' for train/test splitting.

    Raises:
        FileNotFoundError: If data files cannot be found.
        ImportError: If pandas is not installed.
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

    # --- Filtering ---

    # Remove cancellations (Invoice starts with 'C')
    df["Invoice"] = df["Invoice"].astype(str)
    df = df[~df["Invoice"].str.startswith("C")]

    # Drop null Customer IDs
    df = df.dropna(subset=["Customer ID"])
    df["Customer ID"] = df["Customer ID"].astype(int)

    # Filter negative/zero quantities and price outliers
    df = df[
        (df["Quantity"] > 0) &
        (df["Price"].between(MIN_UNIT_PRICE, MAX_UNIT_PRICE))
    ]

    # Parse dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    # --- Monthly periods ---
    df["period"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    # --- Select top products by transaction count ---
    df["StockCode"] = df["StockCode"].astype(str)
    top_products = df["StockCode"].value_counts().head(top_n_categories).index.tolist()
    df = df[df["StockCode"].isin(top_products)]
    products = sorted(str(p) for p in top_products)

    # --- Build price oracle: median price per month per product ---
    periods = sorted(df["period"].unique())
    price_pivot = df.pivot_table(
        values="Price", index="period", columns="StockCode",
        aggfunc="median",
    ).reindex(index=periods, columns=products)
    price_pivot = price_pivot.ffill().bfill().fillna(price_pivot.median())
    price_grid = price_pivot.values
    period_to_idx = {p: i for i, p in enumerate(periods)}

    # --- Build per-customer BehaviorLogs ---
    logs: dict[str, BehaviorLog] = {}
    grouped = df.groupby("Customer ID")
    customer_ids = list(grouped.groups.keys())

    if n_customers is not None:
        customer_ids = customer_ids[:n_customers]

    for cid in customer_ids:
        cust_data = grouped.get_group(cid)

        # Aggregate quantity per month per product
        qty_pivot = cust_data.pivot_table(
            values="Quantity", index="period", columns="StockCode",
            aggfunc="sum",
        ).reindex(columns=products).fillna(0)

        active_periods = qty_pivot[qty_pivot.sum(axis=1) > 0].index.tolist()
        if len(active_periods) < min_months:
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
            "dataset": "online_retail_ii",
            "goods": products,
            "min_months": min_months,
            "top_n_categories": top_n_categories,
            "cutoff_date": CUTOFF_DATE,
            "date_range": "2009-12 to 2011-12",
        },
    )
