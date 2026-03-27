"""Ecuador Favorita Grocery dataset loader.

Loads the Corporacion Favorita grocery sales dataset of 54 stores
across 33 product families over ~5 years of daily transactions,
returning a BehaviorPanel.

Aggregates item-level daily unit_sales into weekly store-level panels.
Since individual product prices are not available in this dataset,
uniform prices ($1/unit) are used, reducing the RP analysis to
quantity-consistency checks (same approach as Instacart).

Data must be downloaded separately from Kaggle:
  kaggle competitions download -c favorita-grocery-sales-forecasting
  unzip favorita-grocery-sales-forecasting.zip -d ~/.pyrevealed/data/favorita/

Source: https://www.kaggle.com/c/favorita-grocery-sales-forecasting
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from pyrevealed.core.panel import BehaviorPanel
from pyrevealed.core.session import BehaviorLog

# --- Constants ---

# All 33 product families in the dataset.
PRODUCT_FAMILIES = [
    "AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES", "BOOKS",
    "BREAD/BAKERY", "CELEBRATION", "CLEANING", "DAIRY", "DELI",
    "EGGS", "FROZEN FOODS", "GROCERY I", "GROCERY II", "HARDWARE",
    "HOME AND KITCHEN", "HOME APPLIANCES", "HOME CARE", "LADIESWEAR",
    "LAWN AND GARDEN", "LINGERIE", "LIQUOR,WINE,BEER", "MAGAZINES",
    "MEATS", "PERSONAL CARE", "PET SUPPLIES", "PLAYERS AND ELECTRONICS",
    "POULTRY", "PREPARED FOODS", "PRODUCE", "SCHOOL AND OFFICE SUPPLIES",
    "SEAFOOD", "STATIONERY",
]

NUM_FAMILIES = len(PRODUCT_FAMILIES)

# Chunk size for reading the ~125M-row train.csv (4.8 GB).
CHUNK_SIZE = 5_000_000


def _find_data_dir(data_dir: str | Path | None) -> Path:
    """Find Favorita data directory via cascade."""
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))

    env = os.environ.get("PYREVEALED_DATA_DIR")
    if env:
        candidates.append(Path(env) / "favorita")

    candidates.extend([
        Path.home() / ".pyrevealed" / "data" / "favorita",
        Path(__file__).resolve().parents[3] / "favorita" / "data",
    ])

    for d in candidates:
        if d.is_dir() and (d / "train.csv").exists():
            return d

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Favorita data not found. Searched:\n  {searched}\n\n"
        "Download from Kaggle:\n"
        "  kaggle competitions download -c favorita-grocery-sales-forecasting\n"
        "  unzip favorita-grocery-sales-forecasting.zip -d ~/.pyrevealed/data/favorita/\n\n"
        "Required files: train.csv, items.csv, stores.csv\n"
        "Then pass data_dir= or set PYREVEALED_DATA_DIR environment variable."
    )


def load_favorita(
    data_dir: str | Path | None = None,
    min_weeks: int = 50,
    max_stores: int | None = None,
) -> BehaviorPanel:
    """Load Ecuador Favorita grocery dataset as a BehaviorPanel.

    Each store is a "user". Rows are weeks, columns are 33 product families.
    Quantities are total unit_sales per family per week. Prices are uniform
    ($1/unit) since individual prices are not in the dataset.

    The 4.8 GB train.csv is read in 5M-row chunks to stay memory-friendly.

    Args:
        data_dir: Path to directory containing train.csv, items.csv, and
            stores.csv. If None, searches standard locations.
        min_weeks: Minimum weeks with nonzero sales required per store
            (default 50).
        max_stores: Optional cap on number of stores returned (None = all 54).

    Returns:
        BehaviorPanel with one BehaviorLog per store.

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

    print(f"  Loading Favorita data from {data_path}...")

    # --- Load item -> family mapping ---
    items = pd.read_csv(data_path / "items.csv", usecols=["item_nbr", "family"])
    item_to_family = dict(zip(items["item_nbr"], items["family"]))

    # --- Read train.csv in chunks, aggregate to store-week-family ---
    # Only load needed columns to keep memory low.
    use_cols = ["date", "store_nbr", "item_nbr", "unit_sales"]

    # Accumulator: {(store_nbr, iso_week_str, family): total_units}
    agg: dict[tuple[int, str, str], float] = {}

    print("  Reading train.csv in chunks...")
    chunk_num = 0
    for chunk in pd.read_csv(
        data_path / "train.csv",
        usecols=use_cols,
        dtype={"store_nbr": np.int32, "item_nbr": np.int32, "unit_sales": np.float32},
        parse_dates=["date"],
        chunksize=CHUNK_SIZE,
    ):
        chunk_num += 1
        if chunk_num % 5 == 0:
            print(f"    chunk {chunk_num}...")

        # Clip negative unit_sales to zero (returns/refunds).
        chunk["unit_sales"] = chunk["unit_sales"].clip(lower=0)

        # Map item_nbr -> family.
        chunk["family"] = chunk["item_nbr"].map(item_to_family)
        chunk.dropna(subset=["family"], inplace=True)

        # Compute ISO week string for grouping (YYYY-WW).
        chunk["week"] = (
            chunk["date"].dt.isocalendar().year.astype(str)
            + "-"
            + chunk["date"].dt.isocalendar().week.astype(str).str.zfill(2)
        )

        # Group and accumulate.
        grouped = chunk.groupby(["store_nbr", "week", "family"])["unit_sales"].sum()
        for (store, week, family), total in grouped.items():
            key = (int(store), week, family)
            agg[key] = agg.get(key, 0.0) + float(total)

    print(f"  Finished reading {chunk_num} chunks.")

    # --- Build per-store BehaviorLogs ---
    # Collect all weeks and sort them.
    all_weeks = sorted({k[1] for k in agg})
    week_to_idx = {w: i for i, w in enumerate(all_weeks)}
    n_weeks = len(all_weeks)

    # Collect all stores.
    all_stores = sorted({k[0] for k in agg})
    if max_stores is not None:
        all_stores = all_stores[:max_stores]

    # Build family -> column index map.
    family_to_col = {f: i for i, f in enumerate(PRODUCT_FAMILIES)}

    logs: dict[str, BehaviorLog] = {}

    for store in all_stores:
        # Build quantity matrix: (n_weeks, 33).
        qty_matrix = np.zeros((n_weeks, NUM_FAMILIES), dtype=np.float64)

        for (s, week, family), total in agg.items():
            if s != store:
                continue
            col = family_to_col.get(family)
            if col is None:
                continue
            row = week_to_idx[week]
            qty_matrix[row, col] = total

        # Keep only weeks with nonzero total sales.
        row_sums = qty_matrix.sum(axis=1)
        active_mask = row_sums > 0
        active_weeks = np.where(active_mask)[0]

        if len(active_weeks) < min_weeks:
            continue

        qty_active = qty_matrix[active_weeks]

        # Uniform prices ($1/unit).
        price_matrix = np.ones_like(qty_active)

        uid = f"store_{store}"
        logs[uid] = BehaviorLog(
            cost_vectors=price_matrix,
            action_vectors=qty_active,
            user_id=uid,
        )

    print(f"  Built {len(logs)} BehaviorLog objects "
          f"({NUM_FAMILIES} product families, {n_weeks} weeks available)")

    return BehaviorPanel(
        _logs=logs,
        metadata={
            "dataset": "favorita",
            "goods": PRODUCT_FAMILIES,
            "n_families": NUM_FAMILIES,
            "n_weeks_available": n_weeks,
            "min_weeks": min_weeks,
            "price_type": "uniform",
        },
    )
