"""Streaming Parquet reader for large-scale revealed preference analysis.

Reads Parquet files row-group by row-group, groups observations by user,
and yields chunks of engine-ready (prices, quantities) tuples. Memory stays
bounded at O(chunk_size) regardless of total dataset size.

Requires ``pyarrow``: ``pip install pyrevealed[parquet]``
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray


def _require_pyarrow() -> Any:
    try:
        import pyarrow
        return pyarrow
    except ImportError:
        raise ImportError(
            "pyarrow is required for Parquet support. "
            "Install with: pip install pyrevealed[parquet]"
        ) from None


class ParquetUserIterator:
    """Stream user data from Parquet files in engine-ready chunks.

    Reads one row group at a time, accumulates per-user observations,
    and yields chunks of ``(user_ids, engine_tuples)`` aligned to
    ``chunk_size`` users.

    Supports two data layouts:

    **Wide format** (one row per observation, items as columns)::

        user_id | period | price_A | price_B | qty_A | qty_B
        u1      | 1      | 1.0     | 2.0     | 4.0   | 1.0

    **Long format** (one row per item per time per user)::

        user_id | time | item | price | quantity
        u1      | 1    | A    | 1.0   | 4.0

    Args:
        path: Path to a Parquet file or directory of partitioned Parquet files.
        user_col: Column containing user identifiers.
        cost_cols: (Wide) List of price/cost column names.
        action_cols: (Wide) List of quantity/action column names.
        item_col: (Long) Column for item identifiers.
        cost_col: (Long) Single price column name.
        action_col: (Long) Single quantity column name.
        time_col: (Long) Column for time/observation identifiers.
        chunk_size: Number of complete users per yielded chunk.

    Yields:
        Tuples of ``(user_ids, engine_tuples)`` where ``engine_tuples``
        is ``list[tuple[ndarray, ndarray]]`` ready for ``Engine.analyze_arrays()``.
    """

    def __init__(
        self,
        path: str | Path,
        user_col: str = "user_id",
        *,
        cost_cols: list[str] | None = None,
        action_cols: list[str] | None = None,
        item_col: str | None = None,
        cost_col: str | None = None,
        action_col: str | None = None,
        time_col: str | None = None,
        chunk_size: int = 50_000,
    ):
        _require_pyarrow()
        self.path = Path(path)
        self.user_col = user_col
        self.chunk_size = chunk_size

        # Detect format
        has_wide = cost_cols is not None or action_cols is not None
        has_long = item_col is not None

        if has_wide and has_long:
            raise ValueError(
                "Provide either wide-format params (cost_cols/action_cols) "
                "or long-format params (item_col/cost_col/action_col/time_col), not both."
            )
        if not has_wide and not has_long:
            raise ValueError(
                "Provide either cost_cols/action_cols (wide) or item_col (long)."
            )

        if has_wide:
            if cost_cols is None or action_cols is None:
                raise ValueError("Wide format requires both cost_cols and action_cols.")
            self.format = "wide"
            self.cost_cols = cost_cols
            self.action_cols = action_cols
            self._read_cols = [user_col] + cost_cols + action_cols
        else:
            self.format = "long"
            self.item_col = item_col
            self.cost_col = cost_col or "price"
            self.action_col = action_col or "quantity"
            self.time_col = time_col or "time"
            self._read_cols = [user_col, self.time_col, self.item_col,
                               self.cost_col, self.action_col]

    def __iter__(self) -> Iterator[tuple[list[str], list[tuple[NDArray, NDArray]]]]:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(self.path)

        # Accumulator: user_id -> list of row dicts (wide) or raw rows (long)
        accum: dict[str, list] = defaultdict(list)
        # Track order of first appearance for stable iteration
        accum_order: list[str] = []

        # Buffer of finalized users ready to yield
        ready_ids: list[str] = []
        ready_tuples: list[tuple[NDArray, NDArray]] = []

        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=self._read_cols)
            self._ingest_table(table, accum, accum_order)

            # Check if we have enough complete users to yield a chunk.
            # Heuristic: users whose data is complete = those whose user_id
            # appeared in a prior row group but NOT this one (for sorted data).
            # For unsorted data, we can only finalize at end-of-file.
            # Compromise: finalize users when accumulator exceeds 2x chunk_size.
            if len(accum) >= self.chunk_size * 2:
                # Finalize the oldest chunk_size users
                to_finalize = accum_order[:self.chunk_size]
                for uid in to_finalize:
                    tup = self._finalize_user(uid, accum[uid])
                    if tup is not None:
                        ready_ids.append(uid)
                        ready_tuples.append(tup)
                    del accum[uid]
                accum_order = accum_order[self.chunk_size:]

                # Yield full chunks
                while len(ready_ids) >= self.chunk_size:
                    yield (
                        ready_ids[:self.chunk_size],
                        ready_tuples[:self.chunk_size],
                    )
                    ready_ids = ready_ids[self.chunk_size:]
                    ready_tuples = ready_tuples[self.chunk_size:]

        # Finalize remaining users in accumulator
        for uid in accum_order:
            if uid in accum:
                tup = self._finalize_user(uid, accum[uid])
                if tup is not None:
                    ready_ids.append(uid)
                    ready_tuples.append(tup)

        # Yield remaining chunks
        while ready_ids:
            batch_size = min(self.chunk_size, len(ready_ids))
            yield (
                ready_ids[:batch_size],
                ready_tuples[:batch_size],
            )
            ready_ids = ready_ids[batch_size:]
            ready_tuples = ready_tuples[batch_size:]

    def _ingest_table(
        self,
        table: Any,
        accum: dict[str, list],
        accum_order: list[str],
    ) -> None:
        """Add rows from an Arrow table into the user accumulator."""
        user_ids = table.column(self.user_col).to_pylist()

        if self.format == "wide":
            # Extract cost and action columns as numpy arrays
            costs = np.column_stack([
                table.column(c).to_numpy(zero_copy_only=False)
                for c in self.cost_cols
            ])  # (n_rows, K)
            actions = np.column_stack([
                table.column(c).to_numpy(zero_copy_only=False)
                for c in self.action_cols
            ])  # (n_rows, K)

            for i, uid in enumerate(user_ids):
                uid_str = str(uid)
                if uid_str not in accum:
                    accum_order.append(uid_str)
                accum[uid_str].append((costs[i], actions[i]))
        else:
            # Long format: store raw (time, item, cost, action) tuples
            times = table.column(self.time_col).to_pylist()
            items = table.column(self.item_col).to_pylist()
            cost_vals = table.column(self.cost_col).to_numpy(zero_copy_only=False)
            action_vals = table.column(self.action_col).to_numpy(zero_copy_only=False)

            for i, uid in enumerate(user_ids):
                uid_str = str(uid)
                if uid_str not in accum:
                    accum_order.append(uid_str)
                accum[uid_str].append((times[i], items[i], cost_vals[i], action_vals[i]))

    def _finalize_user(
        self,
        uid: str,
        rows: list,
    ) -> tuple[NDArray, NDArray] | None:
        """Convert accumulated rows for one user into (prices, quantities) arrays."""
        if not rows:
            return None

        if self.format == "wide":
            # rows = list of (cost_row_1d, action_row_1d)
            prices = np.array([r[0] for r in rows], dtype=np.float64)
            quantities = np.array([r[1] for r in rows], dtype=np.float64)
            if prices.ndim == 1:
                prices = prices.reshape(1, -1)
                quantities = quantities.reshape(1, -1)
            return (
                np.ascontiguousarray(prices),
                np.ascontiguousarray(quantities),
            )
        else:
            # Long format: pivot (time, item, cost, action) into T×K matrices
            # rows = list of (time, item, cost, action)
            times = sorted(set(r[0] for r in rows))
            items = sorted(set(r[1] for r in rows))
            if len(times) < 2 or len(items) < 2:
                return None  # Skip users with insufficient data

            time_idx = {t: i for i, t in enumerate(times)}
            item_idx = {it: i for i, it in enumerate(items)}
            T, K = len(times), len(items)

            prices = np.zeros((T, K), dtype=np.float64)
            quantities = np.zeros((T, K), dtype=np.float64)

            for t_val, it_val, c_val, a_val in rows:
                ti = time_idx[t_val]
                ii = item_idx[it_val]
                prices[ti, ii] = float(c_val)
                quantities[ti, ii] = float(a_val)

            return (
                np.ascontiguousarray(prices),
                np.ascontiguousarray(quantities),
            )


def prepare_parquet(
    input_path: str | Path,
    output_path: str | Path,
    user_col: str = "user_id",
    *,
    row_group_size: int | None = None,
    compression: str = "zstd",
) -> dict[str, Any]:
    """Sort and rewrite a Parquet/CSV file for optimal streaming reads.

    Sorts by ``user_col`` so that all observations for a user are contiguous,
    then writes Parquet with controlled row group sizes. This one-time prep
    step ensures subsequent ``ParquetUserIterator`` reads never need to buffer
    users across row groups.

    Args:
        input_path: Source file (Parquet or CSV).
        output_path: Destination Parquet file.
        user_col: Column to sort by.
        row_group_size: Rows per row group. Defaults to 1_000_000.
            For best alignment with ``Engine(chunk_size=50_000)``, set to
            ``chunk_size * avg_observations_per_user``.
        compression: Parquet compression codec (default ``"zstd"``).

    Returns:
        Dict with ``n_rows``, ``n_users``, ``n_row_groups``, ``output_path``.
    """
    pa = _require_pyarrow()
    import pyarrow.parquet as pq
    import pyarrow.csv as csv_reader

    input_path = Path(input_path)
    output_path = Path(output_path)

    if row_group_size is None:
        row_group_size = 1_000_000

    # Read input
    if input_path.suffix == ".csv":
        table = csv_reader.read_csv(str(input_path))
    else:
        table = pq.read_table(str(input_path))

    # Sort by user_col for contiguous user data
    sort_indices = pa.compute.sort_indices(table, sort_keys=[(user_col, "ascending")])
    table = table.take(sort_indices)

    # Write with controlled row group size
    pq.write_table(
        table,
        str(output_path),
        row_group_size=row_group_size,
        compression=compression,
    )

    # Compute stats
    pf = pq.ParquetFile(str(output_path))
    n_users = len(table.column(user_col).unique())

    return {
        "n_rows": len(table),
        "n_users": n_users,
        "n_row_groups": pf.metadata.num_row_groups,
        "output_path": str(output_path),
    }
