#!/usr/bin/env python3
"""PrefGraph — End-to-End Format Benchmark.

Measures the full disk-to-scores pipeline for budget and menu data
across file formats (CSV, Parquet) to verify the claim:
"analyze large graphs for millions of users direct from disk
 and compute inconsistency scores in under a minute."

    CSV / Parquet on disk  →  Polars read  →  array construction
    →  Rust + Rayon + SCC  →  scored results

Usage:
    python benchmarks/bench_e2e_formats.py                  # 100K users
    python benchmarks/bench_e2e_formats.py --users 10000    # quick test
    python benchmarks/bench_e2e_formats.py --users 1000000  # 1M users
"""

import argparse
import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Formatting helpers (shared with bench_parquet_1m.py) ──────────────

def fmt_time(s):
    if s < 1:
        return f"{s * 1000:.0f}ms"
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{int(s // 60)}m {int(s % 60)}s"
    return f"{s / 3600:.1f}h"


def fmt_bytes(b):
    if b < 1024:
        return f"{b}B"
    if b < 1024**2:
        return f"{b / 1024:.1f}KB"
    if b < 1024**3:
        return f"{b / 1024**2:.1f}MB"
    return f"{b / 1024**3:.2f}GB"


# ── Data generation ───────────────────────────────────────────────────

def users_to_wide_df(users, n_goods):
    """Convert list of (prices, quantities) tuples to a wide Polars DataFrame.

    Output columns: user_id, period, price_0..price_{K-1}, qty_0..qty_{K-1}
    """
    import polars as pl

    rows_uid = []
    rows_period = []
    rows_prices = [[] for _ in range(n_goods)]
    rows_qtys = [[] for _ in range(n_goods)]

    for uid, (prices, quantities) in enumerate(users):
        t = prices.shape[0]
        for obs in range(t):
            rows_uid.append(f"u{uid:07d}")
            rows_period.append(obs)
            for g in range(n_goods):
                rows_prices[g].append(float(prices[obs, g]))
                rows_qtys[g].append(float(quantities[obs, g]))

    data = {"user_id": rows_uid, "period": rows_period}
    for g in range(n_goods):
        data[f"price_{g}"] = rows_prices[g]
    for g in range(n_goods):
        data[f"qty_{g}"] = rows_qtys[g]

    return pl.DataFrame(data)


def generate_menu_data(n_users=1000, n_items=50, min_sessions=20,
                       max_sessions=100, seed=42):
    """Generate synthetic menu choice data.

    Returns list of (menus, choices, n_items) tuples for Engine.analyze_menus().
    """
    rng = np.random.default_rng(seed)
    users = []
    for _ in range(n_users):
        n_sessions = int(rng.integers(min_sessions, max_sessions + 1))
        menus = []
        choices = []
        for _ in range(n_sessions):
            menu_size = int(rng.integers(2, min(10, n_items) + 1))
            menu = sorted(rng.choice(n_items, size=menu_size, replace=False).tolist())
            choice = menu[int(rng.integers(0, len(menu)))]
            menus.append(menu)
            choices.append(choice)
        users.append((menus, choices, n_items))
    return users


def menu_to_csv_df(menu_users):
    """Convert menu tuples to a flat Polars DataFrame for CSV storage.

    Output columns: user_id, session, item_indices (comma-separated), choice
    """
    import polars as pl

    rows_uid, rows_session, rows_menu, rows_choice = [], [], [], []
    for uid, (menus, choices, _) in enumerate(menu_users):
        for sid, (menu, choice) in enumerate(zip(menus, choices)):
            rows_uid.append(f"m{uid:05d}")
            rows_session.append(sid)
            rows_menu.append(",".join(str(x) for x in menu))
            rows_choice.append(choice)

    return pl.DataFrame({
        "user_id": rows_uid,
        "session": rows_session,
        "menu_items": rows_menu,
        "choice": rows_choice,
    })


def csv_to_menu_tuples(df):
    """Reconstruct menu tuples from a flat Polars DataFrame."""
    users = {}
    n_items_map = {}
    for row in df.iter_rows(named=True):
        uid = row["user_id"]
        menu = [int(x) for x in row["menu_items"].split(",")]
        choice = row["choice"]
        if uid not in users:
            users[uid] = ([], [])
            n_items_map[uid] = 0
        users[uid][0].append(menu)
        users[uid][1].append(choice)
        n_items_map[uid] = max(n_items_map[uid], max(menu) + 1)

    return [
        (menus, choices, n_items_map[uid])
        for uid, (menus, choices) in users.items()
    ]


def df_to_engine_tuples(df, n_goods):
    """Convert wide Polars DataFrame back to list of (prices, quantities) tuples."""
    import polars as pl

    cost_cols = [f"price_{g}" for g in range(n_goods)]
    action_cols = [f"qty_{g}" for g in range(n_goods)]

    tuples = []
    for uid, group in df.group_by("user_id", maintain_order=True):
        prices = group.select(cost_cols).to_numpy().astype(np.float64)
        quantities = group.select(action_cols).to_numpy().astype(np.float64)
        tuples.append((
            np.ascontiguousarray(prices),
            np.ascontiguousarray(quantities),
        ))
    return tuples


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PrefGraph E2E Format Benchmark")
    parser.add_argument("--users", type=int, default=100_000,
                        help="Number of budget users (default 100K)")
    parser.add_argument("--menu-users", type=int, default=1000,
                        help="Number of menu users (default 1000)")
    parser.add_argument("--k", type=int, default=5, help="Goods (default 5)")
    parser.add_argument("--chunk", type=int, default=50_000)
    parser.add_argument("--output-dir", type=str, default="benchmarks/output/e2e")
    args = parser.parse_args()

    N = args.users
    K = args.k
    N_MENU = args.menu_users
    CHUNK = args.chunk
    CPU = os.cpu_count() or 1
    METRICS = ["garp", "ccei", "mpi", "harp", "hm"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"budget_{N // 1000}k.csv"
    pq_path = out_dir / f"budget_{N // 1000}k.parquet"
    menu_csv_path = out_dir / f"menu_{N_MENU}.csv"

    # ── Header ─────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  PrefGraph  —  End-to-End Format Benchmark")
    print("  CSV vs Parquet  ·  Polars  ·  Rust + Rayon")
    print("=" * 72)

    # ── Phase 0: Generate & write data ─────────────────────────────
    print(f"\n  [0/5] GENERATING DATA")
    print("  " + "-" * 70)

    import polars as pl
    from prefgraph.datasets import load_demo

    # Budget data
    print(f"    Budget: {N:,} users, K={K}, T=15 obs each ...")
    t0 = time.perf_counter()
    users = load_demo(n_users=N, n_goods=K)
    t_gen = time.perf_counter() - t0
    print(f"    Generated in {fmt_time(t_gen)}")

    print(f"    Converting to wide DataFrame ...")
    t0 = time.perf_counter()
    df = users_to_wide_df(users, K)
    t_convert = time.perf_counter() - t0
    total_rows = len(df)
    print(f"    {total_rows:,} rows in {fmt_time(t_convert)}")

    # Write CSV
    t0 = time.perf_counter()
    df.write_csv(csv_path)
    t_write_csv = time.perf_counter() - t0
    csv_size = csv_path.stat().st_size

    # Write Parquet
    t0 = time.perf_counter()
    df.write_parquet(pq_path, compression="zstd")
    t_write_pq = time.perf_counter() - t0
    pq_size = pq_path.stat().st_size

    print(f"    CSV:     {fmt_bytes(csv_size)} written in {fmt_time(t_write_csv)}")
    print(f"    Parquet: {fmt_bytes(pq_size)} (zstd) written in {fmt_time(t_write_pq)}")
    print(f"    Compression ratio: {csv_size / pq_size:.1f}x")

    # Menu data
    print(f"\n    Menu: {N_MENU:,} users, 50 items, 20-100 sessions ...")
    t0 = time.perf_counter()
    menu_users = generate_menu_data(n_users=N_MENU)
    t_menu_gen = time.perf_counter() - t0

    menu_df = menu_to_csv_df(menu_users)
    menu_df.write_csv(menu_csv_path)
    menu_csv_size = menu_csv_path.stat().st_size
    print(f"    {len(menu_df):,} rows, CSV: {fmt_bytes(menu_csv_size)}")

    # Free the big DataFrame to reduce memory noise
    del df

    # ── Phase 1: Budget — CSV path ────────────────────────────────
    print(f"\n  [1/5] BUDGET — CSV PATH  ({N:,} users, 5 metrics)")
    print("  " + "-" * 70)

    from prefgraph.engine import Engine

    # Read CSV
    t0 = time.perf_counter()
    df_csv = pl.read_csv(csv_path)
    t_csv_read = time.perf_counter() - t0

    # Transform to engine tuples
    t0 = time.perf_counter()
    tuples_csv = df_to_engine_tuples(df_csv, K)
    t_csv_transform = time.perf_counter() - t0
    del df_csv

    # Score
    engine = Engine(metrics=METRICS, chunk_size=CHUNK)
    t0 = time.perf_counter()
    results_csv = engine.analyze_arrays(tuples_csv)
    t_csv_score = time.perf_counter() - t0
    t_csv_total = t_csv_read + t_csv_transform + t_csv_score

    print(f"    Read:      {fmt_time(t_csv_read)}")
    print(f"    Transform: {fmt_time(t_csv_transform)}")
    print(f"    Score:     {fmt_time(t_csv_score)}")
    print(f"    TOTAL:     {fmt_time(t_csv_total)}")
    print(f"    Throughput: {N / t_csv_total:,.0f} users/sec (end-to-end)")

    del tuples_csv

    # ── Phase 2: Budget — Parquet + Polars path ───────────────────
    print(f"\n  [2/5] BUDGET — PARQUET + POLARS PATH  ({N:,} users, 5 metrics)")
    print("  " + "-" * 70)

    # Read Parquet
    t0 = time.perf_counter()
    df_pq = pl.read_parquet(pq_path)
    t_pq_read = time.perf_counter() - t0

    # Transform to engine tuples
    t0 = time.perf_counter()
    tuples_pq = df_to_engine_tuples(df_pq, K)
    t_pq_transform = time.perf_counter() - t0
    del df_pq

    # Score (same engine)
    t0 = time.perf_counter()
    results_pq = engine.analyze_arrays(tuples_pq)
    t_pq_score = time.perf_counter() - t0
    t_pq_total = t_pq_read + t_pq_transform + t_pq_score

    print(f"    Read:      {fmt_time(t_pq_read)}")
    print(f"    Transform: {fmt_time(t_pq_transform)}")
    print(f"    Score:     {fmt_time(t_pq_score)}")
    print(f"    TOTAL:     {fmt_time(t_pq_total)}")
    print(f"    Throughput: {N / t_pq_total:,.0f} users/sec (end-to-end)")

    del tuples_pq

    # ── Phase 3: Budget — Parquet streaming (single call) ─────────
    print(f"\n  [3/5] BUDGET — PARQUET STREAMING  ({N:,} users, 5 metrics)")
    print("  " + "-" * 70)

    from prefgraph._rust_backend import HAS_PARQUET_RUST
    backend = "Rust-native Parquet" if HAS_PARQUET_RUST else "PyArrow streaming"
    print(f"    Backend: {backend}")

    cost_cols = [f"price_{g}" for g in range(K)]
    action_cols = [f"qty_{g}" for g in range(K)]

    tracemalloc.start()
    t0 = time.perf_counter()
    result_df = engine.analyze_parquet(
        str(pq_path),
        user_col="user_id",
        cost_cols=cost_cols,
        action_cols=action_cols,
    )
    t_stream_total = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"    TOTAL:     {fmt_time(t_stream_total)}")
    print(f"    Throughput: {N / t_stream_total:,.0f} users/sec (end-to-end)")
    print(f"    Peak mem:  {fmt_bytes(peak_mem)}")

    # ── Phase 4: Menu — CSV and in-memory ─────────────────────────
    print(f"\n  [4/5] MENU — CSV + IN-MEMORY  ({N_MENU:,} users, SARP+WARP+HM)")
    print("  " + "-" * 70)

    # analyze_menus() always computes SARP+WARP+HM regardless of metrics list
    menu_engine = Engine(metrics=["garp"], chunk_size=CHUNK)

    # In-memory path (baseline)
    t0 = time.perf_counter()
    menu_results_mem = menu_engine.analyze_menus(menu_users)
    t_menu_mem = time.perf_counter() - t0

    # CSV path: read + reconstruct + score
    t0 = time.perf_counter()
    menu_df_read = pl.read_csv(menu_csv_path)
    t_menu_csv_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    menu_tuples_from_csv = csv_to_menu_tuples(menu_df_read)
    t_menu_csv_transform = time.perf_counter() - t0
    del menu_df_read

    t0 = time.perf_counter()
    menu_results_csv = menu_engine.analyze_menus(menu_tuples_from_csv)
    t_menu_csv_score = time.perf_counter() - t0
    t_menu_csv_total = t_menu_csv_read + t_menu_csv_transform + t_menu_csv_score

    print(f"    In-memory:    {fmt_time(t_menu_mem)}")
    print(f"    CSV (total):  {fmt_time(t_menu_csv_total)}  "
          f"(read {fmt_time(t_menu_csv_read)} + "
          f"transform {fmt_time(t_menu_csv_transform)} + "
          f"score {fmt_time(t_menu_csv_score)})")

    # ── Phase 5: Summary card ─────────────────────────────────────
    print()
    print("=" * 72)
    print("  RESULTS")
    print("=" * 72)

    # Score sanity check
    ccei_csv = np.mean([r.ccei for r in results_csv])
    ccei_pq = np.mean([r.ccei for r in results_pq])
    n_garp_csv = sum(1 for r in results_csv if r.is_garp)

    print(f"\n  Sanity check (scores should match across formats):")
    print(f"    GARP consistent: {n_garp_csv:,} / {N:,} ({n_garp_csv / N * 100:.1f}%)")
    print(f"    Mean CCEI (CSV):     {ccei_csv:.6f}")
    print(f"    Mean CCEI (Parquet): {ccei_pq:.6f}")

    # Budget comparison table
    print(f"\n  BUDGET  ({N:,} users, {METRICS}, K={K}, T=15)")
    print(f"  {'Pipeline':<25s}  {'Read':>7s}  {'Transform':>9s}  "
          f"{'Score':>7s}  {'Total':>7s}  {'File':>8s}")
    print("  " + "-" * 70)

    rows = [
        ("CSV + Polars", t_csv_read, t_csv_transform,
         t_csv_score, t_csv_total, csv_size),
        ("Parquet + Polars", t_pq_read, t_pq_transform,
         t_pq_score, t_pq_total, pq_size),
        ("Parquet (streaming)", None, None,
         None, t_stream_total, pq_size),
    ]

    for label, t_r, t_t, t_s, t_tot, fsize in rows:
        r_str = fmt_time(t_r) if t_r is not None else "—"
        t_str = fmt_time(t_t) if t_t is not None else "—"
        s_str = fmt_time(t_s) if t_s is not None else "—"
        print(f"  {label:<25s}  {r_str:>7s}  {t_str:>9s}  "
              f"{s_str:>7s}  {fmt_time(t_tot):>7s}  {fmt_bytes(fsize):>8s}")

    # Menu comparison
    print(f"\n  MENU  ({N_MENU:,} users, SARP+WARP+HM, 50 items)")
    print(f"  {'Pipeline':<25s}  {'Total':>7s}")
    print("  " + "-" * 35)
    print(f"  {'In-memory':<25s}  {fmt_time(t_menu_mem):>7s}")
    print(f"  {'CSV + reconstruct':<25s}  {fmt_time(t_menu_csv_total):>7s}")

    # Projections
    best_budget = min(t_csv_total, t_pq_total, t_stream_total)
    best_rate = N / best_budget
    print(f"\n  Best budget throughput: {best_rate:,.0f} users/sec (end-to-end)")
    print(f"\n  Projections ({CPU} cores, full 5-metric suite):")
    for label, count in [("10K users", 10_000), ("100K users", 100_000),
                         ("1M users", 1_000_000), ("10M users", 10_000_000)]:
        t = count / best_rate
        print(f"    {label:>12s}   {fmt_time(t)}")

    print()
    print(f"  Hardware: {CPU} cores")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
