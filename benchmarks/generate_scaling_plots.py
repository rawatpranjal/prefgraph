#!/usr/bin/env python3
"""Generate performance scaling plots for RTD documentation.

Runs the Rust Engine at multiple scales and produces 3 plots:
1. Throughput vs number of users (linear scaling proof)
2. Time per user vs T (per-metric O(T^3) curves)
3. Memory vs users (streaming keeps it flat)

Saves to docs/_static/perf_*.png.
"""

import os
import sys
import time
import tracemalloc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# PyData color palette (matches RTD theme)
COLORS = {
    "garp": "#0173B2",
    "ccei": "#DE8F05",
    "mpi": "#029E73",
    "harp": "#D55E00",
    "all": "#CC78BC",
    "memory": "#0173B2",
}

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from prefgraph.engine import Engine

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "_static")
os.makedirs(OUT_DIR, exist_ok=True)


def gen_users(n, t_range=(20, 100), k=5, seed=42):
    rng = np.random.default_rng(seed)
    users = []
    for _ in range(n):
        t = int(rng.integers(t_range[0], t_range[1] + 1))
        p = np.ascontiguousarray(rng.random((t, k)) + 0.1, dtype=np.float64)
        q = np.ascontiguousarray(rng.random((t, k)) + 0.1, dtype=np.float64)
        users.append((p, q))
    return users


def plot_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10, frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =========================================================================
# Plot 1: Throughput vs Number of Users
# =========================================================================
def generate_throughput_plot():
    print("Plot 1: Throughput vs Users...")
    user_counts = [1000, 5000, 10000, 50000, 100000]

    configs = [
        (["garp"], "GARP only", COLORS["garp"]),
        (["garp", "ccei"], "GARP + CCEI", COLORS["ccei"]),
        (["garp", "ccei", "mpi", "harp"], "All metrics", COLORS["all"]),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    for metrics, label, color in configs:
        engine = Engine(metrics=metrics, chunk_size=50000)
        throughputs = []

        for n in user_counts:
            users = gen_users(n)
            # Warmup
            if n == user_counts[0]:
                engine.analyze_arrays(gen_users(50))

            t0 = time.perf_counter()
            engine.analyze_arrays(users)
            elapsed = time.perf_counter() - t0
            tp = n / elapsed
            throughputs.append(tp)
            print(f"  {label}: {n:>7,} users -> {tp:,.0f}/s")

        ax.plot(user_counts, throughputs, "o-", color=color, label=label,
                linewidth=2, markersize=6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    plot_style(ax, "Throughput scales linearly with users",
               "Number of users", "Users / second")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "perf_throughput.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =========================================================================
# Plot 2: Time per User vs T (observations per user)
# =========================================================================
def generate_per_user_plot():
    print("Plot 2: Per-user time vs T...")
    t_values = [20, 50, 100, 200, 500]
    n_users = 2000  # Enough to average out

    metric_configs = [
        (["garp"], "garp", "GARP", COLORS["garp"]),
        (["garp", "ccei"], "ccei", "CCEI", COLORS["ccei"]),
        (["garp", "mpi"], "mpi", "MPI", COLORS["mpi"]),
        (["garp", "harp"], "harp", "HARP", COLORS["harp"]),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Baseline: GARP-only timings
    garp_times = {}

    for metrics, key, label, color in metric_configs:
        engine = Engine(metrics=metrics, chunk_size=50000)
        us_per_user = []

        for t_val in t_values:
            users = gen_users(n_users, t_range=(t_val, t_val))
            engine.analyze_arrays(gen_users(20, t_range=(t_val, t_val)))  # warmup

            t0 = time.perf_counter()
            engine.analyze_arrays(users)
            elapsed = time.perf_counter() - t0
            us = elapsed / n_users * 1e6
            us_per_user.append(us)

            if key == "garp":
                garp_times[t_val] = us

            print(f"  {label} T={t_val}: {us:.0f} us/user")

        ax.plot(t_values, us_per_user, "o-", color=color, label=label,
                linewidth=2, markersize=6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    plot_style(ax, "Per-user compute time by metric",
               "Observations per user (T)", "Microseconds per user")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "perf_per_user.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =========================================================================
# Plot 3: Memory vs Users (streaming)
# =========================================================================
def generate_memory_plot():
    print("Plot 3: Memory vs Users...")
    user_counts = [10000, 50000, 100000, 500000]
    chunk_size = 50000

    engine = Engine(metrics=["garp", "ccei", "mpi", "harp"], chunk_size=chunk_size)
    peak_mems = []

    for n in user_counts:
        tracemalloc.start()

        # Stream: generate and process in chunks
        rng = np.random.default_rng(42)
        all_results = []
        processed = 0
        while processed < n:
            batch = min(chunk_size, n - processed)
            chunk = []
            for _ in range(batch):
                t = int(rng.integers(20, 101))
                p = np.ascontiguousarray(rng.random((t, 5)) + 0.1, dtype=np.float64)
                q = np.ascontiguousarray(rng.random((t, 5)) + 0.1, dtype=np.float64)
                chunk.append((p, q))
            results = engine.analyze_arrays(chunk)
            all_results.extend(results)
            processed += batch
            del chunk, results

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / 1e6
        peak_mems.append(peak_mb)
        print(f"  {n:>7,} users: peak {peak_mb:.0f} MB")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(user_counts, peak_mems, "o-", color=COLORS["memory"],
            linewidth=2, markersize=8)

    # Reference line: what it would be without streaming
    naive_mems = [n * 60 * 5 * 8 * 2 / 1e6 for n in user_counts]  # T*K*8bytes*2arrays
    ax.plot(user_counts, naive_mems, "--", color="#999999", linewidth=1,
            label="Without streaming (all in RAM)", alpha=0.7)
    ax.plot(user_counts, peak_mems, "o-", color=COLORS["memory"],
            linewidth=2, markersize=8, label="With streaming (chunks of 50K)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    plot_style(ax, "Memory stays bounded under streaming",
               "Number of users", "Peak memory (MB)")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "perf_memory.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    cpu = os.cpu_count() or 1
    print("=" * 60)
    print(f" Generating scaling plots for RTD ({cpu} cores)")
    print("=" * 60)

    generate_throughput_plot()
    print()
    generate_per_user_plot()
    print()
    generate_memory_plot()

    print()
    print("=" * 60)
    print(" All plots saved to docs/_static/perf_*.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
