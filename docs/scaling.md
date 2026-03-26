# Scaling (Legacy)

> **Deprecated**: This page described the Numba/Python-only performance model.
> PyRevealed now uses a Rust engine (`rpt-core`) with Rayon parallelism.
> See `performance.rst` for current benchmarks and `algorithms.rst` for complexity.

## Current Architecture

| Algorithm | Complexity | Engine |
|-----------|-----------|--------|
| GARP | O(T²) | Rust SCC (Talla Nobibon et al. 2015) |
| CCEI | O(T² log T) | Rust binary search + O(T²) GARP |
| MPI | O(T³) | Rust Karp's algorithm |
| HARP | O(T³) | Rust max-product Floyd-Warshall |
| HM | NP-hard | Rust greedy FVS / ILP (HiGHS) |
| VEI | O(T²) constraints | Rust LP (HiGHS) |

Throughput: ~49,000 users/sec (GARP-only), ~2,400/sec (GARP+CCEI).
