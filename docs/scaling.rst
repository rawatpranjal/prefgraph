Scaling (Legacy)
================

.. deprecated:: 0.3.0
   This page described the Numba/Python-only performance characteristics.
   PyRevealed now uses a Rust compute engine (``rpt-core``) with Rayon
   parallelism. See :doc:`performance` for current benchmarks and
   :doc:`algorithms` for complexity analysis.

The Rust engine replaces the Numba JIT approach with:

- **O(T²) GARP** via SCC arc-scan (Talla Nobibon et al. 2015) — no Floyd-Warshall
- **O(T² log T) CCEI** via discrete binary search with O(T²) GARP per step
- **Rayon thread-pool** parallelism — one thread per user, scratchpad reuse
- **HiGHS LP/MILP** for utility recovery, VEI, and exact Houtman-Maks

Throughput: ~49,000 users/sec for GARP-only, ~2,400/sec for GARP+CCEI
(Apple M-series, 11 cores, T=20-100 observations per user).

Memory: O(T²) per thread (boolean matrices). Streaming chunks bound peak usage
regardless of total users.

See :doc:`performance` for detailed benchmarks.
