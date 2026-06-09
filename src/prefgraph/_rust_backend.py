"""Conditional Rust backend import.

If the Rust extension (_rust_core) is installed, use it for batch analysis.
Otherwise, fall back to the pure-Python implementation.

Set PREFGRAPH_NO_RUST=1 to force the pure-Python fallback even when the compiled
extension is present. Maturin always builds _rust_core, so this env override is
the only way to exercise the HAS_RUST is False path for testing or benchmarking.
"""

import os

_FORCE_NO_RUST = os.environ.get("PREFGRAPH_NO_RUST", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

if _FORCE_NO_RUST:
    HAS_RUST = False
    _rust_analyze_batch = None
    _rust_analyze_menu_batch = None
    _rust_build_preference_graph = None
    _rust_rum_batch = None
else:
    try:
        from prefgraph._rust_core import analyze_batch as _rust_analyze_batch
        from prefgraph._rust_core import analyze_menu_batch as _rust_analyze_menu_batch
        from prefgraph._rust_core import (
            build_preference_graph as _rust_build_preference_graph,
        )
        from prefgraph._rust_core import rum_consistency_batch as _rust_rum_batch

        HAS_RUST = True
    except ImportError:
        HAS_RUST = False
        _rust_analyze_batch = None
        _rust_analyze_menu_batch = None
        _rust_build_preference_graph = None
        _rust_rum_batch = None

# Parquet support (compiled with --features parquet)
if _FORCE_NO_RUST:
    HAS_PARQUET_RUST = False
    _rust_analyze_parquet_file = None
else:
    try:
        from prefgraph._rust_core import (
            analyze_parquet_file as _rust_analyze_parquet_file,
        )

        HAS_PARQUET_RUST = True
    except ImportError:
        HAS_PARQUET_RUST = False
        _rust_analyze_parquet_file = None
