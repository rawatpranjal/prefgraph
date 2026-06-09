"""Conditional Rust backend import.

If the Rust extension (_rust_core) is installed, use it for batch analysis.
Otherwise, fall back to the pure-Python implementation.

Set PREFGRAPH_NO_RUST=1 to force the pure-Python fallback even when the compiled
extension is present. Maturin always builds _rust_core, so this env override is
the only way to exercise the HAS_RUST is False path for testing or benchmarking.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

# Pre-declare the optional Rust-backed callables so the two mutually exclusive
# assignment branches below (None fallback vs. compiled import) are seen by mypy
# as assignments to an already-declared name rather than conflicting
# redefinitions. Runtime value is still set entirely by the if/else.
_rust_analyze_batch: Optional[Callable[..., Any]]
_rust_analyze_menu_batch: Optional[Callable[..., Any]]
_rust_build_preference_graph: Optional[Callable[..., Any]]
_rust_rum_batch: Optional[Callable[..., Any]]
_rust_analyze_parquet_file: Optional[Callable[..., Any]]

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
        # The no-redef ignores below are needed because these conditional
        # import-as bindings live in a branch mutually exclusive with the
        # None fallbacks above, so there is no real redefinition at runtime.
        from prefgraph._rust_core import (  # type: ignore[no-redef]
            analyze_batch as _rust_analyze_batch,
        )
        from prefgraph._rust_core import (  # type: ignore[no-redef]
            analyze_menu_batch as _rust_analyze_menu_batch,
        )
        from prefgraph._rust_core import (  # type: ignore[no-redef]
            build_preference_graph as _rust_build_preference_graph,
        )
        from prefgraph._rust_core import (  # type: ignore[no-redef]
            rum_consistency_batch as _rust_rum_batch,
        )

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
        from prefgraph._rust_core import (  # type: ignore[no-redef]
            analyze_parquet_file as _rust_analyze_parquet_file,
        )

        HAS_PARQUET_RUST = True
    except ImportError:
        HAS_PARQUET_RUST = False
        _rust_analyze_parquet_file = None
