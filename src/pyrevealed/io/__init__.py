"""I/O utilities for large-scale revealed preference analysis.

Provides streaming Parquet readers and preparation utilities for datasets
that don't fit in memory.
"""

from __future__ import annotations

from pyrevealed.io.parquet import ParquetUserIterator, prepare_parquet

__all__ = ["ParquetUserIterator", "prepare_parquet"]
