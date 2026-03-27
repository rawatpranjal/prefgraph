"""Tests for the Rust-native Parquet pipeline.

Requires both pyarrow and the Rust parquet feature compiled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pd = pytest.importorskip("pandas")


def _has_rust_parquet() -> bool:
    try:
        from pyrevealed._rust_backend import HAS_PARQUET_RUST
        return HAS_PARQUET_RUST
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_rust_parquet(),
    reason="Rust parquet feature not compiled",
)


@pytest.fixture
def wide_parquet(tmp_path: Path) -> Path:
    """Create a wide-format Parquet file with 20 users, 5 obs each."""
    rng = np.random.default_rng(42)
    rows = []
    for uid in range(20):
        for t in range(5):
            rows.append({
                "user_id": f"u{uid:03d}",
                "period": t,
                "price_A": rng.uniform(0.5, 2.0),
                "price_B": rng.uniform(0.5, 2.0),
                "qty_A": rng.uniform(0.0, 10.0),
                "qty_B": rng.uniform(0.0, 10.0),
            })
    df = pd.DataFrame(rows)
    df = df.sort_values("user_id")
    path = tmp_path / "rust_test.parquet"
    df.to_parquet(path, engine="pyarrow")
    return path


@pytest.fixture
def parity_data(tmp_path: Path) -> tuple:
    """Create matching DataFrame and Parquet for parity testing."""
    rng = np.random.default_rng(99)
    rows = []
    for uid in range(30):
        for t in range(6):
            rows.append({
                "user_id": f"u{uid:03d}",
                "period": t,
                "price_A": rng.uniform(0.5, 2.0),
                "price_B": rng.uniform(0.5, 2.0),
                "price_C": rng.uniform(0.5, 2.0),
                "qty_A": rng.uniform(0.0, 10.0),
                "qty_B": rng.uniform(0.0, 10.0),
                "qty_C": rng.uniform(0.0, 10.0),
            })
    df = pd.DataFrame(rows).sort_values("user_id")
    path = tmp_path / "parity.parquet"
    df.to_parquet(path, engine="pyarrow")
    return df, path


class TestRustParquetDirect:
    def test_analyze_parquet_file_basic(self, wide_parquet: Path) -> None:
        from pyrevealed._rust_core import analyze_parquet_file

        results = analyze_parquet_file(
            str(wide_parquet),
            "user_id",
            ["price_A", "price_B"],
            ["qty_A", "qty_B"],
        )
        assert len(results) == 20
        for uid, result_dict in results:
            assert isinstance(uid, str)
            assert "is_garp" in result_dict
            assert "ccei" in result_dict
            assert isinstance(result_dict["ccei"], float)

    def test_returns_user_ids(self, wide_parquet: Path) -> None:
        from pyrevealed._rust_core import analyze_parquet_file

        results = analyze_parquet_file(
            str(wide_parquet),
            "user_id",
            ["price_A", "price_B"],
            ["qty_A", "qty_B"],
        )
        user_ids = [uid for uid, _ in results]
        assert len(set(user_ids)) == 20

    def test_small_chunk_size(self, wide_parquet: Path) -> None:
        from pyrevealed._rust_core import analyze_parquet_file

        results = analyze_parquet_file(
            str(wide_parquet),
            "user_id",
            ["price_A", "price_B"],
            ["qty_A", "qty_B"],
            chunk_size=5,
        )
        assert len(results) == 20


class TestRustParquetViaEngine:
    def test_engine_routes_to_rust(self, wide_parquet: Path) -> None:
        from pyrevealed.engine import Engine

        engine = Engine(metrics=["garp", "ccei"])
        result = engine.analyze_parquet(
            wide_parquet,
            user_col="user_id",
            cost_cols=["price_A", "price_B"],
            action_cols=["qty_A", "qty_B"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20
        assert "ccei" in result.columns

    def test_parity_rust_vs_python_pyarrow(self, parity_data: tuple) -> None:
        """Rust Parquet path must match PyArrow streaming path."""
        import pyrevealed as rp
        from pyrevealed.engine import Engine

        df, parquet_path = parity_data
        cost_cols = ["price_A", "price_B", "price_C"]
        action_cols = ["qty_A", "qty_B", "qty_C"]

        # In-memory path (baseline)
        mem_result = rp.analyze(
            df,
            user_col="user_id",
            cost_cols=cost_cols,
            action_cols=action_cols,
            metrics=["garp", "ccei"],
        )

        # Rust Parquet path
        engine = Engine(metrics=["garp", "ccei"])
        rust_result = engine.analyze_parquet(
            parquet_path,
            user_col="user_id",
            cost_cols=cost_cols,
            action_cols=action_cols,
        )

        mem_result = mem_result.sort_index()
        rust_result = rust_result.sort_index()

        assert len(mem_result) == len(rust_result)
        np.testing.assert_array_equal(
            mem_result["is_garp"].values, rust_result["is_garp"].values
        )
        np.testing.assert_allclose(
            mem_result["ccei"].values, rust_result["ccei"].values, atol=0.01
        )
