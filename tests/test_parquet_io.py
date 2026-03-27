"""Tests for Parquet streaming I/O.

Requires pyarrow: pip install pyrevealed[parquet]
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pd = pytest.importorskip("pandas")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wide_parquet(tmp_path: Path) -> Path:
    """Create a small wide-format Parquet file with 10 users, 5 obs each."""
    rng = np.random.default_rng(42)
    rows = []
    for uid in range(10):
        for t in range(5):
            rows.append({
                "user_id": f"u{uid}",
                "period": t,
                "price_A": rng.uniform(0.5, 2.0),
                "price_B": rng.uniform(0.5, 2.0),
                "qty_A": rng.uniform(0.0, 10.0),
                "qty_B": rng.uniform(0.0, 10.0),
            })
    df = pd.DataFrame(rows)
    path = tmp_path / "wide_test.parquet"
    df.to_parquet(path, engine="pyarrow", row_group_size=20)  # 4 obs/user * 5 = small groups
    return path


@pytest.fixture
def long_parquet(tmp_path: Path) -> Path:
    """Create a small long-format Parquet file with 5 users."""
    rng = np.random.default_rng(42)
    rows = []
    items = ["A", "B", "C"]
    for uid in range(5):
        for t in range(4):
            for item in items:
                rows.append({
                    "user_id": f"u{uid}",
                    "time": t,
                    "item": item,
                    "price": rng.uniform(0.5, 3.0),
                    "quantity": rng.uniform(0.0, 5.0),
                })
    df = pd.DataFrame(rows)
    path = tmp_path / "long_test.parquet"
    df.to_parquet(path, engine="pyarrow", row_group_size=20)
    return path


@pytest.fixture
def wide_df_and_parquet(tmp_path: Path) -> tuple:
    """Create matching DataFrame and Parquet for parity testing."""
    rng = np.random.default_rng(123)
    rows = []
    for uid in range(20):
        for t in range(5):
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
    df = pd.DataFrame(rows)
    path = tmp_path / "parity_test.parquet"
    # Sort by user for consistent results
    df = df.sort_values("user_id")
    df.to_parquet(path, engine="pyarrow", row_group_size=25)
    return df, path


# ---------------------------------------------------------------------------
# ParquetUserIterator tests
# ---------------------------------------------------------------------------


class TestParquetUserIterator:
    def test_wide_format_basic(self, wide_parquet: Path) -> None:
        from pyrevealed.io.parquet import ParquetUserIterator

        it = ParquetUserIterator(
            wide_parquet,
            user_col="user_id",
            cost_cols=["price_A", "price_B"],
            action_cols=["qty_A", "qty_B"],
            chunk_size=5,
        )
        all_ids = []
        all_tuples = []
        for user_ids, tuples in it:
            assert len(user_ids) == len(tuples)
            for prices, quantities in tuples:
                assert prices.ndim == 2
                assert quantities.ndim == 2
                assert prices.shape[1] == 2  # 2 items
                assert quantities.shape[1] == 2
                assert prices.dtype == np.float64
            all_ids.extend(user_ids)
            all_tuples.extend(tuples)

        # Should have all 10 users
        assert len(set(all_ids)) == 10
        assert len(all_tuples) == 10

    def test_wide_format_user_obs_count(self, wide_parquet: Path) -> None:
        from pyrevealed.io.parquet import ParquetUserIterator

        it = ParquetUserIterator(
            wide_parquet,
            user_col="user_id",
            cost_cols=["price_A", "price_B"],
            action_cols=["qty_A", "qty_B"],
            chunk_size=100,
        )
        for user_ids, tuples in it:
            for prices, _ in tuples:
                assert prices.shape[0] == 5  # 5 observations per user

    def test_long_format_basic(self, long_parquet: Path) -> None:
        from pyrevealed.io.parquet import ParquetUserIterator

        it = ParquetUserIterator(
            long_parquet,
            user_col="user_id",
            item_col="item",
            cost_col="price",
            action_col="quantity",
            time_col="time",
            chunk_size=10,
        )
        all_ids = []
        for user_ids, tuples in it:
            for prices, quantities in tuples:
                assert prices.ndim == 2
                assert prices.shape == (4, 3)  # 4 times × 3 items
                assert quantities.shape == (4, 3)
            all_ids.extend(user_ids)

        assert len(set(all_ids)) == 5

    def test_chunking(self, wide_parquet: Path) -> None:
        from pyrevealed.io.parquet import ParquetUserIterator

        it = ParquetUserIterator(
            wide_parquet,
            user_col="user_id",
            cost_cols=["price_A", "price_B"],
            action_cols=["qty_A", "qty_B"],
            chunk_size=3,
        )
        chunks = list(it)
        # 10 users / chunk_size 3 = 4 chunks (3+3+3+1)
        assert len(chunks) >= 2
        total = sum(len(ids) for ids, _ in chunks)
        assert total == 10

    def test_format_conflict_raises(self, wide_parquet: Path) -> None:
        from pyrevealed.io.parquet import ParquetUserIterator

        with pytest.raises(ValueError, match="not both"):
            ParquetUserIterator(
                wide_parquet,
                cost_cols=["a"], action_cols=["b"],
                item_col="c",
            )

    def test_missing_params_raises(self, wide_parquet: Path) -> None:
        from pyrevealed.io.parquet import ParquetUserIterator

        with pytest.raises(ValueError, match="cost_cols.*action_cols"):
            ParquetUserIterator(wide_parquet)


# ---------------------------------------------------------------------------
# prepare_parquet tests
# ---------------------------------------------------------------------------


class TestPrepareParquet:
    def test_sort_by_user(self, tmp_path: Path) -> None:
        from pyrevealed.io.parquet import prepare_parquet

        # Create unsorted data
        rng = np.random.default_rng(42)
        rows = []
        for _ in range(100):
            rows.append({
                "user_id": f"u{rng.integers(0, 10)}",
                "price_A": rng.uniform(0.5, 2.0),
                "qty_A": rng.uniform(0.0, 10.0),
            })
        df = pd.DataFrame(rows)
        input_path = tmp_path / "unsorted.parquet"
        df.to_parquet(input_path)

        output_path = tmp_path / "sorted.parquet"
        stats = prepare_parquet(input_path, output_path)

        assert stats["n_rows"] == 100
        assert stats["n_users"] == 10
        assert Path(stats["output_path"]).exists()

        # Verify sorted
        result = pq.read_table(output_path)
        user_ids = result.column("user_id").to_pylist()
        assert user_ids == sorted(user_ids)

    def test_csv_input(self, tmp_path: Path) -> None:
        from pyrevealed.io.parquet import prepare_parquet

        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            "user_id": ["b", "a", "b", "a"],
            "price": [1.0, 2.0, 3.0, 4.0],
            "qty": [5.0, 6.0, 7.0, 8.0],
        })
        df.to_csv(csv_path, index=False)

        output_path = tmp_path / "out.parquet"
        stats = prepare_parquet(csv_path, output_path)
        assert stats["n_rows"] == 4
        assert stats["n_users"] == 2


# ---------------------------------------------------------------------------
# Engine.analyze_parquet tests
# ---------------------------------------------------------------------------


class TestEngineAnalyzeParquet:
    def test_wide_format_returns_dataframe(self, wide_parquet: Path) -> None:
        from pyrevealed.engine import Engine

        engine = Engine(metrics=["garp", "ccei"])
        result = engine.analyze_parquet(
            wide_parquet,
            user_col="user_id",
            cost_cols=["price_A", "price_B"],
            action_cols=["qty_A", "qty_B"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert "is_garp" in result.columns
        assert "ccei" in result.columns

    def test_parity_with_in_memory(self, wide_df_and_parquet: tuple) -> None:
        """Parquet streaming results must match in-memory analyze() results."""
        import pyrevealed as rp
        from pyrevealed.engine import Engine

        df, parquet_path = wide_df_and_parquet
        cost_cols = ["price_A", "price_B", "price_C"]
        action_cols = ["qty_A", "qty_B", "qty_C"]

        # In-memory path
        mem_result = rp.analyze(
            df,
            user_col="user_id",
            cost_cols=cost_cols,
            action_cols=action_cols,
            metrics=["garp", "ccei"],
        )

        # Parquet streaming path
        engine = Engine(metrics=["garp", "ccei"])
        pq_result = engine.analyze_parquet(
            parquet_path,
            user_col="user_id",
            cost_cols=cost_cols,
            action_cols=action_cols,
        )

        # Sort both by index for comparison
        mem_result = mem_result.sort_index()
        pq_result = pq_result.sort_index()

        assert len(mem_result) == len(pq_result)
        np.testing.assert_array_equal(mem_result["is_garp"].values, pq_result["is_garp"].values)
        np.testing.assert_allclose(mem_result["ccei"].values, pq_result["ccei"].values, atol=0.01)

    def test_output_to_parquet_file(self, wide_parquet: Path, tmp_path: Path) -> None:
        from pyrevealed.engine import Engine

        engine = Engine(metrics=["garp", "ccei"])
        output_path = str(tmp_path / "results.parquet")
        result = engine.analyze_parquet(
            wide_parquet,
            user_col="user_id",
            cost_cols=["price_A", "price_B"],
            action_cols=["qty_A", "qty_B"],
            output_path=output_path,
        )
        assert result == output_path
        assert Path(output_path).exists()

        # Read back and verify
        result_df = pd.read_parquet(output_path)
        assert len(result_df) == 10
        assert "ccei" in result_df.columns


# ---------------------------------------------------------------------------
# analyze() Parquet path detection
# ---------------------------------------------------------------------------


class TestAnalyzeParquetPath:
    def test_string_path(self, wide_parquet: Path) -> None:
        import pyrevealed as rp

        result = rp.analyze(
            str(wide_parquet),
            user_col="user_id",
            cost_cols=["price_A", "price_B"],
            action_cols=["qty_A", "qty_B"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10

    def test_path_object(self, wide_parquet: Path) -> None:
        import pyrevealed as rp

        result = rp.analyze(
            wide_parquet,
            user_col="user_id",
            cost_cols=["price_A", "price_B"],
            action_cols=["qty_A", "qty_B"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
