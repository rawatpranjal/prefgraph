"""Tests for rich statsmodels-style summary output."""

import numpy as np
import pytest

from pyrevealed import BehaviorLog, BehavioralSummary


@pytest.fixture
def consistent_log():
    """Create a GARP-consistent BehaviorLog."""
    prices = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
    quantities = np.array([[2.0, 1.0], [1.0, 2.0], [1.5, 1.5]])
    return BehaviorLog(prices, quantities, user_id="test_consistent")


@pytest.fixture
def inconsistent_log():
    """Create a GARP-inconsistent BehaviorLog (random data)."""
    np.random.seed(42)
    return BehaviorLog(
        np.random.rand(20, 5),
        np.random.rand(20, 5),
        user_id="test_inconsistent",
    )


class TestBehavioralSummaryRich:
    """Test that the rich summary output contains expected sections."""

    def test_contains_header_block(self, consistent_log):
        s = BehavioralSummary.from_log(consistent_log)
        text = s.summary()
        assert "BEHAVIORAL SUMMARY" in text
        assert "User ID" in text
        assert "test_consistent" in text
        assert "No. Observations" in text
        assert "No. Goods" in text

    def test_contains_input_data_section(self, consistent_log):
        s = BehavioralSummary.from_log(consistent_log)
        text = s.summary()
        assert "Input Data:" in text
        assert "Prices" in text
        assert "Quantities" in text
        assert "Expenditure" in text

    def test_contains_graph_section(self, consistent_log):
        s = BehavioralSummary.from_log(consistent_log)
        text = s.summary()
        assert "Revealed Preference Graph:" in text
        assert "R  (direct" in text
        assert "P  (strict" in text
        assert "R* (transitive" in text
        assert "edges" in text

    def test_contains_consistency_section(self, consistent_log):
        s = BehavioralSummary.from_log(consistent_log)
        text = s.summary()
        assert "Consistency Tests:" in text
        assert "GARP" in text
        assert "[+] PASS" in text

    def test_contains_goodness_of_fit(self, consistent_log):
        s = BehavioralSummary.from_log(consistent_log)
        text = s.summary()
        assert "Goodness-of-Fit:" in text
        assert "Afriat Efficiency (AEI)" in text
        assert "Binary search iterations" in text
        assert "Budget waste" in text
        assert "Money Pump Index (MPI)" in text
        assert "Total expenditure" in text

    def test_inconsistent_shows_violations(self, inconsistent_log):
        s = BehavioralSummary.from_log(inconsistent_log)
        text = s.summary()
        assert "[-] FAIL" in text
        assert "cycle" in text.lower()
        assert "Houtman-Maks" in text
        assert "Observations removed" in text

    def test_inconsistent_interpretation_details(self, inconsistent_log):
        s = BehavioralSummary.from_log(inconsistent_log)
        text = s.summary()
        assert "budget waste" in text.lower() or "Budget waste" in text

    def test_data_stats_populated(self, consistent_log):
        s = BehavioralSummary.from_log(consistent_log)
        assert s.price_stats is not None
        assert s.quantity_stats is not None
        assert s.expenditure_stats is not None
        assert "mean" in s.price_stats
        assert "std" in s.price_stats
        assert "min" in s.price_stats
        assert "max" in s.price_stats

    def test_graph_density_populated(self, consistent_log):
        s = BehavioralSummary.from_log(consistent_log)
        assert s.r_density is not None
        assert s.p_density is not None
        assert s.r_star_density is not None
        assert 0 <= s.r_density <= 1
        assert 0 <= s.p_density <= 1
        assert 0 <= s.r_star_density <= 1

    def test_user_id_populated(self, consistent_log):
        s = BehavioralSummary.from_log(consistent_log)
        assert s.user_id == "test_consistent"

    def test_backward_compat_positional_construction(self, consistent_log):
        """Existing code that constructs BehavioralSummary positionally
        should still work (new fields have defaults)."""
        from pyrevealed.algorithms.garp import validate_consistency
        from pyrevealed.algorithms.aei import compute_integrity_score
        from pyrevealed.algorithms.mpi import compute_confusion_metric

        garp = validate_consistency(consistent_log)
        aei = compute_integrity_score(consistent_log)
        mpi = compute_confusion_metric(consistent_log)

        # Positional construction with only the required fields
        s = BehavioralSummary(
            garp_result=garp,
            warp_result=None,
            sarp_result=None,
            aei_result=aei,
            mpi_result=mpi,
            houtman_maks_result=None,
            optimal_efficiency_result=None,
            num_observations=3,
            num_goods=2,
            computation_time_ms=1.0,
        )
        # Should still produce output (with N/A for missing sections)
        text = s.summary()
        assert "BEHAVIORAL SUMMARY" in text

    def test_width_consistency(self, inconsistent_log):
        """All separator lines should be 70 chars wide."""
        s = BehavioralSummary.from_log(inconsistent_log)
        text = s.summary()
        for line in text.split("\n"):
            if line.startswith("===") or line.startswith("---"):
                assert len(line) == 70, f"Line width {len(line)}: {line!r}"
