"""EVAL: native coverage for revpref-style budget axiom APIs."""

from __future__ import annotations

import numpy as np
import pytest

from prefgraph import (
    BehaviorLog,
    MPIBoundsResult,
    check_garp,
    check_sarp,
    check_warp,
    compute_aei,
    compute_bronars_power_fast,
    compute_ccei,
    compute_mpi,
    compute_mpi_bounds,
)


def _simple_two_cycle_log() -> BehaviorLog:
    """Two-observation strict preference reversal with known thresholds."""
    return BehaviorLog(
        cost_vectors=np.array([[2.0, 1.0], [1.0, 2.0]]),
        action_vectors=np.array([[3.0, 2.0], [2.0, 3.0]]),
    )


def test_budget_axioms_accept_efficiency_level():
    """WARP/SARP/GARP should match exact defaults and relax at lower e."""
    log = _simple_two_cycle_log()

    assert check_warp(log).is_consistent == check_warp(log, efficiency=1.0).is_consistent
    assert check_sarp(log).is_consistent == check_sarp(log, efficiency=1.0).is_consistent
    assert check_garp(log).is_consistent == check_garp(log, efficiency=1.0).is_consistent

    assert not check_warp(log).is_consistent
    assert not check_sarp(log).is_consistent
    assert not check_garp(log).is_consistent

    assert check_warp(log, efficiency=0.8).is_consistent
    assert check_sarp(log, efficiency=0.8).is_consistent
    assert check_garp(log, efficiency=0.8).is_consistent

    with pytest.raises(ValueError, match="efficiency"):
        check_garp(log, efficiency=1.1)


def test_sarp_allows_repeated_identical_chosen_bundle():
    """SARP rejects cycles across distinct bundles, not repeated same bundles."""
    log = BehaviorLog(
        cost_vectors=np.array([[1.0, 1.0], [2.0, 2.0]]),
        action_vectors=np.array([[1.0, 1.0], [1.0, 1.0]]),
    )

    assert check_sarp(log).is_consistent


def test_ccei_can_target_warp_sarp_or_garp():
    """CCEI/AEI should route to the selected budget axiom."""
    log = _simple_two_cycle_log()

    for axiom in ("warp", "sarp", "garp"):
        result = compute_aei(log, axiom=axiom)
        ccei = compute_ccei(log, axiom=axiom)

        assert result.axiom == axiom
        assert ccei.axiom == axiom
        assert np.isclose(result.efficiency_index, 0.875)
        assert np.isclose(ccei.efficiency_index, result.efficiency_index)

    with pytest.raises(ValueError, match="axiom"):
        compute_aei(log, axiom="harp")


def test_bronars_power_routes_selected_axiom_and_efficiency():
    """Bronars simulations should expose revpref-style model/e arguments."""
    log = _simple_two_cycle_log()

    result = compute_bronars_power_fast(
        log,
        n_simulations=5,
        random_seed=123,
        axiom="warp",
        efficiency=0.8,
    )

    assert result.axiom == "warp"
    assert result.efficiency == 0.8
    assert result.n_simulations == 5
    assert 0 <= result.n_violations <= 5
    assert 0.0 <= result.power_index <= 1.0
    assert result.to_dict()["axiom"] == "warp"

    with pytest.raises(ValueError, match="axiom"):
        compute_bronars_power_fast(log, n_simulations=1, axiom="bad")


def test_mpi_bounds_return_minimum_and_maximum_cycle_ratios():
    """MPI bounds should expose the min/max cycle-ratio quantities."""
    log = _simple_two_cycle_log()

    bounds = compute_mpi_bounds(log, convergence_tolerance=1e-12)
    legacy = compute_mpi(log)

    assert isinstance(bounds, MPIBoundsResult)
    assert np.isclose(bounds.minimum_mpi, 0.125, atol=1e-8)
    assert np.isclose(bounds.maximum_mpi, 0.125, atol=1e-8)
    assert np.isclose(bounds.maximum_mpi, legacy.mpi_value, atol=1e-8)


def test_mpi_bounds_zero_for_consistent_data():
    """Consistent budget data should have no money-pump bounds."""
    log = BehaviorLog(
        cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
        action_vectors=np.array([[4.0, 1.0], [1.0, 4.0]]),
    )

    bounds = compute_mpi_bounds(log)

    assert bounds.is_consistent
    assert bounds.minimum_mpi == 0.0
    assert bounds.maximum_mpi == 0.0
