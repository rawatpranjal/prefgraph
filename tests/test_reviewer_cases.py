"""Regression tests from external reviewer hand-crafted cases.

Covers budget, menu, stochastic, and risk APIs with exact expected values.
These cases were designed to probe nuanced edge cases:
- Budget: consistent, weak/strict reversal, equal-budget ties
- Menu: direct reversal, 3-cycle, nested transitive
- Stochastic: MST-not-SST mixture, regularity violation, cyclic RUM
- Risk: neutral, averse, seeking, inconsistent
"""

from __future__ import annotations

import numpy as np
import pytest

from prefgraph.core.session import (
    BehaviorLog,
    MenuChoiceLog,
    RiskChoiceLog,
    StochasticChoiceLog,
)


# =============================================================================
# Budget cases
# =============================================================================


class TestBudgetReviewerCases:
    def test_consistent_substitution(self):
        """Standard substitution: buy more of the cheaper good."""
        p = np.array([[1.0, 2.0], [2.0, 1.0]])
        q = np.array([[4.0, 1.0], [1.0, 4.0]])
        log = BehaviorLog(prices=p, quantities=q)

        from prefgraph import validate_consistency, compute_integrity_score, compute_confusion_metric

        garp = validate_consistency(log)
        assert garp.is_consistent is True
        assert len(garp.violations) == 0

        aei = compute_integrity_score(log)
        assert aei.efficiency_index == pytest.approx(1.0, abs=1e-6)

        mpi = compute_confusion_metric(log)
        assert mpi.mpi_value == pytest.approx(0.0, abs=1e-6)

    def test_weak_strict_reversal(self):
        """Weak tie one way, strict reversal the other. GARP violation.

        p0*q0=3 >= p0*q1=3 (weak R), p1*q1=5 > p1*q0=4 (strict P).
        So 0 R 1 and 1 P 0, forming a violation cycle.

        This is also a regression test for the Rust r_star bug where
        violations came back empty despite is_consistent=False.
        """
        p = np.array([[1.0, 1.0], [1.0, 2.0]])
        q = np.array([[2.0, 1.0], [1.0, 2.0]])
        log = BehaviorLog(prices=p, quantities=q)

        from prefgraph import validate_consistency, compute_integrity_score, compute_confusion_metric

        garp = validate_consistency(log)
        assert garp.is_consistent is False
        assert len(garp.violations) > 0  # Regression: was [] before fix

        aei = compute_integrity_score(log)
        assert 0 < aei.efficiency_index < 1

        mpi = compute_confusion_metric(log)
        assert mpi.mpi_value > 0

    def test_equal_budget_ties(self):
        """Same prices, different equal-cost bundles. No strict preference."""
        p = np.array([[1.0, 1.0], [1.0, 1.0]])
        q = np.array([[2.0, 1.0], [1.0, 2.0]])
        log = BehaviorLog(prices=p, quantities=q)

        from prefgraph import validate_consistency

        garp = validate_consistency(log)
        assert garp.is_consistent is True
        assert len(garp.violations) == 0


# =============================================================================
# Menu cases
# =============================================================================


class TestMenuReviewerCases:
    def test_direct_reversal(self):
        """Same binary menu, opposite choices. WARP and SARP fail."""
        from prefgraph.algorithms.abstract_choice import validate_menu_warp, validate_menu_sarp

        log = MenuChoiceLog(
            menus=[frozenset({0, 1}), frozenset({0, 1})],
            choices=[0, 1],
        )
        warp = validate_menu_warp(log)
        sarp = validate_menu_sarp(log)
        assert warp.is_consistent is False
        assert sarp.is_consistent is False

    def test_three_cycle(self):
        """3-cycle: 0>1, 1>2, 2>0. WARP passes, SARP fails."""
        from prefgraph.algorithms.abstract_choice import validate_menu_warp, validate_menu_sarp

        log = MenuChoiceLog(
            menus=[frozenset({0, 1}), frozenset({1, 2}), frozenset({0, 2})],
            choices=[0, 1, 2],
        )
        warp = validate_menu_warp(log)
        sarp = validate_menu_sarp(log)
        assert warp.is_consistent is True
        assert sarp.is_consistent is False

    def test_nested_transitive(self):
        """Transitive nested menus. Recovers preference order 0 > 1 > 2."""
        from prefgraph.algorithms.abstract_choice import (
            validate_menu_warp,
            validate_menu_sarp,
            fit_menu_preferences,
        )

        log = MenuChoiceLog(
            menus=[frozenset({0, 1, 2}), frozenset({1, 2})],
            choices=[0, 1],
        )
        warp = validate_menu_warp(log)
        sarp = validate_menu_sarp(log)
        assert warp.is_consistent is True
        assert sarp.is_consistent is True

        prefs = fit_menu_preferences(log)
        assert prefs.success is True
        assert prefs.preference_order == [0, 1, 2]


# =============================================================================
# Stochastic choice cases
# =============================================================================


class TestStochasticReviewerCases:
    def test_mst_not_sst_mixture(self):
        """60/40 mixture of two orderings satisfies MST but not SST.

        60% ordering (0,1,2) + 40% ordering (1,2,0) gives:
        P(0|{0,1})=0.6, P(1|{1,2})=1.0, P(0|{0,2})=0.6
        """
        from prefgraph.contrib.stochastic import test_stochastic_transitivity, test_rum_consistency

        log = StochasticChoiceLog(
            menus=[frozenset({0, 1}), frozenset({1, 2}), frozenset({0, 2})],
            choice_frequencies=[
                {0: 60, 1: 40},
                {1: 100, 2: 0},
                {0: 60, 2: 40},
            ],
        )

        trans = test_stochastic_transitivity(log)
        assert trans.satisfies_mst is True
        assert trans.satisfies_sst is False
        assert trans.strongest_satisfied == "MST"

        rum = test_rum_consistency(log)
        assert rum.is_rum_consistent is True
        dist = rum.rationalizing_distribution
        assert dist is not None
        assert dist[(0, 1, 2)] == pytest.approx(0.6, abs=0.05)
        assert dist[(1, 2, 0)] == pytest.approx(0.4, abs=0.05)

    def test_regularity_violation(self):
        """Decoy effect: P(0|{0,1})=0.55 but P(0|{0,1,2})=0.70.

        Adding item 2 increases the choice probability of item 0,
        which violates regularity and rules out any RUM.
        """
        from prefgraph.contrib.stochastic import test_regularity, test_rum_consistency

        log = StochasticChoiceLog(
            menus=[frozenset({0, 1}), frozenset({0, 1, 2})],
            choice_frequencies=[
                {0: 55, 1: 45},
                {0: 70, 1: 20, 2: 10},
            ],
        )

        reg = test_regularity(log)
        assert reg.satisfies_regularity is False
        assert reg.worst_violation is not None
        assert reg.worst_violation.magnitude == pytest.approx(0.15, abs=0.02)

        rum = test_rum_consistency(log)
        assert rum.is_rum_consistent is False
        assert rum.distance_to_rum == pytest.approx(0.30, abs=0.05)

    def test_cyclic_majorities_rum_consistent(self):
        """Cyclic pairwise majorities can still be RUM-consistent.

        0 beats 1 with 0.60, 1 beats 2 with 0.60, 2 beats 0 with 0.60.
        This fails all levels of stochastic transitivity, yet a valid
        random utility model exists. The rationalizing distribution uses
        four orderings.
        """
        from prefgraph.contrib.stochastic import test_stochastic_transitivity, test_rum_consistency

        log = StochasticChoiceLog(
            menus=[frozenset({0, 1}), frozenset({1, 2}), frozenset({0, 2})],
            choice_frequencies=[
                {0: 60, 1: 40},
                {1: 60, 2: 40},
                {2: 60, 0: 40},
            ],
        )

        trans = test_stochastic_transitivity(log)
        assert trans.satisfies_wst is False
        assert trans.strongest_satisfied == "None"

        rum = test_rum_consistency(log)
        assert rum.is_rum_consistent is True
        assert rum.rationalizing_distribution is not None


# =============================================================================
# Risk cases
# =============================================================================


class TestRiskReviewerCases:
    """CRRA risk profile tests with binary 50/50 lotteries."""

    RISKY_OUTCOMES = np.array([[70.0, 0.0], [90.0, 0.0], [110.0, 0.0]])
    RISKY_PROBS = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

    def test_risk_neutral(self):
        """Choose risky when EV > safe, safe otherwise. Near-zero rho."""
        from prefgraph.contrib.risk import compute_risk_profile

        log = RiskChoiceLog(
            safe_values=np.array([25.0, 50.0, 40.0]),
            risky_outcomes=self.RISKY_OUTCOMES,
            risky_probabilities=self.RISKY_PROBS,
            choices=np.array([True, False, True]),
        )
        result = compute_risk_profile(log)
        assert result.risk_category == "risk_neutral"
        assert abs(result.risk_aversion_coefficient) < 0.5
        assert result.consistency_score == 1.0

    def test_risk_averse(self):
        """Always choose safe when safe > EV. Positive rho."""
        from prefgraph.contrib.risk import compute_risk_profile

        log = RiskChoiceLog(
            safe_values=np.array([40.0, 50.0, 60.0]),
            risky_outcomes=self.RISKY_OUTCOMES,
            risky_probabilities=self.RISKY_PROBS,
            choices=np.array([False, False, False]),
        )
        result = compute_risk_profile(log)
        assert result.risk_category == "risk_averse"
        assert result.risk_aversion_coefficient > 0
        assert result.consistency_score == 1.0

    def test_risk_seeking(self):
        """Always choose risky even when safe > EV. Negative rho."""
        from prefgraph.contrib.risk import compute_risk_profile

        log = RiskChoiceLog(
            safe_values=np.array([40.0, 50.0, 60.0]),
            risky_outcomes=self.RISKY_OUTCOMES,
            risky_probabilities=self.RISKY_PROBS,
            choices=np.array([True, True, True]),
        )
        result = compute_risk_profile(log)
        assert result.risk_category == "risk_seeking"
        assert result.risk_aversion_coefficient < 0
        assert result.consistency_score == 1.0

    def test_inconsistent(self):
        """Mixed choices that do not fit any single CRRA parameter."""
        from prefgraph.contrib.risk import compute_risk_profile

        log = RiskChoiceLog(
            safe_values=np.array([30.0, 40.0, 50.0, 60.0, 70.0]),
            risky_outcomes=np.array([[70.0, 0.0], [90.0, 0.0], [110.0, 0.0], [130.0, 0.0], [150.0, 0.0]]),
            risky_probabilities=np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),
            choices=np.array([False, False, True, True, True]),
        )
        result = compute_risk_profile(log)
        assert result.consistency_score < 1.0
