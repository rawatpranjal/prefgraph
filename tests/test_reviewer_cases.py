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

        # Boundary case (verified against a continuous-bisection oracle and by
        # hand): the only revealed-preference relation, x0 R x1, is an exact
        # affordability tie at e=1 (p0*q1 = p0*q0 = 3). The data is therefore
        # e-GARP-consistent for every e < 1, so the CCEI supremum is exactly 1.0
        # even though GARP fails at e=1 (Smeulders et al. 2014). CCEI=1.0 does
        # NOT imply perfect consistency: is_perfectly_consistent stays False and
        # the violation is still reported above. (Before the supremum fix this
        # returned the spurious lower breakpoint 0.8.)
        aei = compute_integrity_score(log)
        assert aei.efficiency_index == pytest.approx(1.0, abs=1e-9)
        assert aei.is_perfectly_consistent is False

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

    def test_mci_symmetric_violation(self):
        """MCI on symmetric 2-obs GARP violation returns positive cost.

        Regression: MCI had a sign error yielding 0 on obvious violations.
        Slack for each edge is 0.2, so breaking the cheapest edge costs 0.2.
        """
        from prefgraph import compute_minimum_cost_index

        p = np.array([[1.0, 2.0], [2.0, 1.0]])
        q = np.array([[0.2, 0.4], [0.4, 0.2]])
        log = BehaviorLog(prices=p, quantities=q)
        r = compute_minimum_cost_index(log)

        assert r.is_consistent is False
        assert r.mci_value == pytest.approx(0.2, abs=0.01)
        assert r.mci_normalized == pytest.approx(0.1, abs=0.01)

    def test_mci_consistent_data(self):
        """MCI is zero for GARP-consistent data."""
        from prefgraph import compute_minimum_cost_index

        p = np.array([[1.0, 2.0], [2.0, 1.0]])
        q = np.array([[4.0, 1.0], [1.0, 4.0]])
        log = BehaviorLog(prices=p, quantities=q)
        r = compute_minimum_cost_index(log)

        assert r.is_consistent is True
        assert r.mci_value == 0.0
        assert r.mci_normalized == 0.0
        assert r.adjustments == {}

    def test_mci_adjustments_nonempty(self):
        """MCI adjustments dict is non-empty when violations exist."""
        from prefgraph import compute_minimum_cost_index

        p = np.array([[1.0, 2.0], [2.0, 1.0]])
        q = np.array([[0.2, 0.4], [0.4, 0.2]])
        log = BehaviorLog(prices=p, quantities=q)
        r = compute_minimum_cost_index(log)

        assert len(r.adjustments) > 0
        assert all(v > 0 for v in r.adjustments.values())


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


# =============================================================================
# Bug-coverage regression cases (audit-flagged, previously uncovered)
# =============================================================================


class TestHoutmanMaksGreedyTransitiveClosure:
    """Regression for bug (3): the greedy Houtman-Maks decomposition must run
    the SCC step on R_star (the transitive closure of revealed preference),
    not on R (the direct relation).

    The guarantee a Houtman-Maks heuristic must satisfy is feasibility: after
    removing the chosen observations, the surviving subset is itself
    GARP-consistent, and on inconsistent data at least one observation is
    removed. A regression that runs the SCC step on R and fails to group
    transitively-linked observations would skip removals and leave the
    remaining subset inconsistent. This exercises ``method="greedy"`` directly
    so the greedy path is tested rather than the exact ILP used for small T.
    """

    def test_greedy_yields_consistent_remainder(self):
        from prefgraph import compute_houtman_maks_index, validate_consistency

        # 3-observation budget log with a transitive GARP violation. Varied
        # prices create strict revealed-preference edges that form a cycle.
        p = np.array(
            [[1.1588, 1.3826], [0.6910, 1.5892], [0.9201, 0.7859]]
        )
        q = np.array(
            [[6.0711, 27.1108], [27.4690, 8.5543], [24.0283, 9.2515]]
        )
        log = BehaviorLog(prices=p, quantities=q)

        assert validate_consistency(log).is_consistent is False

        hm = compute_houtman_maks_index(log, method="greedy")
        # Greedy must detect the violation and remove at least one observation.
        assert hm.num_removed >= 1
        assert 0.0 < hm.fraction <= 1.0

        # Core Houtman-Maks guarantee: the surviving subset satisfies GARP.
        keep = [
            i for i in range(p.shape[0]) if i not in set(hm.removed_observations)
        ]
        survivor = BehaviorLog(prices=p[keep], quantities=q[keep])
        assert validate_consistency(survivor).is_consistent is True


class TestProductionGarpStrictCondition:
    """Regression for bug (4): production GARP must declare a violation on
    ``R_star[i, j] and P[j, i]`` (a strict reverse profit edge), not on
    ``R_star[i, j] and R_star[j, i]`` (a mere mutual-reachability cycle).

    Two firms that are mutually *weakly* as profitable as each other but never
    strictly so represent a profit tie, which is profit-maximizing consistent.
    The old cycle-only condition falsely rejected such ties.
    """

    def test_profit_tie_is_consistent(self):
        from prefgraph.algorithms.production import test_profit_maximization
        from prefgraph.core.session import ProductionLog

        # Flat input and output prices make cross-profits symmetric. The two
        # firms earn equal profit (2 each) and neither is ever strictly more
        # profitable, so R is mutual but P is empty -> no GARP violation.
        log = ProductionLog(
            input_prices=np.array([[1.0], [1.0]]),
            input_quantities=np.array([[1.0], [2.0]]),
            output_prices=np.array([[1.0], [1.0]]),
            output_quantities=np.array([[3.0], [4.0]]),
        )
        res = test_profit_maximization(log)
        # Old R_star[j,i] cycle condition returned False here.
        assert res.is_profit_maximizing is True
        assert res.num_violations == 0

    def test_strict_profit_cycle_still_rejected(self):
        from prefgraph.algorithms.production import test_profit_maximization
        from prefgraph.core.session import ProductionLog

        # Crossing output bundles at crossing prices: each firm's bundle is
        # strictly more profitable at its own prices, a genuine strict cycle.
        log = ProductionLog(
            input_prices=np.array([[1.0, 1.0], [1.0, 1.0]]),
            input_quantities=np.array([[1.0, 1.0], [1.0, 1.0]]),
            output_prices=np.array([[1.5, 1.0], [1.0, 1.5]]),
            output_quantities=np.array([[4.0, 3.0], [3.0, 4.0]]),
        )
        res = test_profit_maximization(log)
        assert res.is_profit_maximizing is False
        assert res.num_violations > 0


class TestDatasetLoadersBaseInstall:
    """Regression for bug (8): importing ``prefgraph.datasets`` must not require
    pandas.

    The retailrocket, rees46, taobao, and tenrec loaders import pandas at
    module top level, so the package ``__init__`` wraps them in lazy functions.
    With pandas unavailable (a base install), importing the package and binding
    those loader names must still succeed and must not eagerly import the
    offending submodules. The check runs in a subprocess so the pandas block
    does not leak into the rest of the test session.
    """

    def test_datasets_import_without_pandas(self):
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            r"""
            import sys, importlib.abc, importlib.machinery
            for _m in list(sys.modules):
                if _m == "pandas" or _m.startswith("pandas."):
                    del sys.modules[_m]

            class _Block(importlib.abc.MetaPathFinder, importlib.abc.Loader):
                # Return a spec so probes via find_spec succeed (polars does
                # this for its own lazy pandas detection), but make any real
                # `import pandas` fail, exactly as on a base install.
                def find_spec(self, name, path, target=None):
                    if name == "pandas" or name.startswith("pandas."):
                        return importlib.machinery.ModuleSpec(name, self)
                    return None

                def create_module(self, spec):
                    return None

                def exec_module(self, module):
                    raise ModuleNotFoundError("No module named 'pandas'")

            sys.meta_path.insert(0, _Block())

            try:
                import pandas  # noqa: F401
                raise SystemExit("pandas was not blocked")
            except ModuleNotFoundError:
                pass

            import prefgraph  # noqa: F401
            import prefgraph.datasets  # noqa: F401
            from prefgraph.datasets import (  # noqa: F401
                load_retailrocket,
                load_rees46,
                load_taobao,
                load_tenrec,
            )

            for _mod in (
                "prefgraph.datasets._retailrocket",
                "prefgraph.datasets._rees46",
                "prefgraph.datasets._taobao",
                "prefgraph.datasets._tenrec",
            ):
                assert _mod not in sys.modules, f"eagerly imported {_mod}"

            print("BASE_INSTALL_OK")
            """
        )

        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, (
            "datasets import crashed on a simulated base install (no pandas):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
        assert "BASE_INSTALL_OK" in proc.stdout
