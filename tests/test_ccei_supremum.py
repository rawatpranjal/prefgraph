"""Regression and oracle tests for the CCEI/AEI supremum computation.

These pin the fix for the discrete-search supremum bug. Previously the index
snapped to the next-lower breakpoint, and a one-float-ULP probe only patched the
case where the binding own-expenditure was about 1, so the CCEI was wrong on
roughly 7 percent of random data, by up to 0.4. The correct value is the
supremum sup{e : the axiom holds at e}, located exactly by testing the open
intervals between consecutive expenditure ratios (Smeulders et al. 2014, Alg 2).

`test_ccei_matches_oracle_on_random_data` is the structural guardrail: it
cross-checks the implementation against an independent continuous-bisection
reference, which would have caught the original bug. The per-efficiency axiom
check it relies on was separately verified correct; only the search was buggy.
"""

from __future__ import annotations

import numpy as np
import pytest

from prefgraph import BehaviorLog, compute_aei
from prefgraph.algorithms._budget_axioms import check_budget_axiom_at_efficiency


def _oracle_ccei(p: np.ndarray, q: np.ndarray, axiom: str = "garp", iters: int = 80) -> float:
    """Independent supremum reference via continuous bisection over [0, 1].

    Uses the per-efficiency axiom check (not the CCEI search), so it is
    independent of the code under test. Converges to the supremum from below,
    agreeing with the exact breakpoint to bisection precision.
    """
    log = BehaviorLog(prices=p, quantities=q)
    if check_budget_axiom_at_efficiency(log, axiom, 1.0).is_consistent:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if check_budget_axiom_at_efficiency(log, axiom, mid).is_consistent:
            lo = mid
        else:
            hi = mid
    return lo


# (name, prices, quantities, expected_ccei)
_CASES = [
    # own_exp ~= 1: the only family the old one-ULP probe got right.
    ("own_exp_1", [[1.0, 3.0], [3.0, 1.0], [0.4, 1.0]], [[2.0, 0.0], [1.0, 0.0], [0.0, 1.0]], 0.4),
    # own_exp = 25: the probe returned 5/31 ~= 0.161; the true supremum is 23/25.
    ("own_exp_25", [[3.0, 5.0], [5.0, 1.0]], [[0.0, 5.0], [6.0, 1.0]], 0.92),
    # Symmetric 2-good 2-cycle: 7/8.
    ("symmetric", [[2.0, 1.0], [1.0, 2.0]], [[3.0, 2.0], [2.0, 3.0]], 0.875),
]


@pytest.mark.parametrize("name,p,q,expected", _CASES)
def test_ccei_known_supremum(name: str, p: list, q: list, expected: float) -> None:
    parr = np.array(p)
    qarr = np.array(q)
    log = BehaviorLog(prices=parr, quantities=qarr)
    got = compute_aei(log, axiom="garp").efficiency_index
    assert got == pytest.approx(expected, abs=1e-9), f"{name}: {got} != {expected}"
    assert got == pytest.approx(_oracle_ccei(parr, qarr), abs=1e-6)


def test_ccei_boundary_violation_only_at_one() -> None:
    # The only revealed-preference relation is an exact affordability tie at
    # e=1, so the data is e-GARP-consistent for every e < 1 and the supremum is
    # 1.0 even though GARP fails at e=1. CCEI=1.0 does not imply consistency.
    p = np.array([[1.0, 1.0], [1.0, 2.0]])
    q = np.array([[2.0, 1.0], [1.0, 2.0]])
    log = BehaviorLog(prices=p, quantities=q)
    r = compute_aei(log, axiom="garp")
    assert r.efficiency_index == pytest.approx(1.0, abs=1e-9)
    assert r.is_perfectly_consistent is False


def test_sarp_warp_ccei_match_oracle_non_power_of_two() -> None:
    # Non-power-of-two data exercising the SARP and WARP supremum paths, which
    # the old np.nextafter probe computed incorrectly on a large fraction of
    # inputs. Both backends and the oracle now agree on 0.8995.
    p = np.array([[0.9259, 2.4226], [1.6153, 1.0423]])
    q = np.array([[1.7322, 1.9321], [3.5555, 0.9745]])
    log = BehaviorLog(prices=p, quantities=q)
    for axiom in ("garp", "sarp", "warp"):
        got = compute_aei(log, axiom=axiom).efficiency_index
        assert got == pytest.approx(_oracle_ccei(p, q, axiom), abs=1e-6), axiom
        assert got == pytest.approx(0.8995, abs=1e-3), axiom


def test_ccei_matches_oracle_on_random_data() -> None:
    """Guardrail: CCEI must equal an independent supremum oracle on random data.

    This is exactly the check that would have caught the original discrete
    search bug. When the Rust backend is present this exercises it (compute_aei
    routes GARP to Rust); under PREFGRAPH_NO_RUST it exercises the pure-Python
    search. Either way both must agree with the oracle.
    """
    rng = np.random.default_rng(20260609)
    max_err = 0.0
    for _ in range(300):
        t = int(rng.integers(2, 7))
        g = int(rng.integers(2, 4))
        p = rng.uniform(0.2, 3.0, size=(t, g))
        q = rng.uniform(0.1, 5.0, size=(t, g))
        log = BehaviorLog(prices=p, quantities=q)
        got = compute_aei(log, axiom="garp").efficiency_index
        ref = _oracle_ccei(p, q)
        max_err = max(max_err, abs(got - ref))
    assert max_err < 1e-6, f"CCEI deviates from the supremum oracle by {max_err}"
