"""Property-based invariants enforced with Hypothesis.

These encode mathematical guarantees that must hold for *any* admissible
input, not just the hand-picked cases in the example-based suites:

  1. Utility-maximizing (rational) data ALWAYS passes GARP. There are no
     false positives: a consumer whose quantities are the demand that
     maximizes a fixed concave utility at the observed prices is, by
     construction, rationalizable.
  2. CCEI (the Afriat efficiency index) is always in [0, 1].
  3. The Houtman-Maks removed-fraction is always in [0, 1].
  4. The Rust and Python backends agree within the documented tolerances:
     GARP exact, CCEI within 0.01, MPI within 0.05.

Random data is drawn either directly by Hypothesis or via the existing
simulation generators in ``tests/simulations/generators.py``. No new data
simulators are introduced here. Examples are kept small (few goods, few
observations) and bounded so the suite stays fast and numerically stable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# The simulation generators live in a directory that pytest does not collect
# (norecursedirs), but it is still importable as a plain module.
sys.path.insert(0, str(Path(__file__).parent / "simulations"))
from generators import (  # noqa: E402
    cobb_douglas_demand,
    generate_irrational_data,
)

from prefgraph import (  # noqa: E402
    compute_houtman_maks_index,
    compute_integrity_score,
    validate_consistency,
)
from prefgraph._rust_backend import HAS_RUST  # noqa: E402
from prefgraph.core.session import BehaviorLog  # noqa: E402
from prefgraph.engine import Engine  # noqa: E402

# Small, bounded examples keep each property fast and numerically well-behaved.
PROP_SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)

_n_obs = st.integers(min_value=2, max_value=6)
_n_goods = st.integers(min_value=2, max_value=4)
_seeds = st.integers(min_value=0, max_value=10_000)

# Backend-parity flag dict expected by Engine._analyze_chunk_* (mirrors
# tests/test_backend_parity.py). Only GARP/CCEI/MPI are compared.
_PARITY_FLAGS = {
    "ccei": True,
    "mpi": True,
    "harp": False,
    "hm": False,
    "utility": False,
    "vei": False,
    "vei_exact": False,
}


# ---------------------------------------------------------------------------
# Property 1: rational data is never flagged as a GARP violation.
# ---------------------------------------------------------------------------


@PROP_SETTINGS
@given(data=st.data())
def test_rational_data_always_passes_garp(data):
    """Cobb-Douglas demand at any prices satisfies GARP (no false positives).

    Standard construction: fix a strictly concave Cobb-Douglas utility with
    weights alpha on the simplex. Its demand at prices p and budget m is the
    unique utility maximizer x_i = alpha_i * m / p_i. Such data is
    rationalizable by definition, so the GARP test must return consistent.
    """
    n_obs = data.draw(_n_obs, label="n_obs")
    n_goods = data.draw(_n_goods, label="n_goods")

    prices = data.draw(
        arrays(
            np.float64,
            (n_obs, n_goods),
            elements=st.floats(0.5, 5.0, allow_nan=False, allow_infinity=False),
        ),
        label="prices",
    )
    raw_alpha = data.draw(
        arrays(
            np.float64,
            (n_goods,),
            elements=st.floats(0.1, 1.0, allow_nan=False, allow_infinity=False),
        ),
        label="alpha",
    )
    alpha = raw_alpha / raw_alpha.sum()  # weights on the simplex (sum to 1)
    budgets = data.draw(
        arrays(
            np.float64,
            (n_obs,),
            elements=st.floats(10.0, 100.0, allow_nan=False, allow_infinity=False),
        ),
        label="budgets",
    )

    quantities = np.vstack(
        [cobb_douglas_demand(prices[t], budgets[t], alpha) for t in range(n_obs)]
    )

    log = BehaviorLog(prices=prices, quantities=quantities)
    result = validate_consistency(log)
    assert result.is_consistent is True, (
        "Rational (utility-maximizing) data was flagged as inconsistent.\n"
        f"prices=\n{prices}\nquantities=\n{quantities}\nalpha={alpha}"
    )
    assert len(result.violations) == 0


# ---------------------------------------------------------------------------
# Property 2: CCEI is always a valid efficiency score in [0, 1].
# ---------------------------------------------------------------------------


@PROP_SETTINGS
@given(n_obs=_n_obs, n_goods=_n_goods, seed=_seeds)
def test_ccei_always_in_unit_interval(n_obs, n_goods, seed):
    """The Afriat efficiency index lies in [0, 1] for arbitrary choice data."""
    prices, quantities = generate_irrational_data(n_obs, n_goods, seed=seed)
    log = BehaviorLog(prices=prices, quantities=quantities)
    ccei = compute_integrity_score(log).efficiency_index
    assert 0.0 <= ccei <= 1.0 + 1e-9, f"CCEI out of range: {ccei}"


# ---------------------------------------------------------------------------
# Property 3: the Houtman-Maks fraction is always in [0, 1].
# ---------------------------------------------------------------------------


@PROP_SETTINGS
@given(n_obs=_n_obs, n_goods=_n_goods, seed=_seeds)
def test_houtman_maks_fraction_in_unit_interval(n_obs, n_goods, seed):
    """HM removed-fraction (and its complementary efficiency) lie in [0, 1]."""
    prices, quantities = generate_irrational_data(n_obs, n_goods, seed=seed)
    log = BehaviorLog(prices=prices, quantities=quantities)
    hm = compute_houtman_maks_index(log)
    assert 0.0 <= hm.fraction <= 1.0, f"HM fraction out of range: {hm.fraction}"
    assert 0.0 <= hm.efficiency <= 1.0, f"HM efficiency out of range: {hm.efficiency}"
    assert hm.num_removed <= n_obs


# ---------------------------------------------------------------------------
# Property 4: Rust and Python backends agree within documented tolerances.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_RUST, reason="Rust backend not available")
@PROP_SETTINGS
@given(n_obs=_n_obs, n_goods=_n_goods, seed=_seeds)
def test_rust_python_backends_agree(n_obs, n_goods, seed):
    """Both backends must return the same GARP verdict and close CCEI/MPI.

    Tolerances (see CLAUDE.md "Backend Parity"): GARP exact, CCEI within
    0.01 (same discrete binary search), MPI within 0.05 (Python uses cycle
    enumeration, Rust uses Karp's max-mean-weight cycle).
    """
    prices, quantities = generate_irrational_data(n_obs, n_goods, seed=seed)
    chunk = [(prices.astype(np.float64), quantities.astype(np.float64))]

    engine = Engine(metrics=["garp", "ccei", "mpi"])
    py = engine._analyze_chunk_python(chunk, _PARITY_FLAGS)[0]
    rust = engine._analyze_chunk_rust(chunk, _PARITY_FLAGS)[0]

    assert py.is_garp == rust.is_garp, (
        f"GARP disagreement: py={py.is_garp} rust={rust.is_garp}"
    )
    assert abs(py.ccei - rust.ccei) <= 0.01, (
        f"CCEI gap {abs(py.ccei - rust.ccei):.4f} > 0.01 "
        f"(py={py.ccei:.4f} rust={rust.ccei:.4f})"
    )
    assert abs(py.mpi - rust.mpi) <= 0.05, (
        f"MPI gap {abs(py.mpi - rust.mpi):.4f} > 0.05 "
        f"(py={py.mpi:.4f} rust={rust.mpi:.4f})"
    )


# ---------------------------------------------------------------------------
# Properties 5-8: exact VEI (Mononen 2023 Theorem 1, canonical vector).
# ---------------------------------------------------------------------------

_VEI_EXACT_FIELDS = (
    "vei_exact_mean",
    "vei_exact_min",
    "vei_exact_std",
    "vei_exact_q25",
    "vei_exact_q75",
)

_VEI_EXACT_FLAGS = {
    "ccei": False,
    "mpi": False,
    "harp": False,
    "hm": False,
    "utility": False,
    "vei": False,
    "vei_exact": True,
}


@PROP_SETTINGS
@given(n_obs=_n_obs, n_goods=_n_goods, seed=_seeds)
def test_vei_exact_bounds_and_ccei_dominance(n_obs, n_goods, seed):
    """Exact-VEI efficiencies lie in [0, 1] and the mean weakly exceeds CCEI.

    A uniform adjustment of 1 - CCEI at every observation makes the relaxed
    relation acyclic, so it is feasible for Varian's per-observation problem;
    the Varian average adjustment is therefore weakly smaller and the mean
    efficiency weakly larger (Mononen 2023, Section 2.4 relates the measures;
    the tolerance absorbs the discrete CCEI search and supremum convention).
    """
    prices, quantities = generate_irrational_data(n_obs, n_goods, seed=seed)
    engine = Engine(metrics=["garp", "ccei", "vei_exact"])
    res = engine.analyze_arrays(
        [(prices.astype(np.float64), quantities.astype(np.float64))]
    )[0]
    for field in _VEI_EXACT_FIELDS:
        value = getattr(res, field)
        assert -1e-12 <= value <= 1.0 + 1e-12, f"{field} out of range: {value}"
    assert res.vei_exact_min <= res.vei_exact_mean + 1e-12
    assert res.vei_exact_mean >= res.ccei - 1e-6, (
        f"Varian mean efficiency {res.vei_exact_mean:.6f} below "
        f"CCEI {res.ccei:.6f}: a uniform CCEI adjustment is Varian-feasible"
    )


@PROP_SETTINGS
@given(data=st.data())
def test_vei_exact_rational_data_all_ones(data):
    """Utility-maximizing (Cobb-Douglas) data has exact VEI exactly 1.0."""
    n_obs = data.draw(_n_obs, label="n_obs")
    n_goods = data.draw(_n_goods, label="n_goods")
    prices = data.draw(
        arrays(
            np.float64,
            (n_obs, n_goods),
            elements=st.floats(0.5, 5.0, allow_nan=False, allow_infinity=False),
        ),
        label="prices",
    )
    raw_alpha = data.draw(
        arrays(
            np.float64,
            (n_goods,),
            elements=st.floats(0.1, 1.0, allow_nan=False, allow_infinity=False),
        ),
        label="alpha",
    )
    alpha = raw_alpha / raw_alpha.sum()
    budgets = data.draw(
        arrays(
            np.float64,
            (n_obs,),
            elements=st.floats(10.0, 100.0, allow_nan=False, allow_infinity=False),
        ),
        label="budgets",
    )
    quantities = np.vstack(
        [cobb_douglas_demand(prices[t], budgets[t], alpha) for t in range(n_obs)]
    )
    engine = Engine(metrics=["garp", "vei_exact"])
    res = engine.analyze_arrays([(prices, quantities)])[0]
    assert res.vei_exact_mean == 1.0
    assert res.vei_exact_min == 1.0
    assert res.vei_exact_std == 0.0


@PROP_SETTINGS
@given(n_obs=_n_obs, seed=_seeds)
def test_vei_exact_deterministic(n_obs, seed):
    """Two runs on identical tie-rich integer data are bitwise identical:
    the canonical vector leaves no room for solver vertex choice."""
    rng = np.random.default_rng(seed)
    prices = rng.integers(1, 7, size=(n_obs, 2)).astype(np.float64)
    quantities = rng.integers(1, 7, size=(n_obs, 2)).astype(np.float64)
    engine = Engine(metrics=["garp", "vei_exact"])
    first = engine.analyze_arrays([(prices, quantities)])[0]
    second = engine.analyze_arrays([(prices, quantities)])[0]
    for field in _VEI_EXACT_FIELDS:
        assert getattr(first, field) == getattr(second, field), field


@pytest.mark.skipif(not HAS_RUST, reason="Rust backend not available")
@PROP_SETTINGS
@given(n_obs=_n_obs, n_goods=st.integers(min_value=2, max_value=3), seed=_seeds)
def test_vei_exact_backends_agree_exactly_on_integer_data(n_obs, n_goods, seed):
    """Rust and the pure-Python mirror are EXACTLY equal on integer data.

    HAS_RUST is forced off for the Python side (CLAUDE.md Learned Rules:
    parity tests must force the fallback path). Small integer grids maximize
    ties, the regime where the pre-fix backends returned different optima.
    """
    import prefgraph._rust_backend as rb

    rng = np.random.default_rng(seed)
    prices = rng.integers(1, 7, size=(n_obs, n_goods)).astype(np.float64)
    quantities = rng.integers(1, 7, size=(n_obs, n_goods)).astype(np.float64)
    chunk = [(prices, quantities)]

    engine = Engine(metrics=["garp", "vei_exact"])
    rust = engine._analyze_chunk_rust(chunk, _VEI_EXACT_FLAGS)[0]
    saved = rb.HAS_RUST
    rb.HAS_RUST = False
    try:
        py = engine._analyze_chunk_python(chunk, _VEI_EXACT_FLAGS)[0]
    finally:
        rb.HAS_RUST = saved

    for field in _VEI_EXACT_FIELDS:
        pv, rv = getattr(py, field), getattr(rust, field)
        assert pv == rv, f"{field}: python={pv!r} != rust={rv!r}"
