"""Exact VEI (Varian index) tests against a definition-based oracle.

Varian's index, as defined in Mononen (2023) p. 9 (papers/
Mononen2023_ComputingMeasures.pdf), is

    I_Var(D) = (1/T) inf_{d in [0,1]^T} sum_t d_t
    such that the relaxed strict revealed preference is acyclic,

where d_t is the budget adjustment at observation t (the efficiency level is
e_t = 1 - d_t) and the relaxed strict preference keeps arc (i, j) iff
cost_ij = 1 - (p_i . x_j)/(p_i . x_i) exceeds d_i. At the infimum each d_t is
one of the arc costs out of t, or 0 (Mononen p. 10), so the index is computed
exactly by enumerating the finite per-observation cost grids. That enumeration
is `varian_oracle` below: an independent oracle derived from the definition,
never from the package's own output (see CLAUDE.md Learned Rules).

The optimal adjustment VECTOR is not unique under ties. PrefGraph reports the
canonical vector: among value-optimal solutions, first minimize the maximum
adjustment (equivalently maximize the minimum efficiency), then take the
lexicographically smallest adjustment vector in observation order. The oracle
applies the same convention, so vector statistics (min/std/q25/q75) are pinned.

Tolerances: arc construction uses the parse_budget strict tolerance 1e-10 and
the cost filter 1e-12; an arc survives adjustment d_i iff cost - d_i > 1e-12
(AddCost > 0 in Mononen Algorithm 1). These must match the Rust and Python
implementations exactly.
"""

import itertools

import numpy as np
import pytest

from prefgraph._rust_backend import HAS_RUST
from prefgraph.engine import Engine

TOL_STRICT = 1e-10  # parse_budget tolerance: strict arc iff own - E[i,j] > tol
TOL_COST = 1e-12  # cost filter and AddCost survival tolerance


# ---------------------------------------------------------------------------
# Definition-based oracle (independent of the package implementation)
# ---------------------------------------------------------------------------


def arcs_and_costs(prices, quantities):
    """Strict revealed-preference arcs and their removal costs.

    Arc (i, j) exists iff p_i.x_i > p_i.x_j + TOL_STRICT, with removal cost
    cost_ij = 1 - (p_i.x_j)/(p_i.x_i). Costs below TOL_COST are dropped: such
    arcs are removable at infimum cost zero and never bind (Mononen p. 10
    restricts attention to strict preferences for exactly this reason).
    """
    p = np.asarray(prices, dtype=float)
    x = np.asarray(quantities, dtype=float)
    expenditure = p @ x.T
    own = np.diag(expenditure)
    arcs = {}
    n_obs = p.shape[0]
    for i in range(n_obs):
        for j in range(n_obs):
            if i != j and own[i] - expenditure[i, j] > TOL_STRICT:
                cost = 1.0 - expenditure[i, j] / own[i]
                if cost > TOL_COST:
                    arcs[(i, j)] = cost
    return arcs


def _is_acyclic(n_obs, surviving_arcs):
    color = [0] * n_obs  # 0 white, 1 gray, 2 black
    adj = [[] for _ in range(n_obs)]
    for i, j in surviving_arcs:
        adj[i].append(j)

    def dfs(u):
        color[u] = 1
        for v in adj[u]:
            if color[v] == 1:
                return False
            if color[v] == 0 and not dfs(v):
                return False
        color[u] = 2
        return True

    return all(dfs(s) for s in range(n_obs) if color[s] == 0)


def varian_oracle(prices, quantities):
    """Exact Varian index by exhaustive grid enumeration.

    Returns (total_adjustment, all_optimal_d_vectors, canonical_d_vector).
    Only valid for small T (grid product is exponential); tests guard size.
    """
    p = np.asarray(prices, dtype=float)
    n_obs = p.shape[0]
    arcs = arcs_and_costs(prices, quantities)
    grids = []
    for i in range(n_obs):
        grids.append(sorted({0.0} | {c for (a, _), c in arcs.items() if a == i}))

    best = None
    optima = []
    for d in itertools.product(*grids):
        surviving = [(i, j) for (i, j), c in arcs.items() if c - d[i] > TOL_COST]
        if not _is_acyclic(n_obs, surviving):
            continue
        total = sum(d)
        if best is None or total < best - 1e-12:
            best, optima = total, [d]
        elif abs(total - best) <= 1e-12:
            optima.append(d)
    return best, optima, canonical_choice(optima)


def canonical_choice(optima):
    """Max-min-then-lex convention: among value-optimal adjustment vectors,
    keep those minimizing the maximum adjustment, then take the
    lexicographically smallest vector in observation order."""
    min_max = min(max(d) for d in optima)
    candidates = [d for d in optima if max(d) <= min_max + 1e-12]
    return min(candidates)


def canonical_stats(d_vector):
    """The five Engine summary statistics implied by a canonical d vector."""
    e = 1.0 - np.asarray(d_vector, dtype=float)
    return {
        "mean": float(np.mean(e)),
        "min": float(np.min(e)),
        "std": float(np.std(e)),
        "q25": float(np.percentile(e, 25)),
        "q75": float(np.percentile(e, 75)),
    }


def engine_exact(prices, quantities):
    eng = Engine(metrics=["vei_exact"])
    res = eng.analyze_arrays(
        [(np.asarray(prices, dtype=float), np.asarray(quantities, dtype=float))]
    )[0]
    return {
        "mean": res.vei_exact_mean,
        "min": res.vei_exact_min,
        "std": res.vei_exact_std,
        "q25": res.vei_exact_q25,
        "q75": res.vei_exact_q75,
    }


# ---------------------------------------------------------------------------
# Fixtures with hand-derived optima (verified by the oracle by exhaustion)
# ---------------------------------------------------------------------------

# 2-obs WARP violation. E = [[8, 7], [7, 8]]. Arcs 0->1 and 1->0, both cost
# 1/8. Optima: d = (1/8, 0) or (0, 1/8); equal max, lex picks (0, 1/8).
# Value 1/8, canonical e = (1, 7/8), mean 15/16 = 0.9375.
ANCHOR_2OBS = (
    [[2.0, 1.0], [1.0, 2.0]],
    [[3.0, 2.0], [2.0, 3.0]],
)
ANCHOR_2OBS_TOTAL = 1.0 / 8.0
ANCHOR_2OBS_CANON_D = (0.0, 1.0 / 8.0)

# Nested-removal divergence (the bug this fix targets). T=3, G=3.
# E = [[39, 37, 39], [45, 55, 45], [51, 41, 51]].
# Arcs: 0->1 cost 2/39, 1->0 cost 2/11, 1->2 cost 2/11, 2->1 cost 10/51.
# Two 2-cycles, (0,1) and (1,2), sharing observation 1. One adjustment
# d_1 = 2/11 removes BOTH arcs out of 1 (equal cost) and breaks both cycles:
# true value 2/11 ~ 0.18182. Charging each removed arc independently (the
# pre-fix formulation, which lacks the Theorem 1 U-set expansion) instead
# pays 2/39 + 2/11 ~ 0.23310. Canonical e = (1, 9/11, 1), mean 31/33.
NESTED_T3 = (
    [[2.0, 3.0, 5.0], [1.0, 6.0, 6.0], [6.0, 3.0, 5.0]],
    [[3.0, 1.0, 6.0], [1.0, 5.0, 4.0], [3.0, 1.0, 6.0]],
)
NESTED_T3_TOTAL = 2.0 / 11.0
NESTED_T3_CANON_D = (0.0, 2.0 / 11.0, 0.0)

# Stage-B (max-min) fixture. T=4, G=2.
# E = [[12, 18, 18, 15], [12, 20, 16, 22], [7, 10, 11, 7], [12, 20, 16, 22]].
# Two value-optimal solutions, both totaling 4/11:
#   A: d = (0, 0, 1/11, 3/11)  max 3/11
#   B: d = (0, 0, 4/11, 0)     max 4/11
# The max-min stage must pick A. Canonical e = (1, 1, 10/11, 8/11).
STAGE_B = (
    [[3.0, 3.0], [6.0, 2.0], [1.0, 2.0], [6.0, 2.0]],
    [[1.0, 3.0], [2.0, 4.0], [1.0, 5.0], [3.0, 2.0]],
)
STAGE_B_TOTAL = 4.0 / 11.0
STAGE_B_CANON_D = (0.0, 0.0, 1.0 / 11.0, 3.0 / 11.0)

# Stage-C (lexicographic) fixture. T=4, G=2.
# E = [[13, 21, 18, 17], [10, 16, 14, 13], [5, 7, 8, 6], [9, 13, 14, 11]].
# The only strict cycle is the 2-cycle between observations 1 and 2 (arcs
# 1->2 and 2->1, both cost 1/8). Optima: d = (0, 1/8, 0, 0) or
# (0, 0, 1/8, 0); equal max, lex minimizes d_1 first so observation 1 keeps
# efficiency 1. Canonical e = (1, 1, 7/8, 1).
STAGE_C = (
    [[4.0, 1.0], [3.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
    [[3.0, 1.0], [5.0, 1.0], [4.0, 2.0], [4.0, 1.0]],
)
STAGE_C_TOTAL = 1.0 / 8.0
STAGE_C_CANON_D = (0.0, 0.0, 1.0 / 8.0, 0.0)

# Larger regression: T=7, G=3 integer data found by random search where the
# pre-fix formulation overpays (0.30752 vs the true 0.24141). The optimum
# adjusts observation 0 by 23/136 (removing arc 0->2 and the cheaper 0->1,
# 0->4) and observation 6 by 6/83 (removing arc 6->3 and the cheaper 6->4).
# Optimality is established by the oracle's exhaustive enumeration.
LARGE_T7 = (
    [
        [9.0, 7.0, 6.0],
        [3.0, 4.0, 1.0],
        [1.0, 1.0, 2.0],
        [9.0, 7.0, 10.0],
        [6.0, 7.0, 10.0],
        [8.0, 7.0, 6.0],
        [6.0, 10.0, 3.0],
    ],
    [
        [9.0, 7.0, 1.0],
        [4.0, 9.0, 6.0],
        [1.0, 8.0, 8.0],
        [9.0, 2.0, 1.0],
        [9.0, 1.0, 6.0],
        [1.0, 3.0, 5.0],
        [5.0, 5.0, 1.0],
    ],
)
LARGE_T7_TOTAL = 23.0 / 136.0 + 6.0 / 83.0
LARGE_T7_CANON_D = (23.0 / 136.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0 / 83.0)

ALL_FIXTURES = [
    ("anchor_2obs", ANCHOR_2OBS, ANCHOR_2OBS_TOTAL, ANCHOR_2OBS_CANON_D),
    ("nested_t3", NESTED_T3, NESTED_T3_TOTAL, NESTED_T3_CANON_D),
    ("stage_b", STAGE_B, STAGE_B_TOTAL, STAGE_B_CANON_D),
    ("stage_c", STAGE_C, STAGE_C_TOTAL, STAGE_C_CANON_D),
    ("large_t7", LARGE_T7, LARGE_T7_TOTAL, LARGE_T7_CANON_D),
]


# ---------------------------------------------------------------------------
# Oracle self-checks (the hand derivations above must match the enumeration)
# ---------------------------------------------------------------------------


class TestOracleSelfCheck:
    @pytest.mark.parametrize("name,data,total,canon", ALL_FIXTURES)
    def test_fixture_derivations(self, name, data, total, canon):
        prices, quantities = data
        oracle_total, _, oracle_canon = varian_oracle(prices, quantities)
        assert oracle_total == pytest.approx(total, abs=1e-12), name
        assert np.allclose(oracle_canon, canon, atol=1e-12), (
            f"{name}: oracle canonical {oracle_canon} != derived {canon}"
        )

    def test_consistent_data_zero_adjustment(self):
        prices = [[1.0, 2.0], [2.0, 1.0]]
        quantities = [[4.0, 1.0], [1.0, 4.0]]
        total, _, canon = varian_oracle(prices, quantities)
        assert total == 0.0
        assert all(d == 0.0 for d in canon)


# ---------------------------------------------------------------------------
# Rust backend (Engine) against the oracle
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_RUST, reason="Rust backend not available")
class TestRustExactValue:
    """vei_exact_mean must equal 1 - I_Var. The value is unique (it is the
    optimum of Theorem 1), so these hold regardless of tie-breaking."""

    @pytest.mark.parametrize("name,data,total,canon", ALL_FIXTURES)
    def test_value_matches_oracle(self, name, data, total, canon):
        prices, quantities = data
        n_obs = len(prices)
        got = engine_exact(prices, quantities)
        assert got["mean"] == pytest.approx(1.0 - total / n_obs, abs=1e-9), name


@pytest.mark.skipif(not HAS_RUST, reason="Rust backend not available")
class TestRustCanonicalVector:
    """All five summary statistics must come from the canonical vector
    (max-min over value-optimal solutions, then lexicographic)."""

    @pytest.mark.parametrize("name,data,total,canon", ALL_FIXTURES)
    def test_stats_match_canonical(self, name, data, total, canon):
        prices, quantities = data
        expected = canonical_stats(canon)
        got = engine_exact(prices, quantities)
        for key in ("mean", "min", "std", "q25", "q75"):
            assert got[key] == pytest.approx(expected[key], abs=1e-9), (
                f"{name}.{key}: engine {got[key]} != canonical {expected[key]}"
            )


@pytest.mark.skipif(not HAS_RUST, reason="Rust backend not available")
class TestRustOracleSweep:
    """Random integer and continuous datasets, value and canonical vector."""

    def _sweep(self, rng, n_datasets, n_obs, n_goods, integer):
        checked = 0
        for _ in range(n_datasets):
            if integer:
                prices = rng.integers(1, 7, size=(n_obs, n_goods)).astype(float)
                quantities = rng.integers(1, 7, size=(n_obs, n_goods)).astype(float)
            else:
                prices = rng.uniform(0.5, 5.0, size=(n_obs, n_goods))
                quantities = rng.uniform(0.5, 5.0, size=(n_obs, n_goods))
            arcs = arcs_and_costs(prices, quantities)
            grid_size = 1
            for i in range(n_obs):
                grid_size *= 1 + sum(1 for (a, _) in arcs if a == i)
            if grid_size > 20000:
                continue
            total, _, canon = varian_oracle(prices, quantities)
            expected = canonical_stats(canon)
            got = engine_exact(prices, quantities)
            for key in ("mean", "min", "std", "q25", "q75"):
                assert got[key] == pytest.approx(expected[key], abs=1e-9), (
                    f"seed-case {checked} {key}: prices={prices.tolist()} "
                    f"quantities={quantities.tolist()}"
                )
            checked += 1
        assert checked >= n_datasets // 2, "size guard skipped too many cases"

    def test_integer_t4(self):
        self._sweep(np.random.default_rng(42), 40, 4, 2, integer=True)

    def test_integer_t5(self):
        self._sweep(np.random.default_rng(43), 40, 5, 3, integer=True)

    def test_continuous_t4(self):
        self._sweep(np.random.default_rng(44), 30, 4, 2, integer=False)


# ---------------------------------------------------------------------------
# Pure-Python mirror (per-user function and the Engine fallback path)
# ---------------------------------------------------------------------------


def python_exact(prices, quantities):
    from prefgraph.algorithms.vei import compute_vei_exact
    from prefgraph.core.session import ConsumerSession

    session = ConsumerSession(
        prices=np.asarray(prices, dtype=float),
        quantities=np.asarray(quantities, dtype=float),
    )
    return compute_vei_exact(session)


class TestPythonExactFunction:
    """compute_vei_exact in prefgraph.algorithms.vei must reproduce the
    oracle value and the full canonical vector, mirroring Rust exactly."""

    @pytest.mark.parametrize("name,data,total,canon", ALL_FIXTURES)
    def test_value_and_canonical_vector(self, name, data, total, canon):
        res = python_exact(*data)
        assert res.optimization_success, name
        assert res.total_inefficiency == pytest.approx(total, abs=1e-9), name
        expected_e = 1.0 - np.asarray(canon, dtype=float)
        assert np.allclose(res.efficiency_vector, expected_e, atol=1e-12), (
            f"{name}: {res.efficiency_vector} != {expected_e}"
        )

    def test_consistent_data_all_ones(self):
        prices = [[1.0, 2.0], [2.0, 1.0]]
        quantities = [[4.0, 1.0], [1.0, 4.0]]
        res = python_exact(prices, quantities)
        assert res.optimization_success
        assert res.total_inefficiency == 0.0
        assert np.all(res.efficiency_vector == 1.0)

    def test_sweep_integer_t4(self):
        rng = np.random.default_rng(45)
        checked = 0
        for _ in range(30):
            prices = rng.integers(1, 7, size=(4, 2)).astype(float)
            quantities = rng.integers(1, 7, size=(4, 2)).astype(float)
            arcs = arcs_and_costs(prices, quantities)
            grid_size = 1
            for i in range(4):
                grid_size *= 1 + sum(1 for (a, _) in arcs if a == i)
            if grid_size > 20000:
                continue
            total, _, canon = varian_oracle(prices, quantities)
            res = python_exact(prices, quantities)
            assert res.total_inefficiency == pytest.approx(total, abs=1e-9)
            assert np.allclose(
                res.efficiency_vector, 1.0 - np.asarray(canon), atol=1e-12
            ), f"prices={prices.tolist()} quantities={quantities.tolist()}"
            checked += 1
        assert checked >= 15


@pytest.mark.skipif(not HAS_RUST, reason="Rust backend not available")
class TestPerformanceGuard:
    """Loose wall-clock guard: 100 users x T=20 well under 30 s (measured
    0.56 s locally, 2026-06-10; the pre-fix code took 0.19 s, the canonical
    stages add the difference)."""

    def test_batch_100_users_t20(self):
        import time

        rng = np.random.default_rng(7)
        users = [
            (rng.uniform(0.5, 5.0, (20, 5)), rng.uniform(0.5, 5.0, (20, 5)))
            for _ in range(100)
        ]
        eng = Engine(metrics=["garp", "vei_exact"])
        start = time.perf_counter()
        res = eng.analyze_arrays(users)
        elapsed = time.perf_counter() - start
        assert len(res) == 100
        assert elapsed < 30.0, f"vei_exact batch took {elapsed:.1f}s"


class TestPythonEngineFallback:
    """The Engine fallback must compute the real exact index, not substitute
    the LP relaxation (forcing HAS_RUST=False per CLAUDE.md Learned Rules)."""

    def test_fallback_matches_canonical(self):
        import prefgraph._rust_backend as rb

        prices, quantities = NESTED_T3
        chunk = [(np.asarray(prices, dtype=float), np.asarray(quantities, dtype=float))]
        engine = Engine(metrics=["garp", "vei_exact"])
        saved = rb.HAS_RUST
        rb.HAS_RUST = False
        try:
            res = engine._analyze_chunk_python(chunk, {"vei_exact": True})[0]
        finally:
            rb.HAS_RUST = saved
        expected = canonical_stats(NESTED_T3_CANON_D)
        assert res.vei_exact_mean == pytest.approx(expected["mean"], abs=1e-9)
        assert res.vei_exact_min == pytest.approx(expected["min"], abs=1e-9)
        assert res.vei_exact_std == pytest.approx(expected["std"], abs=1e-9)
        assert res.vei_exact_q25 == pytest.approx(expected["q25"], abs=1e-9)
        assert res.vei_exact_q75 == pytest.approx(expected["q75"], abs=1e-9)
