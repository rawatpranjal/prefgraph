"""Cross-backend parity tests: Python vs Rust produce consistent results.

Tests that _analyze_chunk_python and _analyze_chunk_rust produce equivalent
results within known tolerance bounds for shared metrics (GARP, CCEI, MPI)
and for the new fields added to the Python fallback (SCC stats, network stats,
VEI stats, VEI exact, menu parity).

Tolerance rationale:
- GARP (bool): exact match - same Floyd-Warshall algorithm
- CCEI (float): within 0.01 - same discrete binary search, minor float rounding
- MPI (float): within 1e-6 - both backends compute the minimum cost-to-budget
  cycle ratio (the correct money-pump index), so they agree tightly
- max_scc, n_scc (int): exact - same SCC algorithm on same R matrix
- scc_mean_size (float): within 1e-10 - integer division, no rounding error
- r_density (float): within 1e-10 - counting edges over int constant
- r_out_degree_std (float): within 1e-8 - same formula, minor float order diffs
- degree_gini (float): within 1e-8 - same formula
- ew_mean/std/skew (float): within 1e-6 - same log-ratio formula
- vei_mean/min (float): within 0.05 - different LP implementations (scipy vs HiGHS)
- vei_std/q25/q75 (float): within 0.05 - derived from efficiency vector
- vei_exact: not tested for Rust parity (Rust uses a different exact LP;
  Python compute_vei is the closest equivalent, see engine.py docstring)
- Menu is_sarp/is_warp/n_violations (bool/int): exact
- Menu max_scc/n_scc: exact
- Menu r_density/pref_entropy/choice_diversity: within 1e-8
- Menu is_warp_la: exact (same algorithm path)
"""

import pytest
import numpy as np
from prefgraph._rust_backend import HAS_RUST
from prefgraph.engine import Engine

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust backend not available")

FLAGS = {"ccei": True, "mpi": True, "harp": False, "hm": False,
         "utility": False, "vei": False, "vei_exact": False, "network": False}

FLAGS_FULL = {"ccei": True, "mpi": True, "harp": True, "hm": True,
              "utility": True, "vei": True, "vei_exact": True, "network": True}


def _run_both(chunk, flags=None):
    """Run both backends on the same data.

    The per-user algorithm functions read HAS_RUST at call time and delegate to
    Rust when it is available, so calling ``_analyze_chunk_python`` while Rust is
    installed would silently compare Rust against Rust (the bug that let a CCEI
    backend divergence go unnoticed). We force HAS_RUST=False for the Python side
    so it is genuinely the pure-Python implementation.
    """
    import prefgraph._rust_backend as rb

    if flags is None:
        flags = FLAGS

    metrics = []
    if flags.get("ccei"):
        metrics.append("ccei")
    if flags.get("mpi"):
        metrics.append("mpi")
    if flags.get("harp"):
        metrics.append("harp")
    if flags.get("hm"):
        metrics.append("hm")
    if flags.get("utility"):
        metrics.append("utility")
    if flags.get("vei"):
        metrics.append("vei")
    if flags.get("vei_exact"):
        metrics.append("vei_exact")
    if flags.get("network"):
        metrics.append("network")

    engine = Engine(metrics=["garp"] + metrics)
    rust = engine._analyze_chunk_rust(chunk, flags)
    saved = rb.HAS_RUST
    rb.HAS_RUST = False
    try:
        py = engine._analyze_chunk_python(chunk, flags)
    finally:
        rb.HAS_RUST = saved
    return py, rust


def _run_both_menu(chunk, compute_warp_la=False, network=False):
    """Run both menu backends on the same data."""
    import prefgraph._rust_backend as rb

    metrics = ["garp"]
    if network:
        metrics.append("network")

    engine = Engine(metrics=metrics)

    if engine.backend == "rust" and rb._rust_analyze_menu_batch is not None:
        menus_list = [u[0] for u in chunk]
        choices_list = [u[1] for u in chunk]
        n_items_list = [u[2] for u in chunk]

        from typing import cast, Callable, Any
        analyze_menu_batch = cast(
            "Callable[..., Any]", rb._rust_analyze_menu_batch
        )
        raw = analyze_menu_batch(
            menus_list, choices_list, n_items_list,
            compute_warp_la, network,
        )
        from prefgraph.engine import MenuResult
        rust = [
            MenuResult(
                is_sarp=r["is_sarp"],
                is_warp=r["is_warp"],
                is_warp_la=r.get("is_warp_la", False),
                n_sarp_violations=r["n_sarp_violations"],
                n_warp_violations=r["n_warp_violations"],
                hm_consistent=r["hm_consistent"],
                hm_total=r["hm_total"],
                max_scc=r["max_scc"],
                n_scc=r.get("n_scc", 0),
                r_density=r.get("r_density", 0.0),
                pref_entropy=r.get("pref_entropy", 0.0),
                choice_diversity=r.get("choice_diversity", 0.0),
                compute_time_us=r["compute_time_us"],
            )
            for r in raw
        ]
    else:
        pytest.skip("Rust menu backend not available")
        rust = []

    saved = rb.HAS_RUST
    rb.HAS_RUST = False
    try:
        py = engine.analyze_menus(chunk, compute_warp_la=compute_warp_la)
    finally:
        rb.HAS_RUST = saved
    return py, rust


# --- Fixtures ---

@pytest.fixture
def consistent_data():
    """GARP-consistent: budget line rotation, no violations."""
    p = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
    q = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    return [(p, q)]


@pytest.fixture
def violation_data():
    """Random data (seed=123) that produces GARP violations.

    Varied prices + random quantities create strict revealed preferences
    that form cycles - unlike equal-price data where all expenditures match.
    """
    rng = np.random.RandomState(123)
    p = (rng.rand(20, 3) + 0.1).astype(np.float64)
    q = (rng.rand(20, 3) * 10).astype(np.float64)
    return [(p, q)]


@pytest.fixture
def random_users():
    """100 random users for statistical parity checking."""
    rng = np.random.RandomState(42)
    users = []
    for _ in range(100):
        p = (rng.rand(10, 5) + 0.1).astype(np.float64)
        q = (rng.rand(10, 5) + 0.1).astype(np.float64)
        users.append((p, q))
    return users


@pytest.fixture
def network_users():
    """100 random users with larger T for network stat parity."""
    rng = np.random.RandomState(7)
    users = []
    for _ in range(100):
        t = rng.randint(5, 15)
        p = (rng.rand(t, 3) + 0.1).astype(np.float64)
        q = (rng.rand(t, 3) * 10 + 0.1).astype(np.float64)
        users.append((p, q))
    return users


# --- GARP parity ---

class TestGARPParity:
    def test_consistent(self, consistent_data):
        py, rust = _run_both(consistent_data)
        assert py[0].is_garp == rust[0].is_garp is True

    def test_violation(self, violation_data):
        py, rust = _run_both(violation_data)
        assert py[0].is_garp == rust[0].is_garp is False

    def test_random_match(self, random_users):
        py, rust = _run_both(random_users)
        for i, (p, r) in enumerate(zip(py, rust)):
            assert p.is_garp == r.is_garp, f"User {i}: py={p.is_garp}, rust={r.is_garp}"


# --- CCEI parity ---

class TestCCEIParity:
    def test_consistent_is_one(self, consistent_data):
        py, rust = _run_both(consistent_data)
        assert abs(py[0].ccei - rust[0].ccei) < 0.01
        assert py[0].ccei == pytest.approx(1.0, abs=0.01)

    def test_violation(self, violation_data):
        py, rust = _run_both(violation_data)
        assert abs(py[0].ccei - rust[0].ccei) < 0.01

    def test_random_close(self, random_users):
        py, rust = _run_both(random_users)
        for i, (p, r) in enumerate(zip(py, rust)):
            assert abs(p.ccei - r.ccei) < 0.01, (
                f"User {i}: py_ccei={p.ccei:.4f}, rust_ccei={r.ccei:.4f}"
            )


# --- MPI parity ---

class TestMPIParity:
    def test_consistent_is_zero(self, consistent_data):
        py, rust = _run_both(consistent_data)
        assert abs(py[0].mpi - rust[0].mpi) < 1e-6
        assert py[0].mpi == pytest.approx(0.0, abs=0.01)

    def test_violation(self, violation_data):
        py, rust = _run_both(violation_data)
        assert abs(py[0].mpi - rust[0].mpi) < 1e-6

    def test_random_close(self, random_users):
        # Both backends now compute the same min-cycle-ratio MPI, so they agree
        # tightly. The old Karp-vs-cycle-enumeration divergence is fixed.
        py, rust = _run_both(random_users)
        for i, (p, r) in enumerate(zip(py, rust)):
            assert abs(p.mpi - r.mpi) < 1e-6, (
                f"User {i}: py_mpi={p.mpi:.6f}, rust_mpi={r.mpi:.6f}"
            )


# --- SCC stats parity ---

class TestSCCParity:
    """SCC stats are always computed (no flag gate) - exact integer match expected."""

    def test_consistent_scc(self, consistent_data):
        # Consistent 2-obs data: each observation is its own SCC (no cycles)
        py, rust = _run_both(consistent_data, FLAGS_FULL)
        assert py[0].max_scc == rust[0].max_scc, (
            f"max_scc: py={py[0].max_scc}, rust={rust[0].max_scc}"
        )
        assert py[0].n_scc == rust[0].n_scc, (
            f"n_scc: py={py[0].n_scc}, rust={rust[0].n_scc}"
        )
        assert abs(py[0].scc_mean_size - rust[0].scc_mean_size) < 1e-10

    def test_violation_scc(self, violation_data):
        py, rust = _run_both(violation_data, FLAGS_FULL)
        assert py[0].max_scc == rust[0].max_scc, (
            f"max_scc: py={py[0].max_scc}, rust={rust[0].max_scc}"
        )
        assert py[0].n_scc == rust[0].n_scc, (
            f"n_scc: py={py[0].n_scc}, rust={rust[0].n_scc}"
        )
        assert abs(py[0].scc_mean_size - rust[0].scc_mean_size) < 1e-10

    def test_random_scc_exact(self, random_users):
        py, rust = _run_both(random_users, FLAGS_FULL)
        for i, (p, r) in enumerate(zip(py, rust)):
            assert p.max_scc == r.max_scc, (
                f"User {i} max_scc: py={p.max_scc}, rust={r.max_scc}"
            )
            assert p.n_scc == r.n_scc, (
                f"User {i} n_scc: py={p.n_scc}, rust={r.n_scc}"
            )
            assert abs(p.scc_mean_size - r.scc_mean_size) < 1e-10, (
                f"User {i} scc_mean_size: py={p.scc_mean_size}, rust={r.scc_mean_size}"
            )


# --- Network stats parity ---

class TestNetworkParity:
    """Network stats parity with flags network=True, harp=True."""

    def test_consistent_network(self, consistent_data):
        py, rust = _run_both(consistent_data, FLAGS_FULL)
        assert abs(py[0].r_density - rust[0].r_density) < 1e-8
        assert abs(py[0].r_out_degree_std - rust[0].r_out_degree_std) < 1e-8
        assert abs(py[0].degree_gini - rust[0].degree_gini) < 1e-8

    def test_violation_network(self, violation_data):
        py, rust = _run_both(violation_data, FLAGS_FULL)
        assert abs(py[0].r_density - rust[0].r_density) < 1e-8, (
            f"r_density: py={py[0].r_density:.6f}, rust={rust[0].r_density:.6f}"
        )
        assert abs(py[0].r_out_degree_std - rust[0].r_out_degree_std) < 1e-8
        assert abs(py[0].degree_gini - rust[0].degree_gini) < 1e-8

    def test_violation_ew(self, violation_data):
        """Edge-weight stats only populated when both harp and network are set."""
        py, rust = _run_both(violation_data, FLAGS_FULL)
        assert abs(py[0].ew_mean - rust[0].ew_mean) < 1e-6, (
            f"ew_mean: py={py[0].ew_mean:.6f}, rust={rust[0].ew_mean:.6f}"
        )
        assert abs(py[0].ew_std - rust[0].ew_std) < 1e-6, (
            f"ew_std: py={py[0].ew_std:.6f}, rust={rust[0].ew_std:.6f}"
        )
        assert abs(py[0].ew_skew - rust[0].ew_skew) < 1e-4, (
            f"ew_skew: py={py[0].ew_skew:.6f}, rust={rust[0].ew_skew:.6f}"
        )

    def test_ew_zero_without_harp_flag(self, violation_data):
        """Without harp flag, ew_* should be 0.0 in both backends."""
        flags_no_harp = dict(FLAGS_FULL)
        flags_no_harp["harp"] = False
        py, rust = _run_both(violation_data, flags_no_harp)
        assert py[0].ew_mean == 0.0
        assert rust[0].ew_mean == 0.0

    def test_random_network_close(self, network_users):
        """Bulk parity check for 100 users with varied T."""
        py, rust = _run_both(network_users, FLAGS_FULL)
        max_density_err = max(abs(p.r_density - r.r_density)
                              for p, r in zip(py, rust))
        max_gini_err = max(abs(p.degree_gini - r.degree_gini)
                           for p, r in zip(py, rust))
        max_ew_mean_err = max(abs(p.ew_mean - r.ew_mean)
                              for p, r in zip(py, rust))
        assert max_density_err < 1e-8, f"max r_density error: {max_density_err}"
        assert max_gini_err < 1e-8, f"max degree_gini error: {max_gini_err}"
        assert max_ew_mean_err < 1e-6, f"max ew_mean error: {max_ew_mean_err}"


# --- VEI stats parity ---

class TestVEIParity:
    """VEI stats (std, q25, q75) parity.  Tolerance 0.05 - different LP backends."""

    def test_consistent_vei_defaults(self, consistent_data):
        """Consistent data: all VEI fields should be at their defaults."""
        py, rust = _run_both(consistent_data, FLAGS_FULL)
        assert py[0].vei_mean == pytest.approx(1.0, abs=1e-6)
        assert rust[0].vei_mean == pytest.approx(1.0, abs=1e-6)

    def test_violation_vei_mean(self, violation_data):
        py, rust = _run_both(violation_data, FLAGS_FULL)
        # VEI for violating data should be < 1.0 in both backends
        assert py[0].vei_mean < 1.0, f"Python VEI mean unexpectedly 1.0"
        assert rust[0].vei_mean < 1.0, f"Rust VEI mean unexpectedly 1.0"
        assert abs(py[0].vei_mean - rust[0].vei_mean) < 0.05, (
            f"vei_mean: py={py[0].vei_mean:.4f}, rust={rust[0].vei_mean:.4f}"
        )

    def test_violation_vei_stats(self, violation_data):
        """vei_std, vei_q25, vei_q75 - both backends should agree within 0.05."""
        py, rust = _run_both(violation_data, FLAGS_FULL)
        assert abs(py[0].vei_std - rust[0].vei_std) < 0.05, (
            f"vei_std: py={py[0].vei_std:.4f}, rust={rust[0].vei_std:.4f}"
        )
        assert abs(py[0].vei_q25 - rust[0].vei_q25) < 0.05, (
            f"vei_q25: py={py[0].vei_q25:.4f}, rust={rust[0].vei_q25:.4f}"
        )
        assert abs(py[0].vei_q75 - rust[0].vei_q75) < 0.05, (
            f"vei_q75: py={py[0].vei_q75:.4f}, rust={rust[0].vei_q75:.4f}"
        )

    def test_harp_severity_always_one(self, random_users):
        """harp_severity is always 1.0 in both backends (no severity metric exists)."""
        py, rust = _run_both(random_users, FLAGS_FULL)
        for i, (p, r) in enumerate(zip(py, rust)):
            assert p.harp_severity == 1.0, f"User {i}: Python harp_severity={p.harp_severity}"
            assert r.harp_severity == 1.0, f"User {i}: Rust harp_severity={r.harp_severity}"


# --- Menu parity ---

@pytest.fixture
def consistent_menu_data():
    """SARP-consistent menu choices: transitive preferences."""
    # 3 items, item 0 > item 1 > item 2
    menus = [[0, 1, 2], [0, 1], [1, 2]]
    choices = [0, 0, 1]
    n_items = 3
    return [(menus, choices, n_items)]


@pytest.fixture
def violation_menu_data():
    """SARP-violating menu choices: intransitive cycle."""
    # 3 items: 0 > 1 > 2 > 0 (cycle)
    menus = [[0, 1], [1, 2], [0, 2]]
    choices = [0, 1, 2]
    n_items = 3
    return [(menus, choices, n_items)]


@pytest.fixture
def random_menu_users():
    """100 random menu users for bulk parity checking."""
    rng = np.random.RandomState(99)
    users = []
    for _ in range(100):
        n_items = rng.randint(3, 8)
        n_obs = rng.randint(5, 15)
        menus = []
        choices = []
        for _ in range(n_obs):
            menu_size = rng.randint(2, min(n_items + 1, 5))
            menu = list(rng.choice(n_items, size=menu_size, replace=False))
            choice = menu[rng.randint(len(menu))]
            menus.append(menu)
            choices.append(choice)
        users.append((menus, choices, n_items))
    return users


class TestMenuParity:
    """Menu backend parity: Python fallback vs Rust."""

    def test_consistent_sarp_warp(self, consistent_menu_data):
        py, rust = _run_both_menu(consistent_menu_data)
        assert py[0].is_sarp == rust[0].is_sarp
        assert py[0].is_warp == rust[0].is_warp

    def test_violation_sarp_warp(self, violation_menu_data):
        py, rust = _run_both_menu(violation_menu_data)
        assert py[0].is_sarp == rust[0].is_sarp
        assert py[0].is_warp == rust[0].is_warp

    def test_scc_consistent(self, consistent_menu_data):
        py, rust = _run_both_menu(consistent_menu_data)
        assert py[0].max_scc == rust[0].max_scc, (
            f"max_scc: py={py[0].max_scc}, rust={rust[0].max_scc}"
        )
        assert py[0].n_scc == rust[0].n_scc, (
            f"n_scc: py={py[0].n_scc}, rust={rust[0].n_scc}"
        )

    def test_scc_violation(self, violation_menu_data):
        py, rust = _run_both_menu(violation_menu_data)
        assert py[0].max_scc == rust[0].max_scc, (
            f"max_scc: py={py[0].max_scc}, rust={rust[0].max_scc}"
        )
        assert py[0].n_scc == rust[0].n_scc

    def test_network_violation(self, violation_menu_data):
        py, rust = _run_both_menu(violation_menu_data, network=True)
        assert abs(py[0].r_density - rust[0].r_density) < 1e-8, (
            f"r_density: py={py[0].r_density:.6f}, rust={rust[0].r_density:.6f}"
        )
        assert abs(py[0].pref_entropy - rust[0].pref_entropy) < 1e-8, (
            f"pref_entropy: py={py[0].pref_entropy:.6f}, rust={rust[0].pref_entropy:.6f}"
        )
        assert abs(py[0].choice_diversity - rust[0].choice_diversity) < 1e-8

    def test_warp_la_consistent(self, consistent_menu_data):
        """WARP-LA flag must be honored; consistent data should satisfy WARP-LA."""
        py, rust = _run_both_menu(consistent_menu_data, compute_warp_la=True)
        assert py[0].is_warp_la == rust[0].is_warp_la, (
            f"is_warp_la: py={py[0].is_warp_la}, rust={rust[0].is_warp_la}"
        )

    def test_warp_la_not_computed_when_flag_false(self, violation_menu_data):
        """Without compute_warp_la=True, is_warp_la should remain False."""
        py, rust = _run_both_menu(violation_menu_data, compute_warp_la=False)
        assert py[0].is_warp_la is False
        assert rust[0].is_warp_la is False

    def test_random_menu_bulk(self, random_menu_users):
        """Bulk parity for 100 random menu users."""
        py, rust = _run_both_menu(random_menu_users, network=True)
        for i, (p, r) in enumerate(zip(py, rust)):
            assert p.is_sarp == r.is_sarp, f"User {i} is_sarp mismatch"
            assert p.is_warp == r.is_warp, f"User {i} is_warp mismatch"
            assert p.max_scc == r.max_scc, (
                f"User {i} max_scc: py={p.max_scc}, rust={r.max_scc}"
            )
            assert p.n_scc == r.n_scc, (
                f"User {i} n_scc: py={p.n_scc}, rust={r.n_scc}"
            )
            assert abs(p.r_density - r.r_density) < 1e-8, (
                f"User {i} r_density: py={p.r_density:.6f}, rust={r.r_density:.6f}"
            )
