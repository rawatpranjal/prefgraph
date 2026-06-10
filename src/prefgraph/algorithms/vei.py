"""Varian's Efficiency Index (VEI) - per-observation efficiency scores."""

from __future__ import annotations

import time
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

from prefgraph.core.session import ConsumerSession
from prefgraph.core.result import VEIResult
from prefgraph.core.exceptions import SolverError, OptimizationError


def compute_vei(
    session: ConsumerSession,
    tolerance: float = 1e-8,
    efficiency_threshold: float = 0.9,
) -> VEIResult:
    """
    Per-observation efficiency lower bounds (a fast VEI approximation).

    Unlike AEI, which finds a single global efficiency e for all observations,
    this returns an efficiency score e_i per observation. It solves a relaxation:
    for each observation it returns the tightest e_i consistent with that
    observation's DIRECT revealed-preference relations,
        e_i = max_j { (p_i @ x_j) / (p_i @ x_i) : x_i directly preferred to x_j },
    a lower bound on the per-observation efficiency and a fast way to flag
    problematic observations. It is NOT the exact Varian (1990) index: it does
    not re-impose GARP at the deflated budgets, and the code minimizes sum(e_i)
    rather than the index's sum(1 - e_i). For the exact per-observation index
    use compute_vei_exact (Mononen 2023 Theorem 1, implemented identically in
    Rust and in this module).

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: Numerical tolerance for optimization (default: 1e-8)
        efficiency_threshold: Threshold below which observations are flagged
            as problematic (default: 0.9)

    Returns:
        VEIResult with per-observation efficiency vector and summary statistics

    Example:
        >>> import numpy as np
        >>> from prefgraph import ConsumerSession, compute_vei
        >>> prices = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
        >>> quantities = np.array([[4.0, 1.0], [1.0, 4.0], [2.0, 2.0]])
        >>> session = ConsumerSession(prices=prices, quantities=quantities)
        >>> result = compute_vei(session)
        >>> print(f"Worst observation: {result.worst_observation}")
        >>> print(f"Min efficiency: {result.min_efficiency:.3f}")

    References:
        Varian, H. R. (1990). Goodness-of-fit in optimizing models.
        Journal of Econometrics, 46(1-2), 125-140.
        Mononen, L. (2023). Computing measures of rationality (exact VEI via
        weighted feedback arc set; see vei_exact).
    """
    start_time = time.perf_counter()

    from prefgraph.algorithms.garp import check_garp

    T = session.num_observations
    E = session.expenditure_matrix  # T x T
    own_exp = session.own_expenditures  # T

    # GARP check gates computation: consistent data has e_i = 1.0 for all i
    # by definition - no LP needed. Varian (1982), Econometrica 50(4), 945-972.
    garp_result = check_garp(session)

    if garp_result.is_consistent:
        computation_time = (time.perf_counter() - start_time) * 1000
        return VEIResult(
            efficiency_vector=np.ones(T),
            mean_efficiency=1.0,
            min_efficiency=1.0,
            worst_observation=0,
            problematic_observations=[],
            total_inefficiency=0.0,
            optimization_success=True,
            optimization_status="Data is perfectly consistent",
            computation_time_ms=computation_time,
        )

    # --------------------------------------------------------------------------
    # VEI LP formulation - Varian (1990) "Goodness-of-fit in optimizing models",
    # J. Econometrics 46(1-2), 125-140.
    #
    # Smeulders, Cherchye, De Rock & Spieksma (2014) "Goodness-of-Fit Measures
    # for Revealed Preference Tests", ACM Trans. Econ. Comp. 2(1), Art. 3,
    # define VEI (Section 2.2):
    #
    #   "The Varian index (VI) differs from AI by allowing for observation-
    #    specific perturbations. VI equals the vector e that is closest to one,
    #    for some given norm, such that the data satisfies the revealed-
    #    preference axiom under study."
    #
    # The relaxed revealed-preference relation R^0(e) is (Section 2.2):
    #   q_i R^0(e) q_j  iff  e_i * p_i · q_i >= p_i · q_j
    #
    # For each (i,j) where i directly revealed-prefers j (R[i,j] = True),
    # the efficiency-adjusted budget must still cover j's bundle:
    #   e_i >= (p_i · q_j) / (p_i · q_i) = E[i,j] / own_exp[i]
    #
    # We use direct R (not the transitive closure R*) because:
    #   - For direct pairs, E[i,j]/own_exp[i] <= 1 by definition (LP always feasible).
    #   - Transitivity is implicit: if each direct link is maintained under e,
    #     the transitive closure R*(e) follows.
    #   - Using R* would include transitive pairs where E[i,j] > own_exp[i],
    #     making the LP infeasible.
    #
    # The LP minimizes sum(e_i), finding the tightest per-observation efficiency
    # that maintains all direct revealed-preference links. Lower e_i means
    # observation i's budget is more "wasted" - the observation is less efficient.
    #
    # NP-hardness: Smeulders et al. (2014) Theorem 4.2 shows VI-GARP_d is
    # NP-complete via reduction from independent set. This LP is an approximation
    # using direct-R constraints only.
    # --------------------------------------------------------------------------

    # Build direct revealed-preference relation R (not transitive closure).
    # R[i,j] = True iff own_exp[i] >= E[i,j], i.e. i could afford j's bundle.
    R_direct = own_exp[:, np.newaxis] >= E - tolerance
    np.fill_diagonal(R_direct, False)

    # Collect per-observation lower-bound constraints from direct R pairs.
    # Each constraint says: e_i >= E[i,j] / own_exp[i]  (the efficiency ratio).
    # Rearranged for linprog: -e_i <= -E[i,j] / own_exp[i].
    A_ub_list = []
    b_ub_list = []

    for i in range(T):
        for j in range(T):
            if i == j:
                continue
            if R_direct[i, j]:
                row = np.zeros(T)
                row[i] = -1.0
                A_ub_list.append(row)
                b_ub_list.append(-E[i, j] / own_exp[i])

    if A_ub_list:
        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)
    else:
        A_ub = np.zeros((0, T))
        b_ub = np.zeros(0)

    # Objective: minimize sum(e_i).
    # Each e_i is pushed down to its tightest lower bound from the constraints.
    # Observations involved in GARP violations will have e_i < 1.0.
    # The old code used c = -np.ones(T) (maximize), which trivially returned
    # e_i = 1.0 for all observations because all lower bounds are <= 1.
    c = np.ones(T)

    # Bounds: 0 <= e_i <= 1
    bounds = [(0.0, 1.0) for _ in range(T)]

    try:
        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )
        if result.success:
            success = True
            status = result.message
            # On success scipy guarantees ``x`` is populated; the stub types it
            # as ``ndarray | None`` so narrow it here without runtime effect.
            e_vector = cast(NDArray[np.float64], result.x)
        else:
            raise SolverError(
                f"LP solver failed to compute VEI. Status: {result.status}, "
                f"Message: {result.message}"
            )
    except SolverError:
        raise
    except Exception as e:
        raise SolverError(
            f"LP solver failed during VEI computation. Original error: {e}"
        ) from e

    # Clip to [0, 1] for numerical stability
    e_vector = np.clip(e_vector, 0.0, 1.0)

    # Compute summary statistics
    mean_eff = float(np.mean(e_vector))
    min_eff = float(np.min(e_vector))
    worst_obs = int(np.argmin(e_vector))
    problematic = [i for i in range(T) if e_vector[i] < efficiency_threshold]
    total_ineff = float(np.sum(1.0 - e_vector))

    computation_time = (time.perf_counter() - start_time) * 1000

    return VEIResult(
        efficiency_vector=e_vector,
        mean_efficiency=mean_eff,
        min_efficiency=min_eff,
        worst_observation=worst_obs,
        problematic_observations=problematic,
        total_inefficiency=total_ineff,
        optimization_success=success,
        optimization_status=status,
        computation_time_ms=computation_time,
    )


# =============================================================================
# Exact VEI: Mononen (2023) Theorem 1 with the canonical-vector convention.
# This is a line-for-line mirror of rust/crates/rpt-core/src/vei.rs
# compute_vei_exact; any change here must be made there too (and vice versa)
# so cross-backend parity holds exactly on discrete data.
# =============================================================================

# Tolerance below which an arc removal cost is treated as zero. Mononen
# (2023, p. 10): non-strict preferences are removable at zero cost, so only
# strict arcs with a real cost participate. The same tolerance is the AddCost
# survival test (an arc survives adjustment d_i iff cost - d_i > tol) and the
# U-set inclusion tolerance.
_VEI_COST_TOL = 1e-12

# Slack on caps between the sequential canonical-vector solves so the prior
# incumbent stays feasible under float summation noise. Grid gaps on real
# data are orders of magnitude larger.
_VEI_CAP_EPS = 1e-9

# Hard cap on MILP solves across all stages; hitting it raises SolverError,
# never a silently truncated answer.
_VEI_MAX_SOLVES = 1000


def compute_vei_exact(
    session: ConsumerSession,
    tolerance: float = 1e-10,
    efficiency_threshold: float = 0.9,
) -> VEIResult:
    """
    Exact Varian index (Mononen 2023, Theorem 1) with a canonical vector.

    Varian's index relaxes the revealed preference with observation-specific
    adjustments d_t in [0, 1] (efficiency e_t = 1 - d_t) and is the least
    average adjustment that makes the relaxed strict preference acyclic
    (Mononen 2023, p. 9). Theorem 1 (p. 11) reformulates it as a binary LP
    with one variable per strict arc, objective min sum(theta_ij * cost_ij)
    where cost_ij = 1 - (p_i.x_j)/(p_i.x_i), and for every strict cycle a
    U-set expanded covering row: a cycle arc (a, b) is covered by ANY arc out
    of a costing at least cost_ab, because one budget adjustment removes that
    preference "and all the cheaper revealed preferences" (p. 10). Rows are
    generated lazily (Algorithm 2) with a DFS separation oracle that keeps an
    arc alive iff its AddCost = cost_ij - d_i is positive and greedily breaks
    each found cycle at its minimum AddCost arc (Algorithm 1).

    The optimal adjustment vector is not unique under ties, so PrefGraph
    reports a canonical vector: among value-optimal solutions, first minimize
    the maximum adjustment (stage B), then minimize each d_i in observation
    order (stage C, lexicographic). All reported numbers derive from the
    binary incumbent, never raw solver values, so the Rust backend and this
    mirror agree exactly on discrete data.

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: strict-preference tolerance, matching the Rust
            parse_budget default (default: 1e-10)
        efficiency_threshold: threshold below which observations are flagged
            as problematic (default: 0.9)

    Returns:
        VEIResult with the canonical per-observation efficiency vector

    Raises:
        SolverError: if a MILP solve fails or the solve cap is hit

    References:
        Varian, H. R. (1990). Goodness-of-fit in optimizing models.
        Journal of Econometrics, 46(1-2), 125-140.
        Mononen, L. (2023). Computing and comparing measures of rationality.
    """
    start_time = time.perf_counter()

    T = session.num_observations
    E = session.expenditure_matrix
    own_exp = session.own_expenditures

    if T == 0:
        return _vei_exact_result(np.ones(0), True, "Empty session", start_time, [])

    # Strict preference arcs with costs. Arc (i, j): own[i] - E[i,j] > tol,
    # cost = 1 - E[i,j]/own[i], dropping sub-tolerance costs (see above).
    arc_from: list[int] = []
    arc_to: list[int] = []
    arc_cost: list[float] = []
    arc_ratio: list[float] = []
    for i in range(T):
        if own_exp[i] <= 0.0:
            continue
        for j in range(T):
            if i != j and own_exp[i] - E[i, j] > tolerance:
                ratio = E[i, j] / own_exp[i]
                cost = 1.0 - ratio
                if cost > _VEI_COST_TOL:
                    arc_from.append(i)
                    arc_to.append(j)
                    arc_cost.append(cost)
                    arc_ratio.append(ratio)

    n_arcs = len(arc_from)
    if n_arcs == 0:
        return _vei_exact_result(
            np.ones(T), True, "No strict preference arcs", start_time, []
        )

    # Adjacency for the DFS oracle and per-observation arc lists sorted by
    # cost DESCENDING (ties by arc index, stable) so each U-set is a prefix.
    # The Rust backend builds the identical order.
    adj: list[list[tuple[int, int]]] = [[] for _ in range(T)]
    for idx in range(n_arcs):
        adj[arc_from[idx]].append((arc_to[idx], idx))
    arcs_of_obs: list[list[int]] = [[] for _ in range(T)]
    for idx in range(n_arcs):
        arcs_of_obs[arc_from[idx]].append(idx)
    for lst in arcs_of_obs:
        lst.sort(key=lambda a: (-arc_cost[a], a))

    # Seed cycles: all 2-cycles (Algorithm 2 first step); if none, search for
    # longer cycles from the zero solution (Algorithm 2 fallback).
    arc_lookup: dict[tuple[int, int], int] = {
        (arc_from[idx], arc_to[idx]): idx for idx in range(n_arcs)
    }
    seed_cycles: list[list[int]] = []
    for idx in range(n_arcs):
        i, j = arc_from[idx], arc_to[idx]
        if i < j and (j, i) in arc_lookup:
            seed_cycles.append([idx, arc_lookup[(j, i)]])
    if not seed_cycles:
        seed_cycles = _vei_separation(adj, arc_from, arc_cost, [0.0] * T, n_arcs, T)
        if not seed_cycles:
            # The strict-arc graph is acyclic: every violation runs through
            # zero-cost arcs, so the infimum adjustment is zero everywhere.
            return _vei_exact_result(
                np.ones(T), True, "Strict-arc graph is acyclic", start_time, []
            )

    rows: set[tuple[int, ...]] = set()
    for cyc in seed_cycles:
        rows.add(_vei_u_expand(cyc, arc_from, arc_cost, arcs_of_obs))

    solves = 0

    def _solve_verified(
        budget: float | None,
        minimize_max: bool,
        objective_obs: int | None,
        caps: list[tuple[int, float]],
    ) -> list[bool]:
        """Solve one stage to separation-verified optimality, adding rows."""
        nonlocal solves
        while True:
            solves += 1
            if solves > _VEI_MAX_SOLVES:
                raise SolverError(
                    f"Exact VEI row generation exceeded {_VEI_MAX_SOLVES} solves"
                )
            theta = _vei_solve_stage(
                arc_cost,
                arc_from,
                arcs_of_obs,
                rows,
                budget,
                minimize_max,
                objective_obs,
                caps,
            )
            d = _vei_d_from_theta(theta, arc_from, arc_cost, T)
            new_cycles = _vei_separation(adj, arc_from, arc_cost, d, n_arcs, T)
            if not new_cycles:
                return theta
            for cyc in new_cycles:
                rows.add(_vei_u_expand(cyc, arc_from, arc_cost, arcs_of_obs))

    # Stage A: the Theorem 1 value.
    incumbent = _solve_verified(None, False, None, [])
    d_star = _vei_d_from_theta(incumbent, arc_from, arc_cost, T)
    v_star = sum(d_star)
    # Float-summation slack only: grid totals differ by far more, so the
    # budget admits exactly the value-optimal solutions.
    budget = v_star + 1e-9 * max(1.0, v_star)

    # Stage B: among optima, minimize the maximum adjustment.
    incumbent = _solve_verified(budget, True, None, [])
    d_inc = _vei_d_from_theta(incumbent, arc_from, arc_cost, T)
    m_b = max(d_inc) if d_inc else 0.0
    caps: list[tuple[int, float]] = [(i, m_b + _VEI_CAP_EPS) for i in range(T)]

    # Stage C: lexicographic minimization in observation order. Once d_i is
    # minimized to g_i and capped, later programs are subsets, so their d_i
    # stays in [g_i, g_i + eps] and the only grid point there is g_i: the
    # vector is pinned independent of solver vertex choice. Skip the solve
    # when the incumbent already attains the global lower bound d_i = 0.
    for i in range(T):
        if d_inc[i] <= _VEI_CAP_EPS:
            caps.append((i, _VEI_CAP_EPS))
            continue
        incumbent = _solve_verified(budget, False, i, caps)
        d_inc = _vei_d_from_theta(incumbent, arc_from, arc_cost, T)
        caps.append((i, d_inc[i] + _VEI_CAP_EPS))

    # Efficiency vector from the verified incumbent: e_i is the ratio of the
    # selected arc (exactly the grid float, not 1 - cost), or 1.0 when no arc
    # is selected. SOS rows guarantee at most one selection per observation;
    # equal-cost arcs share the same ratio, so the vector does not depend on
    # which of them the solver picked.
    efficiency = np.ones(T)
    for idx in range(n_arcs):
        if incumbent[idx] and arc_ratio[idx] < efficiency[arc_from[idx]]:
            efficiency[arc_from[idx]] = arc_ratio[idx]

    problematic = [i for i in range(T) if efficiency[i] < efficiency_threshold]
    return _vei_exact_result(efficiency, True, "Optimal", start_time, problematic)


def _vei_exact_result(
    efficiency: NDArray[np.float64],
    success: bool,
    status: str,
    start_time: float,
    problematic: list[int],
) -> VEIResult:
    t = len(efficiency)
    computation_time = (time.perf_counter() - start_time) * 1000
    return VEIResult(
        efficiency_vector=efficiency,
        mean_efficiency=float(np.mean(efficiency)) if t else 1.0,
        min_efficiency=float(np.min(efficiency)) if t else 1.0,
        worst_observation=int(np.argmin(efficiency)) if t else 0,
        problematic_observations=problematic,
        total_inefficiency=float(np.sum(1.0 - efficiency)) if t else 0.0,
        optimization_success=success,
        optimization_status=status,
        computation_time_ms=computation_time,
    )


def _vei_d_from_theta(
    theta: list[bool], arc_from: list[int], arc_cost: list[float], t: int
) -> list[float]:
    """Observation spend d_i = sum of selected arc costs at i. Under the SOS
    rows at most one arc per observation is selected, so d_i is exactly one
    cost grid float (or 0)."""
    d = [0.0] * t
    for idx, selected in enumerate(theta):
        if selected:
            d[arc_from[idx]] += arc_cost[idx]
    return d


def _vei_u_expand(
    cycle: list[int],
    arc_from: list[int],
    arc_cost: list[float],
    arcs_of_obs: list[list[int]],
) -> tuple[int, ...]:
    """U-set expansion of a cycle into a covering row (Mononen 2023, p. 10):
    every arc out of the cycle arc's observation costing at least the cycle
    arc covers it, since selecting it removes the cycle arc as one of its
    cheaper preferences. Rows are sorted tuples so duplicates dedupe."""
    cols: set[int] = set()
    for a in cycle:
        f = arc_from[a]
        ca = arc_cost[a]
        for idx in arcs_of_obs[f]:
            # Sorted descending: the U-set is the prefix with cost >= ca,
            # with the same tolerance as the survival test.
            if arc_cost[idx] >= ca - _VEI_COST_TOL:
                cols.add(idx)
            else:
                break
    return tuple(sorted(cols))


def _vei_solve_stage(
    arc_cost: list[float],
    arc_from: list[int],
    arcs_of_obs: list[list[int]],
    rows: set[tuple[int, ...]],
    budget: float | None,
    minimize_max: bool,
    objective_obs: int | None,
    caps: list[tuple[int, float]],
) -> list[bool]:
    """One MILP subproblem over the current row set, mirroring the Rust
    vei_solve_stage: binary theta per arc plus a continuous M in stage B;
    U-rows (>= 1), SOS rows (<= 1), optional value budget, stage B max rows
    (d_i - M <= 0), and stage C caps (d_i <= cap)."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    n_arcs = len(arc_cost)
    n_cols = n_arcs + (1 if minimize_max else 0)

    c = np.zeros(n_cols)
    if minimize_max:
        c[-1] = 1.0
    elif objective_obs is not None:
        for idx in arcs_of_obs[objective_obs]:
            c[idx] = arc_cost[idx]
    else:
        c[:n_arcs] = arc_cost

    a_rows: list[NDArray[np.float64]] = []
    lb: list[float] = []
    ub: list[float] = []

    # Covering rows: at least one selection per U-expanded cycle row.
    # Sorted iteration keeps the model deterministic.
    for row in sorted(rows):
        r = np.zeros(n_cols)
        r[list(row)] = 1.0
        a_rows.append(r)
        lb.append(1.0)
        ub.append(np.inf)

    # SOS rows: at most one selected arc per observation (p. 11 dominance).
    for obs_arcs in arcs_of_obs:
        if len(obs_arcs) > 1:
            r = np.zeros(n_cols)
            r[obs_arcs] = 1.0
            a_rows.append(r)
            lb.append(-np.inf)
            ub.append(1.0)

    # Value budget: restrict to (float-slack) value-optimal solutions.
    if budget is not None:
        r = np.zeros(n_cols)
        r[:n_arcs] = arc_cost
        a_rows.append(r)
        lb.append(-np.inf)
        ub.append(budget)

    # Stage B max rows: d_i <= M for every observation with arcs.
    if minimize_max:
        for obs_arcs in arcs_of_obs:
            if not obs_arcs:
                continue
            r = np.zeros(n_cols)
            for idx in obs_arcs:
                r[idx] = arc_cost[idx]
            r[-1] = -1.0
            a_rows.append(r)
            lb.append(-np.inf)
            ub.append(0.0)

    # Stage C caps: d_i <= cap for already-pinned observations.
    for obs, cap in caps:
        if not arcs_of_obs[obs]:
            continue
        r = np.zeros(n_cols)
        for idx in arcs_of_obs[obs]:
            r[idx] = arc_cost[idx]
        a_rows.append(r)
        lb.append(-np.inf)
        ub.append(cap)

    integrality = np.ones(n_cols, dtype=np.int64)
    if minimize_max:
        integrality[-1] = 0

    result = milp(
        c,
        constraints=LinearConstraint(np.array(a_rows), np.array(lb), np.array(ub)),
        integrality=integrality,
        bounds=Bounds(0.0, 1.0),
        # Exactness: the default relative MIP gap could accept a suboptimal
        # incumbent between adjacent grid totals on near-tie data.
        options={"mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise SolverError(
            f"Exact VEI MILP failed: status {result.status}, {result.message}"
        )
    return [bool(v > 0.5) for v in result.x[:n_arcs]]


def _vei_separation(
    adj: list[list[tuple[int, int]]],
    arc_from: list[int],
    arc_cost: list[float],
    d: list[float],
    n_arcs: int,
    t: int,
) -> list[list[int]]:
    """Separation oracle (Mononen 2023, Algorithm 1): find strict cycles
    surviving the adjustment d. An arc is alive iff its AddCost
    = cost_ij - d_i is positive (the selected arc itself sits at exactly 0
    and counts as removed). Each found cycle is greedily broken at its
    minimum-AddCost arc and the search continues; an empty return certifies
    optimality (Algorithm 2 termination)."""
    alive = [arc_cost[idx] - d[arc_from[idx]] > _VEI_COST_TOL for idx in range(n_arcs)]
    cycles: list[list[int]] = []
    while True:
        cycle = _vei_find_cycle(adj, alive, t)
        if cycle is None:
            return cycles
        break_arc = min(cycle, key=lambda a: (arc_cost[a] - d[arc_from[a]], a))
        alive[break_arc] = False
        cycles.append(cycle)


def _vei_find_cycle(
    adj: list[list[tuple[int, int]]], alive: list[bool], t: int
) -> list[int] | None:
    """First cycle among alive arcs as a list of arc indices, or None if the
    alive subgraph is acyclic. Colored DFS; gray nodes are exactly the
    current path, so a back edge closes a cycle."""
    color = [0] * t  # 0 white, 1 gray, 2 black
    path_arcs: list[int] = []
    path_nodes: list[int] = []

    def visit(node: int) -> list[int] | None:
        color[node] = 1
        path_nodes.append(node)
        for nxt, arc_idx in adj[node]:
            if not alive[arc_idx]:
                continue
            if color[nxt] == 1:
                pos = len(path_nodes) - 1 - path_nodes[::-1].index(nxt)
                return path_arcs[pos:] + [arc_idx]
            if color[nxt] == 0:
                path_arcs.append(arc_idx)
                found = visit(nxt)
                if found is not None:
                    return found
                path_arcs.pop()
        path_nodes.pop()
        color[node] = 2
        return None

    for start in range(t):
        if color[start] == 0:
            found = visit(start)
            if found is not None:
                return found
    return None


def compute_vei_l2(
    session: ConsumerSession,
    tolerance: float = 1e-8,
    efficiency_threshold: float = 0.9,
) -> VEIResult:
    """
    Compute VEI using L2 norm (minimize sum of squared deviations).

    This version minimizes sum((1 - e_i)^2) instead of sum(1 - e_i),
    which penalizes large deviations more than small ones.

    Uses scipy.optimize.minimize with SLSQP method.

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: Numerical tolerance for optimization
        efficiency_threshold: Threshold for flagging problematic observations

    Returns:
        VEIResult with per-observation efficiency vector
    """
    from scipy.optimize import minimize

    start_time = time.perf_counter()

    from prefgraph.algorithms.garp import check_garp

    T = session.num_observations
    E = session.expenditure_matrix
    own_exp = session.own_expenditures

    garp_result = check_garp(session)

    if garp_result.is_consistent:
        computation_time = (time.perf_counter() - start_time) * 1000
        return VEIResult(
            efficiency_vector=np.ones(T),
            mean_efficiency=1.0,
            min_efficiency=1.0,
            worst_observation=0,
            problematic_observations=[],
            total_inefficiency=0.0,
            optimization_success=True,
            optimization_status="Data is perfectly consistent",
            computation_time_ms=computation_time,
        )

    R_star = garp_result.transitive_closure

    def objective(e: NDArray) -> float:
        return float(np.sum((1.0 - e) ** 2))

    def grad(e: NDArray) -> NDArray:
        return -2.0 * (1.0 - e)

    # Build constraint functions
    constraints = []
    for i in range(T):
        for j in range(T):
            if i == j:
                continue
            if R_star[i, j]:
                bound_val = E[i, j] / own_exp[i]
                # Constraint: e[i] >= bound_val
                constraints.append(
                    {"type": "ineq", "fun": lambda e, idx=i, bv=bound_val: e[idx] - bv}
                )

    bounds = [(0.0, 1.0) for _ in range(T)]
    e0 = np.ones(T) * 0.9  # Initial guess

    try:
        # scipy-stubs models `minimize`'s fun/jac as Concatenate[_Float1D, *Any]
        # callables and constraints as a strict TypedDict. Idiomatic SLSQP usage
        # with single-argument callables and plain "ineq" dicts is correct at
        # runtime but does not satisfy those overloads (third-party stub gap).
        result = minimize(  # type: ignore[call-overload]
            objective,
            e0,
            method="SLSQP",
            jac=grad,
            bounds=bounds,
            constraints=constraints,
            options={"ftol": tolerance, "maxiter": 1000},
        )
        if result.success:
            success = True
            status = result.message
            e_vector = result.x
        else:
            raise OptimizationError(
                f"SLSQP optimization failed for VEI L2 computation. "
                f"Message: {result.message}"
            )
    except OptimizationError:
        raise
    except Exception as e:
        raise OptimizationError(
            f"Optimization failed during VEI L2 computation. Original error: {e}"
        ) from e

    e_vector = np.clip(e_vector, 0.0, 1.0)

    mean_eff = float(np.mean(e_vector))
    min_eff = float(np.min(e_vector))
    worst_obs = int(np.argmin(e_vector))
    problematic = [i for i in range(T) if e_vector[i] < efficiency_threshold]
    total_ineff = float(np.sum(1.0 - e_vector))

    computation_time = (time.perf_counter() - start_time) * 1000

    return VEIResult(
        efficiency_vector=e_vector,
        mean_efficiency=mean_eff,
        min_efficiency=min_eff,
        worst_observation=worst_obs,
        problematic_observations=problematic,
        total_inefficiency=total_ineff,
        optimization_success=success,
        optimization_status=status,
        computation_time_ms=computation_time,
    )


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

# compute_granular_integrity: Tech-friendly name for compute_vei
compute_granular_integrity = compute_vei
"""
Compute integrity scores for each individual observation.

This is the tech-friendly alias for compute_vei (Varian's Efficiency Index).

Unlike compute_integrity_score which gives one global score, this
identifies which specific observations are problematic.

Use this to:
- Find specific transactions to investigate
- Identify when user behavior changed
- Detect specific sessions with issues

Example:
    >>> from prefgraph import BehaviorLog, compute_granular_integrity
    >>> result = compute_granular_integrity(user_log)
    >>> for obs_idx in result.problematic_observations:
    ...     print(f"Investigate observation {obs_idx}")
"""

compute_granular_integrity_l2 = compute_vei_l2
"""
L2 version of compute_granular_integrity.

Penalizes large deviations more than small ones.
"""
