"""Afriat Efficiency Index (AEI/CCEI) computation.

Supports two methods:
- "discrete" (default): Binary search over the T^2 critical efficiency ratios.
  Finds the EXACT analytical CCEI with zero floating-point approximation error.
- "continuous": Legacy binary search over [0,1] interval with tolerance.
"""

from __future__ import annotations

import time
from typing import Any, Callable, cast

import numpy as np

from prefgraph.core.session import ConsumerSession
from prefgraph.core.result import AEIResult, GARPResult
from prefgraph.core.types import Cycle
from prefgraph.algorithms._budget_axioms import (
    BudgetAxiomCheck,
    check_budget_axiom_at_efficiency,
    normalize_budget_axiom,
)


def compute_aei(
    session: ConsumerSession,
    tolerance: float = 1e-6,
    max_iterations: int = 50,
    method: str = "discrete",
    axiom: str = "garp",
) -> AEIResult:
    """
    Compute Afriat Efficiency Index (CCEI).

    The AEI measures how close consumer behavior is to perfect rationality:

        AEI = sup{e in [0,1] : data satisfies the selected axiom with efficiency e}

    where efficiency e deflates budgets by factor e:
        R_e[i,j] = True iff e * (p_i @ x_i) >= p_i @ x_j

    The critical value e* is guaranteed to equal one of the T^2 efficiency
    ratios E[i,j] / own_exp[i]. The "discrete" method exploits this by
    binary searching over these exact ratios, giving the analytical CCEI
    with zero floating-point error in ~2*log2(T) GARP checks.

    Interpretation:
    - AEI = 1.0: Perfectly consistent under the selected axiom
    - AEI = 0.5: Consumer wastes ~50% of budget on inconsistent choices
    - AEI = 0.0: Completely irrational behavior

    Args:
        session: ConsumerSession with prices and quantities
        tolerance: Convergence tolerance (used by "continuous" method)
        max_iterations: Max iterations (used by "continuous" method)
        method: "discrete" (exact, default) or "continuous" (legacy)
        axiom: Budget axiom to use for the efficiency index: "garp"
            (default), "sarp", or "warp".

    Returns:
        AEIResult with efficiency index and supporting data

    Example:
        >>> import numpy as np
        >>> from prefgraph import ConsumerSession, compute_aei
        >>> prices = np.array([[1.0, 2.0], [2.0, 1.0]])
        >>> quantities = np.array([[4.0, 1.0], [1.0, 4.0]])
        >>> session = ConsumerSession(prices=prices, quantities=quantities)
        >>> result = compute_aei(session)
        >>> print(f"AEI: {result.efficiency_index:.4f}")
    """
    start_time = time.perf_counter()
    axiom = normalize_budget_axiom(axiom)

    # Try Rust backend for CCEI (binary search over T² ratios in Rust)
    from prefgraph._rust_backend import HAS_RUST, _rust_analyze_batch

    if HAS_RUST and method == "discrete" and axiom == "garp":
        try:
            import numpy as np

            p = np.ascontiguousarray(session.prices, dtype=np.float64)
            q = np.ascontiguousarray(session.quantities, dtype=np.float64)
            # Guarded by HAS_RUST above, so the callable is never None here.
            analyze_batch = cast("Callable[..., Any]", _rust_analyze_batch)
            results = analyze_batch(
                [p],
                [q],
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                tolerance,
            )
            ccei = results[0]["ccei"]
            is_consistent = results[0]["is_garp"]

            from prefgraph.algorithms.garp import check_garp

            garp_result = check_garp(session, tolerance)

            computation_time = (time.perf_counter() - start_time) * 1000
            return AEIResult(
                efficiency_index=1.0 if is_consistent else ccei,
                is_perfectly_consistent=is_consistent,
                garp_result_at_threshold=garp_result,
                binary_search_iterations=0,
                tolerance=tolerance,
                computation_time_ms=computation_time,
                axiom=axiom,
            )
        except Exception:
            pass  # Fall through to Python

    # Python fallback
    is_consistent, axiom_result = _check_axiom_at_efficiency(
        session, axiom, 1.0, tolerance=tolerance
    )

    if is_consistent:
        computation_time = (time.perf_counter() - start_time) * 1000
        return AEIResult(
            efficiency_index=1.0,
            is_perfectly_consistent=True,
            garp_result_at_threshold=axiom_result,
            binary_search_iterations=0,
            tolerance=tolerance,
            computation_time_ms=computation_time,
            axiom=axiom,
        )

    if method == "discrete":
        aei, iterations, last_result = _discrete_binary_search(session, axiom)
    else:
        aei, iterations, last_result = _continuous_binary_search(
            session, axiom, tolerance, max_iterations
        )

    if last_result is None:
        _, last_result = _check_axiom_at_efficiency(
            session, axiom, 0.0, tolerance=1e-10
        )

    computation_time = (time.perf_counter() - start_time) * 1000

    return AEIResult(
        efficiency_index=aei,
        is_perfectly_consistent=False,
        garp_result_at_threshold=last_result,
        binary_search_iterations=iterations,
        tolerance=tolerance,
        computation_time_ms=computation_time,
        axiom=axiom,
    )


def _discrete_binary_search(
    session: ConsumerSession,
    axiom: str,
) -> tuple[float, int, GARPResult | None]:
    """
    Find exact CCEI by binary search over discrete efficiency ratios.

    The critical e* that breaks/restores GARP must equal one of the T^2
    ratios E[i,j] / own_exp[i]. Binary searching over this sorted array
    gives the exact answer in ~2*log2(T) iterations.
    """
    E = session.expenditure_matrix
    own_exp = session.own_expenditures

    # Compute all T^2 efficiency ratios: e at which R_e[i,j] flips
    # R_e[i,j] = True iff e * own_exp[i] >= E[i,j]
    # Critical e for (i,j): e = E[i,j] / own_exp[i]
    ratios = E / own_exp[:, np.newaxis]

    # Filter to (0, 1) range, flatten, deduplicate, sort descending
    flat = ratios.ravel()
    mask = (flat > 0) & (flat < 1.0)
    candidates = np.unique(flat[mask])
    candidates = np.sort(candidates)[::-1]  # Descending: try high e first

    if len(candidates) == 0:
        return 0.0, 0, None

    # Binary search over sorted candidates
    lo, hi = 0, len(candidates) - 1
    iterations = 0
    best_e = 0.0
    best_result: GARPResult | None = None

    while lo <= hi:
        mid = (lo + hi) // 2
        e = float(candidates[mid])

        is_consistent, result_at_e = _check_axiom_at_breakpoint(
            session,
            axiom,
            e,
        )
        iterations += 1

        if is_consistent:
            best_e = e
            best_result = result_at_e
            # Try higher e (lower index in descending array)
            hi = mid - 1
        else:
            # Need lower e (higher index in descending array)
            lo = mid + 1

    return best_e, iterations, best_result


def _continuous_binary_search(
    session: ConsumerSession,
    axiom: str,
    tolerance: float,
    max_iterations: int,
) -> tuple[float, int, GARPResult | None]:
    """Legacy continuous binary search over [0, 1] interval."""
    e_low = 0.0
    e_high = 1.0
    iterations = 0
    last_consistent_e = 0.0
    last_consistent_result: GARPResult | None = None

    while (e_high - e_low > tolerance) and (iterations < max_iterations):
        e_mid = (e_low + e_high) / 2

        is_consistent, result_at_e = _check_axiom_at_efficiency(
            session, axiom, e_mid, tolerance=1e-10
        )

        if is_consistent:
            e_low = e_mid
            last_consistent_e = e_mid
            last_consistent_result = result_at_e
        else:
            e_high = e_mid

        iterations += 1

    return last_consistent_e, iterations, last_consistent_result


def _check_axiom_at_breakpoint(
    session: ConsumerSession,
    axiom: str,
    efficiency: float,
) -> tuple[bool, GARPResult]:
    """Check whether the CCEI supremum reaches a discrete breakpoint."""
    is_consistent, result_at_e = _check_axiom_at_efficiency(
        session,
        axiom,
        efficiency,
        tolerance=1e-10,
    )
    if is_consistent or efficiency <= 0.0:
        return is_consistent, result_at_e

    # For SARP-style weak cycles, the supremum can occur at a breakpoint even
    # when the axiom fails exactly at equality. Probe the one-sided limit from
    # below to recover the Afriat/CCEI supremum.
    probe_e = float(np.nextafter(efficiency, 0.0))
    if probe_e == efficiency:
        probe_e = max(0.0, efficiency - np.finfo(float).eps)

    probe_consistent, probe_result = _check_axiom_at_efficiency(
        session,
        axiom,
        probe_e,
        tolerance=0.0,
    )
    if probe_consistent:
        return True, probe_result

    return False, result_at_e


def _check_axiom_at_efficiency(
    session: ConsumerSession,
    axiom: str,
    efficiency: float,
    tolerance: float = 1e-10,
) -> tuple[bool, GARPResult]:
    """Check a budget axiom at e and adapt it to AEIResult's stored result."""
    result = check_budget_axiom_at_efficiency(
        session,
        axiom=axiom,
        efficiency=efficiency,
        tolerance=tolerance,
    )
    return result.is_consistent, _axiom_check_to_garp_result(result)


def _axiom_check_to_garp_result(result: BudgetAxiomCheck) -> GARPResult:
    """Store an axiom-at-e check in the existing AEIResult result slot."""
    violations: list[Cycle]
    if result.axiom == "warp":
        violations = [(int(i), int(j), int(i)) for i, j in result.violations]
    else:
        violations = result.violations  # type: ignore[assignment]

    transitive_closure = result.transitive_closure
    if transitive_closure is None:
        transitive_closure = result.direct_revealed_preference

    return GARPResult(
        is_consistent=result.is_consistent,
        violations=violations,
        direct_revealed_preference=result.direct_revealed_preference,
        transitive_closure=transitive_closure,
        strict_revealed_preference=result.strict_revealed_preference,
        computation_time_ms=0.0,
    )


def _check_garp_at_efficiency(
    session: ConsumerSession,
    efficiency: float,
    tolerance: float = 1e-10,
) -> tuple[bool, GARPResult]:
    """
    Check GARP at a given efficiency level e.

    Modified revealed preference relation:
        R_e[i,j] = True iff e * (p_i @ x_i) >= p_i @ x_j

    Args:
        session: ConsumerSession
        efficiency: Efficiency parameter e in [0, 1]
        tolerance: Numerical tolerance

    Returns:
        Tuple of (is_consistent, GARPResult)
    """
    return _check_axiom_at_efficiency(
        session,
        axiom="garp",
        efficiency=efficiency,
        tolerance=tolerance,
    )


def compute_varian_index(
    session: ConsumerSession,
    tolerance: float = 1e-6,
    max_iterations: int = 50,
) -> float:
    """
    Compute Varian's index of efficiency (alternative to Afriat's).

    Varian's index finds the smallest uniform efficiency e such that
    all observations can be rationalized. It is equivalent to AEI for
    most practical purposes.

    Args:
        session: ConsumerSession
        tolerance: Convergence tolerance
        max_iterations: Maximum binary search iterations

    Returns:
        Efficiency index in [0, 1]

    Note:
        This is functionally equivalent to compute_aei but included for
        completeness as referenced in the literature.
    """
    return compute_aei(session, tolerance, max_iterations).efficiency_index


# =============================================================================
# TECH-FRIENDLY ALIASES
# =============================================================================

# compute_integrity_score: Tech-friendly name for compute_aei
compute_integrity_score = compute_aei
"""
Compute the behavioral integrity score (0-1).

This is the tech-friendly alias for compute_aei (Afriat Efficiency Index).

The integrity score measures consistency with utility maximization:
- 1.0 = Perfectly consistent behavior
- 0.9+ = Minor deviations from rationality
- 0.7-0.9 = Moderate inconsistencies
- <0.7 = Notable violations of rationality

Example:
    >>> from prefgraph import BehaviorLog, compute_integrity_score
    >>> result = compute_integrity_score(user_log)
    >>> print(f"Integrity: {result.efficiency_index:.2f}")

Returns:
    IntegrityResult with efficiency_index in [0, 1]
"""

compute_ccei = compute_aei
"""Compatibility alias: CCEI and AEI are the same efficiency index."""
