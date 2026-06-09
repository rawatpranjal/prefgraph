"""Shared budget revealed-preference checks at an efficiency level."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from prefgraph._kernels import bfs_find_path_numba
from prefgraph.core.session import ConsumerSession
from prefgraph.core.types import Cycle
from prefgraph.graph.transitive_closure import floyd_warshall_transitive_closure


@dataclass(frozen=True)
class BudgetAxiomCheck:
    """Internal result for WARP/SARP/GARP checks at efficiency e."""

    axiom: str
    is_consistent: bool
    violations: list[Cycle] | list[tuple[int, int]]
    direct_revealed_preference: NDArray[np.bool_]
    transitive_closure: NDArray[np.bool_] | None
    strict_revealed_preference: NDArray[np.bool_]


def normalize_budget_axiom(axiom: str) -> str:
    """Normalize and validate a budget revealed-preference axiom name."""
    normalized = axiom.lower().strip()
    if normalized not in {"warp", "sarp", "garp"}:
        raise ValueError("axiom must be one of 'warp', 'sarp', or 'garp'")
    return normalized


def validate_efficiency_level(efficiency: float) -> float:
    """Validate an Afriat-style efficiency level."""
    e = float(efficiency)
    if not 0.0 <= e <= 1.0:
        raise ValueError("efficiency must be between 0 and 1")
    return e


def build_efficiency_relations(
    session: ConsumerSession,
    efficiency: float = 1.0,
    tolerance: float = 1e-10,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Build weak and strict direct revealed-preference matrices at e."""
    e = validate_efficiency_level(efficiency)
    expenditures = session.expenditure_matrix
    own_expenditures = session.own_expenditures

    weak = e * own_expenditures[:, np.newaxis] >= expenditures - tolerance
    strict = e * own_expenditures[:, np.newaxis] > expenditures + tolerance
    np.fill_diagonal(strict, False)
    return weak, strict


def check_budget_axiom_at_efficiency(
    session: ConsumerSession,
    axiom: str,
    efficiency: float = 1.0,
    tolerance: float = 1e-10,
) -> BudgetAxiomCheck:
    """Check WARP, SARP, or GARP using efficiency-adjusted budgets."""
    normalized = normalize_budget_axiom(axiom)
    weak, strict = build_efficiency_relations(session, efficiency, tolerance)

    if normalized == "warp":
        violation_matrix = weak & strict.T
        violations: list[Cycle] = [
            (int(i), int(j)) for i, j in np.argwhere(violation_matrix) if i < j
        ]
        return BudgetAxiomCheck(
            axiom=normalized,
            is_consistent=len(violations) == 0,
            violations=violations,
            direct_revealed_preference=weak,
            transitive_closure=None,
            strict_revealed_preference=strict,
        )

    weak_star = floyd_warshall_transitive_closure(weak)

    if normalized == "garp":
        violation_matrix = weak_star & strict.T
        violations = _find_garp_violations(weak, violation_matrix)
        return BudgetAxiomCheck(
            axiom=normalized,
            is_consistent=len(violations) == 0,
            violations=violations,
            direct_revealed_preference=weak,
            transitive_closure=weak_star,
            strict_revealed_preference=strict,
        )

    violation_matrix = weak_star & weak_star.T
    np.fill_diagonal(violation_matrix, False)

    # SARP rules out mutual revealed preference between distinct chosen bundles.
    # ConsumerSession.__post_init__ always sets quantities to a non-None array,
    # so the Optional in the field type cannot occur here.
    quantities = cast(NDArray[np.float64], session.quantities)
    same_bundle = np.isclose(
        quantities[:, np.newaxis, :],
        quantities[np.newaxis, :, :],
        rtol=tolerance,
        atol=tolerance,
    ).all(axis=2)
    violation_matrix &= ~same_bundle

    violations = _find_sarp_violations(weak, violation_matrix)
    return BudgetAxiomCheck(
        axiom=normalized,
        is_consistent=len(violations) == 0,
        violations=violations,
        direct_revealed_preference=weak,
        transitive_closure=weak_star,
        strict_revealed_preference=strict,
    )


def _find_garp_violations(
    weak: NDArray[np.bool_],
    violation_matrix: NDArray[np.bool_],
) -> list[Cycle]:
    """Find representative GARP violation cycles."""
    violations: list[Cycle] = []
    seen_cycles: set[frozenset[int]] = set()
    weak_c = np.ascontiguousarray(weak, dtype=np.bool_)

    for i, j in np.argwhere(violation_matrix):
        path = _path_with_cycle_close(weak_c, int(i), int(j))
        if path is None:
            continue
        cycle_set = frozenset(path[:-1])
        if cycle_set not in seen_cycles:
            seen_cycles.add(cycle_set)
            violations.append(tuple(path))

    return violations


def _find_sarp_violations(
    weak: NDArray[np.bool_],
    violation_matrix: NDArray[np.bool_],
) -> list[Cycle]:
    """Find representative SARP mutual-preference cycles."""
    violations: list[Cycle] = []
    seen_cycles: set[frozenset[int]] = set()
    weak_c = np.ascontiguousarray(weak, dtype=np.bool_)

    for i, j in np.argwhere(violation_matrix):
        if i >= j:
            continue
        first = _path_without_cycle_close(weak_c, int(i), int(j))
        second = _path_without_cycle_close(weak_c, int(j), int(i))
        if first is None or second is None:
            continue
        cycle = tuple(first[:-1] + second)
        cycle_set = frozenset(cycle[:-1])
        if cycle_set not in seen_cycles:
            seen_cycles.add(cycle_set)
            violations.append(cycle)

    return violations


def _path_with_cycle_close(
    adjacency: NDArray[np.bool_],
    start: int,
    end: int,
) -> list[int] | None:
    path = bfs_find_path_numba(adjacency, np.int64(start), np.int64(end))
    if len(path) == 0 or path[0] == -1:
        return None
    return [int(x) for x in path]


def _path_without_cycle_close(
    adjacency: NDArray[np.bool_],
    start: int,
    end: int,
) -> list[int] | None:
    path = _path_with_cycle_close(adjacency, start, end)
    if path is None:
        return None
    return path[:-1]
