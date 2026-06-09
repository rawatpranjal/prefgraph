"""Regression and oracle tests for the menu Houtman-Maks index.

The Houtman-Maks index is the largest fraction of OBSERVATIONS consistent with
the axiom (Houtman & Maks 1985; Demuynck & Rehbeck 2023 Definition 3), not a
count over items. The previous menu implementation worked in item space with an
ad-hoc heuristic (Rust) and a last-writer-wins edge-to-observation map (Python),
which over- or under-counted and could return an invalid subset that still
violated SARP. These tests pin the exact value against an exhaustive oracle.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from prefgraph import MenuChoiceLog, compute_menu_efficiency


def _is_sarp_consistent(obs_edges: list[list[tuple[int, int]]], subset, n_items: int) -> bool:
    r = np.zeros((n_items, n_items), dtype=bool)
    for o in subset:
        for c, k in obs_edges[o]:
            r[c, k] = True
    closure = r.copy()
    for _ in range(n_items):
        closure = closure | (closure @ closure)
    viol = closure & closure.T
    np.fill_diagonal(viol, False)
    return not viol.any()


def _brute_force_kept(menus: list[list[int]], choices: list[int], n_items: int) -> int:
    """Exhaustive maximum number of observations forming a SARP-consistent subset."""
    n = len(menus)
    obs_edges = [[(choices[o], it) for it in menus[o] if it != choices[o]] for o in range(n)]
    for keep in range(n, -1, -1):
        for subset in itertools.combinations(range(n), keep):
            if _is_sarp_consistent(obs_edges, subset, n_items):
                return keep
    return 0


def _impl_kept(menus: list[list[int]], choices: list[int]) -> int:
    log = MenuChoiceLog(menus=[frozenset(m) for m in menus], choices=list(choices))
    return len(compute_menu_efficiency(log).remaining_observations)


def test_menu_hm_consistent_keeps_all() -> None:
    menus = [[0, 1], [0, 1], [1, 2]]
    choices = [0, 0, 1]  # 0>1, 0>1, 1>2: acyclic
    log = MenuChoiceLog(menus=[frozenset(m) for m in menus], choices=choices)
    r = compute_menu_efficiency(log)
    assert r.efficiency_index == pytest.approx(1.0)
    assert r.removed_observations == []


def test_menu_hm_four_cycle_removes_one() -> None:
    # 0>1>2>3>0: removing any one observation breaks the cycle, so HM keeps 3/4.
    menus = [[0, 1], [1, 2], [2, 3], [3, 0]]
    choices = [0, 1, 2, 3]
    log = MenuChoiceLog(menus=[frozenset(m) for m in menus], choices=choices)
    r = compute_menu_efficiency(log)
    assert len(r.remaining_observations) == 3
    assert r.efficiency_index == pytest.approx(3.0 / 4.0)
    # The kept subset must itself be SARP-consistent (the old code could return
    # an invalid subset that still violated SARP).
    obs_edges = [[(choices[o], it) for it in menus[o] if it != choices[o]] for o in range(4)]
    assert _is_sarp_consistent(obs_edges, r.remaining_observations, 4)


def test_menu_hm_matches_exhaustive_oracle() -> None:
    """Guardrail: the menu HM must equal the exhaustive maximum-consistent-subset
    on random menu logs. This catches the item-vs-observation and invalid-subset
    bugs the old implementation had."""
    rng = np.random.default_rng(20260609)
    for _ in range(200):
        n_items = int(rng.integers(3, 6))
        n_obs = int(rng.integers(2, 8))
        menus, choices = [], []
        for _o in range(n_obs):
            size = int(rng.integers(2, n_items + 1))
            menu = sorted(rng.choice(n_items, size=size, replace=False).tolist())
            menus.append(menu)
            choices.append(int(rng.choice(menu)))
        assert _impl_kept(menus, choices) == _brute_force_kept(menus, choices, n_items)
