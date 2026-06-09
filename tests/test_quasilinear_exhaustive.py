"""Regression: check_quasilinearity's verdict must be exhaustive.

The default previously enumerated cycles only up to length 3, so a violation
that first appeared in a longer cycle was silently reported as quasilinear (a
false negative). The is_quasilinear verdict is now authoritative via the
all-lengths Bellman-Ford negative-cycle check.
"""

from __future__ import annotations

import numpy as np

from prefgraph import ConsumerSession
from prefgraph.algorithms.quasilinear import (
    check_quasilinearity,
    check_quasilinearity_exhaustive,
)


def test_default_verdict_matches_exhaustive() -> None:
    rng = np.random.default_rng(20260609)
    n_violations = 0
    for _ in range(2000):
        t = int(rng.integers(4, 7))
        g = int(rng.integers(2, 4))
        p = rng.uniform(0.3, 3.0, size=(t, g))
        q = rng.uniform(0.1, 4.0, size=(t, g))
        session = ConsumerSession(prices=p, quantities=q)
        default = check_quasilinearity(session).is_quasilinear
        exhaustive = check_quasilinearity_exhaustive(session).is_quasilinear
        assert default == exhaustive
        if not exhaustive:
            n_violations += 1
    # Guard against the test passing vacuously on all-consistent data.
    assert n_violations > 0
