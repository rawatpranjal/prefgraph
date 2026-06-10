# Adversarial audit: exact Varian Efficiency Index (Mononen 2023)

Date 2026-06-10. Branch `fix/062-vei-exact-varian`. Auditor did not write the code.
Target: `compute_vei_exact` in `rust/crates/rpt-core/src/vei.rs` and its mirror in
`src/prefgraph/algorithms/vei.py`, plus the engine fallback and batch stats.

## Verdict: PASS-WITH-FINDINGS

The core mathematics is correct. The Theorem 1 value AND the canonical
(max-min-then-lex) per-observation vector match an independent
exact-`Fraction` oracle on every one of roughly 950 integer datasets, 120
continuous datasets, and 20 hand-constructed degenerate cases, with bit-exact
Rust/Python parity on integer data. No dataset was found where either backend
disagrees with the oracle on value or canonical vector (the protocol's BLOCKER
condition never triggered). All findings below are about error surfacing and
test hygiene, not about a wrong number on a valid input.

Findings by severity: 0 BLOCKER, 2 MED, 1 LOW, 2 NOTE.

## How the math was re-derived and checked

Read Mononen (2023) pp. 9-14. Pinned the index definition (p. 9), the
infimum-grid characterisation (each `e_t` is 0 or an arc cost `1 - p_t.x_j / p_t.x_t`,
p. 10), the U-set `U(x_t,x_{t*}) = {(x_t,x_{t'}) | p_t.(x_t - x_{t*}) <= p_t.(x_t - x_{t'})}`
= arcs out of `t` costing at least `cost(t,t*)` (p. 10), Theorem 1 (p. 11),
Algorithm 1 separation / AddCost (p. 13), Algorithm 2 row generation (p. 14).

Line-checked the Rust implementation against the derivation and confirmed each
piece:

- **Arc costs** `cost_ij = 1 - E[i,j]/own[i]` for strict arcs `own[i] > E[i,j]`
  (vei.rs 236-252). Matches p. 10.
- **U-set prefix** built from per-observation arc lists sorted by cost
  descending; the prefix `cost >= ca - tol` is exactly the U-set restricted to
  strict arcs (vei.rs 268-279, 502-525). Matches.
- **SOS rows** (at most one arc per observation, vei.rs 577-583). I verified the
  dominance argument independently: if two arcs `(t,a)` cost more than `(t,b)`,
  every covering row containing `(t,b)` also contains `(t,a)` because
  `U(.,a) subset U(.,b)`, so dropping `(t,b)` keeps feasibility and lowers the
  objective. Hence every value-optimal solution already has at most one arc per
  observation; the SOS row is implied, not a real cut, in all three stages.
- **Survival / AddCost** `cost_ij - d_i > tol` equals `(1-e_i) own_i > E[i,j]`
  (vei.rs 644-654). Matches Algorithm 1.
- **Budget row** in stages B/C restricts to value-optimal solutions; **caps**
  pin the lexicographic prefix (vei.rs 585-616). Correct.
- **Vector extraction** from the binary incumbent's arc ratios, not raw solver
  columns, with numpy argmin convention (vei.rs 432-447). This is why the two
  backends agree exactly on discrete data.

The Python mirror is semantically identical (sort key `(-cost, idx)`, same
tolerances, same staged solves via `scipy.optimize.milp`, `mip_rel_gap=0`).
Confirmed `compute_vei_exact` in `algorithms/vei.py` does NOT delegate to the
Rust extension (no `HAS_RUST`/`_rust_backend` reference; it calls scipy.milp at
vei.py:571).

## Independent oracle

`/tmp/audit_oracle.py`, written from the definition before reading the test
oracle. It builds the per-observation cost grid `{0} U {cost(t,j)}`, enumerates
it by branch-and-bound, finds the minimum total adjustment such that the graph
of surviving strict arcs `(1-e_t) own_t > E[t,j]` has no directed cycle,
enumerates ALL value-optimal grid vectors, and applies the canonical selection
(min of max adjustment, then lexicographic min of the adjustment vector). On
integer data it uses exact `fractions.Fraction`, eliminating float ambiguity.

The oracle reproduces all five repo fixtures independently: 2-obs WARP value
1/8, NESTED_T3 value 2/11 with `e=(1,9/11,1)`, STAGE_B value 4/11 with
`m_b=3/11` and `e=(1,1,10/11,8/11)`, STAGE_C value 1/8 with `e=(1,1,7/8,1)`, and
LARGE_T7 value `23/136 + 6/83 = 0.241406804` with the claimed canonical vector.
So the fixture goldens are not circular and the T7 claimed optimum is genuinely
optimal.

## Sweep results (scripts in /tmp/audit_sweep.py, audit_hard.py, audit_probe.py)

| Sweep | N | matched (Rust vs oracle) | matched (Python vector vs oracle) | matched (Rust vs Python) | mismatch |
|---|---|---|---|---|---|
| Integer T in 3..7, small values | 350 | 350 | 350 | 350 | 0 |
| Continuous T in 3..6 | 120 | 120 | 120 | 120 | 0 |
| Hard integer (denser cycles) | 600 | 600 | 600 | 600 | 0 |

Of the 950 integer datasets, 235+ were GARP-violating (non-trivial index).
Compared value AND all five stats (mean, min, std, q25, q75; numpy
conventions) for both backends, plus the full efficiency vector for Python.
Rust q25/q75 were bit-exact against numpy on every dataset (the branch-corrected
lerp in `batch.rs::percentile` holds; perc_ulp_fail = 0).

Targeted degeneracy probes, all OK:
- Two and three disjoint 2-cycles (4 and 8 value-optimal solutions) -
  canonical matches.
- Random search surfaced datasets with up to 4 optima; canonical matches.
- Equal-cost multiple arcs at one observation - matches.
- 3-good symmetric ring (3 optima) - matches.
- Strict-graph-acyclic but GARP-violating (a 2-cycle with one non-strict arc):
  oracle and both backends return index 0, confirming the reduction to strict
  cycles is implemented correctly (vei.rs 297-305).
- All-consistent: index 0.
- T=1: handled.

## Findings

### 1. MED - Rust batch silently reports `mean=0` on solver failure / solve cap
`rust/crates/rpt-python/src/batch.rs` 311-314 calls `run_vei_exact(graph)` and
reads `mean_efficiency`/`min_efficiency` WITHOUT checking `vei.success`. On
failure `vei_failure` returns `efficiency_vector = [0.0; t]`, `mean = 0.0`,
`min = 0.0` (vei.rs 472-481), so a HiGHS failure or the 1000-solve cap (vei.rs
162, 319) produces `vei_exact_mean = 0.0, vei_exact_min = 0.0` - a perfectly
plausible "maximally irrational user" value, indistinguishable from a real
result. The `success` flag is computed but never threaded into the result dict
(grep: `vei_exact_success` appears 0 times in batch.rs and engine.py). The
Python Engine fallback mirrors this: `engine.py` 813-820 catches the
`SolverError` and sets the stats to 0.0 with the comment "a conspicuous zero" -
but 0.0 is NOT conspicuous for an efficiency mean, it is a valid extreme.

This directly contradicts the code's own stated intent (vei.rs 159-162: "Hitting
it returns success = false, never a silently truncated answer"). The per-user
`compute_vei_exact` DOES raise `SolverError`, so only batch users are affected.

Evidence: `batch.rs:311-314`, `vei.rs:472-481`, `engine.py:813-820`. The sweeps
never hit the cap (performance guard: 100 users x T=20 in 0.56 s), so this is
rare, but on a large dense user it would mislabel a failure as a real zero.

Fix: add a `vei_exact_success` boolean to `BatchResult`/`EngineResult`, set from
`vei.success`, and have batch.rs report NaN (or propagate the failure) instead
of 0.0 when `!vei.success`; make the Python fallback set NaN, not 0.0.

### 2. MED - a neighbouring Python-vs-Rust parity test does not force the fallback
`tests/test_properties.py::test_rust_python_backends_agree` (line 170) compares
`engine._analyze_chunk_python(...)` against `engine._analyze_chunk_rust(...)` for
GARP/CCEI/MPI WITHOUT setting `HAS_RUST=False`. I verified that the Python path
delegates to Rust at call time (`aei.py:81`, `mpi.py:106`, `garp.py:107` all
read `HAS_RUST` and call `_rust_*`), so the "Python" side is actually Rust: this
is the exact Rust-vs-Rust vacuous-parity smell called out in CLAUDE.md Learned
Rules. The VEI-exact parity tests do NOT have this defect -
`test_properties.py:305` and `test_backend_parity.py::_run_both` both correctly
force `rb.HAS_RUST=False`. This finding is collateral (not part of the vei_exact
change) but is reported because the protocol asked to grep for it.

Evidence: `test_properties.py:181` (no `HAS_RUST=False`), vs the correct pattern
at `test_properties.py:321-326`.

Fix: wrap the Python call in `rb.HAS_RUST=False` exactly as `_run_both` does.

### 3. LOW - HiGHS solver chatter leaks to stdout from the Python MILP path
Running the suites floods stdout with
`HighsMipSolverData::transformNewIntegerFeasibleSolution tmpSolver.run();`. The
Rust path silences HiGHS via `model.make_quiet()` (vei.rs 619); the Python
`_vei_solve_stage` (vei.py 571-579) passes no display suppression, so scipy's
HiGHS prints internal progress. Cosmetic but it buries pytest output and would
spam any batch run that falls back to Python. Fix: redirect/​suppress HiGHS
output around the `milp` call.

### 4. NOTE - the 1e-12 cost tolerance is a theoretical, not practical, exposure
The shared `VEI_COST_TOL = 1e-12` governs the arc filter (`cost > tol`), U-set
inclusion (`cost >= ca - tol`), and survival (`cost - d > tol`). On
adversarially constructed continuous data with two arc costs at one observation
within 1e-12 of each other, an arc could be mis-classified as covering/removed,
producing a grid-step error in the value. Empirically this never fires: over
200,000 random continuous datasets the smallest gap between two distinct arc
costs at one observation was 5.2e-8, about four orders of magnitude above the
tolerance. Worst-case impact on realistic data is bounded by about `T * 1e-12`
in the index, and both backends share the tolerance so parity is preserved
regardless. Acceptable for float64; left as a documented caveat.

### 5. NOTE - dead T=0 branch in the Python function
`compute_vei_exact`'s `if T == 0` early return (vei.py 289-290) is unreachable
from the per-user API: `BehaviorLog`/`ConsumerSession` raise
`InsufficientDataError` on empty data before the function runs. The Rust Engine
path handles T=0 (returns mean=1). Harmless.

## Suites

- `python3.11 -m pytest tests/test_vei_exact.py tests/test_backend_parity.py
  tests/test_properties.py -q` -> 70 passed.
- `cargo test -p rpt-core vei` -> 10 passed.
