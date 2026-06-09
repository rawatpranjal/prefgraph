# PrefGraph Roadmap — v0.6 "Trust Release"

> Caveman header. Created 2026-06-09. This doc is the anchor for the current hardening
> cycle. It freezes new algorithms and spends one release on CI, tests, build, and
> identification docs. Now/Next/Later horizons below. Goalposts are locked unless the
> user relocks them. Draft lived at `~/.claude/plans/need-you-to-thrash-lexical-knuth.md`.

## Why this cycle exists

An external reviewer surveyed PrefGraph (PyPI 0.5.17) and argued the highest-leverage work
is trust, API discipline, and adoption friction, not more algorithms. Three parallel repo
audits confirmed the direction but corrected the biggest item.

The reviewer's flagship ask was "build a golden test suite." It is roughly 80 percent already
built. The repo has `tests/test_engine_corner_cases.py` with hand-verified values from Afriat,
Varian, Houtman-Maks, and Echenique, plus `tests/test_backend_parity.py`,
`tests/test_reviewer_cases.py` pinning the exact CHANGELOG bugs, 43 Python test files with 715
test functions, and 78 Rust tests. The real problem is that none of it runs on pull requests.
Only `docs.yml` triggers on `pull_request`, and it only builds Sphinx. So the trust unlock is
enforcement, not authorship.

Audit verdicts. **True:** no PR test CI; `test-install.yml` pins a stale 0.5.3; dual build
config where setuptools is declared at `pyproject.toml:3` but maturin actually ships the
wheels; no Linux aarch64 wheels; 8 algorithm-correctness bugs fixed in a 2-week span; NetworkX
is viz-only and Polars is data-loading-only so both can be extras; no
CONTRIBUTING/SECURITY/cargo-audit. **False:** version incoherence. All three sources
(`pyproject.toml:7`, `src/prefgraph/__init__.py:577`, `docs/conf.py:38`) read 0.5.17, so the
RTD mismatch was stale cached builds, at most a one-time rebuild.

## Locked decisions (2026-06-09)

1. Hardening-only cycle. Freeze all new algorithms and methods for this release.
2. API left untouched. No tiering, no reorg, no deprecation-policy work. The 313-symbol
   surface stays exactly as-is. The reviewer's API-pruning P0 is deferred.
3. Economists first in the second tier. The Identification and Failure Modes page leads.

Target outcome: ship 0.6.0 as a trust release. A research codebase that now behaves like a
reliable library, with every commit gated by the tests that already exist, one unambiguous
build path, and centralized identification guidance for applied users.

---

## NOW — CI enforcement and build unification

The trust unlock. Cheap because the test assets already exist. Do these in order 1 to 4.

### 1. `ci.yml` — gate every PR with the existing suite (highest leverage)
New `.github/workflows/ci.yml`, triggers on `pull_request` and `push: main`. Matrix Python
3.10 to 3.13, ubuntu always, macOS and Windows at least on 3.12 to stay cost-aware. Steps
install with `pip install -e ".[dev]"` (compiles Rust via maturin), then `pytest` with
coverage, `ruff check src/`, `ruff format --check src/`, `mypy src/`, and `cargo test` in
`rust/crates/rpt-core/`. Add a no-Rust-backend fallback job that installs without the Rust
toolchain, asserts `HAS_RUST is False`, and runs the Python-fallback subset (GARP, CCEI, MPI,
HM, HARP, utility), a path claimed in CLAUDE.md but never CI-tested. Add a build-from-source
smoke job that runs `pip install .` from a clean checkout, imports, and runs `load_demo` plus
`Engine.analyze_arrays`.

### 2. Fix `test-install.yml` stale pin
Line 17 installs `prefgraph[datasets]==0.5.3`. Install the current version instead, either
parametrized from the `pyproject.toml` version or from the built wheel or PR artifact rather
than a hardcoded old PyPI release. Keep the existing smoke body at lines 19 to 36.

### 3. Close the two real test gaps (small, not a rewrite)
Property-based tests with `hypothesis` added to `[dev]`, encoding invariants the suite asserts
only on fixed cases. Utility-maximizing data always passes GARP, CCEI lies in 0 to 1, the HM
fraction lies in 0 to 1, and Rust matches Python within the documented tolerance. New file
`tests/test_properties.py`. Then a bug-coverage audit confirming each of the 8 CHANGELOG
correctness bugs has a named regression test (MPI sign and cycle, VEI objective sign, HM SCC
R versus R_star, production GARP R_star[j,i] versus P[j,i], MCI sign, Engine VEI fallback,
`compute_mpi` argument count, base-install pandas crash). Most live in
`tests/test_reviewer_cases.py`; add tests for any uncovered.

### 4. Unify the build on maturin as the single source of truth
CI `release.yml` already builds all wheels via maturin-action, and PyPI 0.5.17 was uploaded by
maturin, so setuptools-rust at `pyproject.toml:1-3` is declared but not what ships. Switch
`[build-system]` `build-backend` to maturin and `requires` to `["maturin>=1.7"]`, drop the
setuptools-rust declaration, and keep `[tool.maturin]` at lines 103 to 108. Verify both
`pip install .` and `pip install -e .` still work and the `_rust_backend.HAS_RUST` fallback
imports clean. The build-from-source CI job guards this going forward. Update the CLAUDE.md
build commands to state maturin as the one backend.

Critical files: `.github/workflows/ci.yml` (new), `.github/workflows/test-install.yml`,
`pyproject.toml`, `tests/test_properties.py` (new), `tests/test_reviewer_cases.py`, `CLAUDE.md`.

---

## NEXT — identification docs and install slimming

### 5. Identification and Failure Modes docs page (economists first)
New page under `docs/` (for example `docs/identification.rst`), added to the RTD nav.
Centralize guidance now scattered across `docs/budget/theory_foundations.rst` (the A1 to A5
assumptions), `case_studies/benchmarks/datasets_issues.md` (menu-dataset caveats), and the
`docs/quickstart.rst` warning. Cover the failure modes the repo under-documents: stockouts and
unavailable items, recommendation and platform bias, reconstructed versus observed menus,
multi-purchase session aggregation, category aggregation, repeated exposure, and time-varying
preferences. State plainly that budget datasets get less scrutiny than menu datasets today and
that scores measure artifacts not behavior when the feasible set is wrong. Follow the project
writing rules. References stay in `docs/papers.rst` only.

### 6. Slim the base install
Move Polars (data-loaders only) and NetworkX (`graph/violation_graph.py` viz only) out of core
`[project]` dependencies into the `datasets` and `viz` extras, guarding their imports so
`import prefgraph` and core scoring work without them. Keep Numba core for now given its 15-plus
`_kernels.py` functions; revisit a pure-NumPy fallback later. Keep NumPy and SciPy core. Update
`docs/install.rst` to show the lean core plus opt-in extras.

Critical files: `docs/identification.rst` (new), `docs/index.rst`, `pyproject.toml`, import
guards in `src/prefgraph/datasets/*` and `src/prefgraph/graph/violation_graph.py`,
`docs/install.rst`.

---

## LATER — reach and contributor trust (deferred, low urgency)

Linux aarch64 wheels by adding aarch64 to the linux target matrix in `release.yml:29-46`, only
if the Rust deps cross-compile cleanly. An sklearn-compatible `PrefGraphTransformer`, since
`encoder.py` already has fit, transform, and fit_transform; add BaseEstimator and
TransformerMixin plus get_params and set_params so it drops into Pipeline and ColumnTransformer,
positioned as diagnostics plus interpretable features consistent with the honest near-zero-lift
case-study finding. Contributor trust files: CONTRIBUTING.md, SECURITY.md,
`.github/ISSUE_TEMPLATE/`, cargo audit or CodeQL in CI, and a public roadmap link. A one-time
RTD rebuild so all pages show 0.6.0, since the source is already coherent.

## Out of scope this cycle

Any new algorithm or method. Any API tiering, reorg, experimental namespace, or
deprecation-policy work. The 313-symbol surface is frozen as-is.

## CCEI supremum fix and the metric-correctness audit

A re-audit of the parallel work found that the CCEI/AEI supremum was computed incorrectly. The
discrete efficiency search returned the next-lower breakpoint instead of the true supremum on
roughly seven percent of datasets, with error up to 0.4, in both backends. The lineage is worth
remembering. The original discrete implementation (March 2026) returned the next-lower breakpoint
and a docstring over-claimed it was "exact." A golden test then hardcoded the buggy output (the
comment literally read "land on the next-lower discrete ratio") and was never checked against the
paper, so it looked verified. Two commits this session added and propagated a one-float-ULP probe
that only worked when own-expenditure was about one, and the second declared it the "paper-backed
supremum convention" while only patching the toy case. The backend parity test could not catch any
of it because it compared Rust against Rust.

The fix (this cycle) computes the index exactly as the upper breakpoint of the highest open
interval on which the axiom holds, tested at interval midpoints, per Smeulders et al. (2014)
Algorithm 2. Both backends now agree with an independent supremum oracle to about 1e-11. The
parity test was made real (it forces the pure-Python path), and `tests/test_ccei_supremum.py` adds
a brute-force oracle property test that would have caught the original bug.

## Parked findings (confirmed, deferred for sequenced follow-up)

These were surfaced by the metric-correctness audit and are recorded so nothing is lost. None are
fixed yet.

The Rust and pure-Python MPI values diverge on some users, because Rust uses Karp's max-mean cycle
and the Python fallback uses cycle enumeration. The parity test now marks this `xfail`. The
pure-Python budget Houtman-Maks can return zero removals on GARP-inconsistent data, which is
impossible and a residual of the earlier HM fix. The menu Houtman-Maks counts items not in cycles
in Rust but observations kept in Python, so the two backends report different quantities under the
same name. Several pure-Python fallback fields return placeholders (`max_scc` is 0, `vei_exact` is
1.0, `is_warp_la` is False, various SCC fields default) instead of computing, so a no-Rust install
reports confident wrong values. The greedy Houtman-Maks path uses `find_sccs(R_star)` and
over-removes by roughly a factor of two at large T, where the public API switches from the exact
ILP to greedy. Smaller items include a VEI docstring and objective mismatch and a quasilinear
default cycle-length truncation.

## Verification

CI: open a throwaway PR and confirm `ci.yml` runs pytest, ruff, mypy, and cargo test across the
matrix and that the fallback and build-from-source jobs pass; confirm `test-install.yml`
installs the current version. Tests: `pytest tests/test_properties.py` and the full suite green,
`cargo test` in `rust/crates/rpt-core/` green, each of the 8 bugs reachable by `pytest -k`.
Build: in a clean venv both `pip install .` and `pip install -e .` succeed and import; with the
Rust toolchain off, `HAS_RUST is False` and fallback scoring runs. Docs:
`python3 -m sphinx docs docs/_build/html` builds clean with the new page in nav. Install
slimming: in a fresh venv `pip install .` then import and score without Polars or NetworkX
present.
