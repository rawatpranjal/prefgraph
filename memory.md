# Memory for PrefGraph (revealed)

*Long-term project memory. Decisions taken and learnings learnt, with the why, so they are never
re-derived or re-asked. A line earns a place here only if it is super-critical, something a future
session would get wrong without it. No trivia. Rarely changes. Seeded 2026-06-28 from the v0.6
cycle docs (CLAUDE.md learned rules, the retired `legacy/roadmap-v0.6.md`, and git).*

## Decisions

| Date | Decision | Reason |
|------|----------|--------|
| 2026-06-09 | v0.6 is a hardening-only "Trust Release": freeze all new algorithms, lock the API surface, spend the cycle on CI, tests, build unification, and identification docs. | An external review of PyPI 0.5.17 argued the highest-leverage work was trust and adoption friction, not more algorithms. Three parallel repo audits confirmed the direction. |
| 2026-06-09 | Unify the build on maturin as the single backend; remove the declared setuptools-rust path. | PyPI wheels were already shipped by maturin, so the setuptools-rust declaration was dead config that misled the build story. |
| 2026-06-09 | Slim the base install: move Polars to the `datasets` extra and NetworkX to the `viz` extra, behind import guards. | Polars is data-loading only and NetworkX is viz only, so neither belongs in core; `import prefgraph` and core scoring must work without them. |
| 2026-06-10 | Close exact VEI per Mononen (2023) Theorem 1 in 0.6.2, with a canonical reporting vector (minimize the largest single adjustment among value-optimal solutions, then resolve ties in observation order). | The feedback-arc-set cost objective is not Varian's L1 norm and the per-observation optimum is non-unique under ties, so Rust (HiGHS) and Python (scipy.milp) disagreed on discrete data. Theorem 1 with U-set covering rows fixes the objective; the canonical vector makes the reported vector bit-identical across backends on integer data. |

## Learnings

| Date | Learning | Where it bit |
|------|----------|--------------|
| 2026-06 | Golden test values come from the paper or an independent oracle, never the code's own output. When you must change a golden, prove the new value independently and say why. | A CCEI golden was hardcoded to 1/3 to match a buggy implementation (comment: "land on the next-lower discrete ratio"), so the bug looked verified for months. |
| 2026-06 | Never label a metric "exact" or a fix "paper-backed" without an independent cross-check, and never let the author bless their own fix. A separate adversarial check (brute-force oracle, second method, or fresh agent) must clear it first. | A one-ULP "supremum fix" shipped as the "paper-backed convention" while patching only one case. |
| 2026-06 | Parity tests must force the fallback path (`HAS_RUST=False`, or `PREFGRAPH_NO_RUST=1` in a subprocess). | `compute_aei`/`check_garp`/`compute_mpi` read `HAS_RUST` at call time, so a "Python vs Rust" test silently compared Rust vs Rust and hid the CCEI divergence. |
| 2026-06-10 | Version bumps are the smallest honest increment: patch for fixes (even ones that change reported numbers), minor only for new API surface. Never inflate a version to signal effort. | Shipping a correctness fix as 0.7.0 to look bigger is the smell; 0.6.2 with a plain changelog line is right (commit fd2dfed). |
| 2026-06-10 | Editable install is not live for Python source: maturin `pip install -e .` copies to site-packages, so `.py` edits need a rebuild (`pip install -e . --no-build-isolation`) before pytest sees them. `docs/conf.py` reads `../src`, so docs autodoc does not. | Bit repeatedly during the v0.6 cycle. |
| 2026-06-10 | `gh pr checks --watch && gh pr merge` returns a misleading non-zero exit even when the merge succeeds; verify the outcome with `gh pr view <n> --json state` (MERGED), never the watcher exit code. | Bit repeatedly during the six-agent 0.6.1 workflow. |
