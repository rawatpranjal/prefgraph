# Roadmap for PrefGraph (revealed)

*The single point of contact for this project. Three sections: where we are (STATUS), what we are
driving at (CURRENT GOAL), and everything to do (BACKLOG). The other working doc is `memory.md`
(long-term decisions and learned rules). STATUS is refreshed in place between its markers. CURRENT
GOAL is frozen, only `/goal` (the human) sets it and the agent never edits it. Git is the log: the
v0.6 cycle history lives in `git log` and in the retired `legacy/roadmap-v0.6.md`.*

<!-- STATUS:START -->
## STATUS

1. **North star.** PrefGraph (`prefgraph` on PyPI, v0.6.2) is the only Python revealed-preference
   package: rationality tests, efficiency indices, and utility recovery across budget, discrete,
   production, and intertemporal choice, with a Rust/Rayon batch engine. The architecture and full
   method map live in [CLAUDE.md](CLAUDE.md); decisions and learned rules in [memory.md](memory.md).
2. **Workstreams.** The v0.6 "Trust Release" cycle is complete and shipped to PyPI (0.6.0
   trust and correctness, 0.6.1 Tier 2/3, 0.6.2 exact-VEI close-out). Just done: exact VEI closed per
   Mononen (2023) Theorem 1 with a canonical cross-backend vector (the last substantive correctness
   item, commit 1f44834), then the aarch64 wheel CI reworked to a bindgen cross sysroot with a
   tag-gated publish (commit 50c84be). Next ahead: the aarch64 wheels are unverified until the next
   `v*` tag, and the v0.7 cycle is unscoped.
3. **Needs you now.** Nothing blocking; the repo is between cycles. Two pre-existing benchmark
   artifacts (`case_studies/benchmarks/datasets_issues.md` and `case_studies/benchmarks/output/results.json`)
   are left uncommitted by design. Run `/goal` to open the next cycle.
<!-- STATUS:END -->

## CURRENT GOAL

*Frozen. Only `/goal` (you) sets this; the agent reads it, works it, and never edits it.*

- **Not set.** The v0.6 cycle closed with 0.6.2 and the prior anchor (the exact-VEI tie-break) is
  resolved, so there is no active goal. Run `/goal` to open the next cycle.
- **Candidate next anchors (your choice).** Verify the aarch64 wheels on the next release tag; or
  scope a v0.7 cycle, for example lifting the v0.6 algorithm freeze for new methods, or the
  API-pruning work the v0.6 review marked P0 and deferred (see `legacy/roadmap-v0.6.md`).

## BACKLOG

- [ ] **`aarch64-verify`** · context: the Linux aarch64 wheel build failed on its first 0.6.1 run
  (`dnf: command not found`) and was fixed forward; commit 50c84be reworked it to a bindgen cross
  sysroot with a tag-gated dispatch publish. · done: `release.yml` detects the package manager and
  cross-compiles. · ahead: confirm the aarch64 job is green on the next `v*` tag; if it fails again,
  read the job log and adjust, or drop aarch64 from the linux matrix. · notes: a continue-on-error
  guard means a residual aarch64 failure cannot block the x86_64 publish.
- [ ] **`v0.7-cycle`** · context: the v0.6 algorithm and API freezes were cycle-scoped, so with
  0.6.2 shipped they can lift. · sources: the deferred items in the v0.6 review (the API-pruning P0
  and deprecation-policy work) are preserved in `legacy/roadmap-v0.6.md`. · ahead: scope the cycle
  with `/goal`; the prior anchor (exact-VEI) is closed. · notes: `src/prefgraph/contrib/` (the MLE
  estimation package) is deprecated behind DeprecationWarning shims and is a removal candidate.

## Done log

- v0.6.0 trust and correctness release: PR-CI gating, maturin as the single build, six oracle-verified
  metric fixes (CCEI supremum, MPI objective, Houtman-Maks ILP, fallback parity, exhaustive
  quasilinearity, honest VEI labeling)  @cf59b07  2026-06-09
- v0.6.1 Tier 2/3: identification and failure-modes docs page, slim core install, Linux aarch64
  wheels, sklearn-compatible encoders, contributor and security files  @74989c6  2026-06-10
- v0.6.2: exact VEI per Mononen (2023) Theorem 1 with a canonical cross-backend vector  @1f44834  2026-06-10
- aarch64 wheel CI rework: bindgen cross sysroot, tag-gated publish  @50c84be  2026-06-10
- learned rule on version bumps recorded  @fd2dfed  2026-06-10
