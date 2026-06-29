# Roadmap for PrefGraph (revealed)

*The single point of contact for this project. Three sections: where we are (STATUS), what we are
driving at (CURRENT GOAL), and everything to do (BACKLOG). The other working doc is `memory.md`
(long-term decisions and learned rules). STATUS is refreshed in place between its markers. CURRENT
GOAL is frozen, only `/goal` (the human) sets it and the agent never edits it. Git is the log: the
v0.6 cycle history lives in `git log` and in the retired `legacy/roadmap-v0.6.md`.*

<!-- STATUS:START -->
## STATUS

1. **North star.** PrefGraph (`prefgraph` on PyPI, v0.6.2 verified 2026-06-29) is the
   revealed-preference package in this checkout, with rationality tests, efficiency indices, utility
   recovery, and a Rust/Rayon batch engine across budget, discrete, production, and intertemporal
   choice. The architecture and full method map live in [CLAUDE.md](CLAUDE.md); decisions and learned
   rules in [memory.md](memory.md).
2. **Workstreams.** The v0.6 "Trust Release" is complete in Git and public on PyPI through `v0.6.2`:
   PR-CI gating, maturin as the single build, six oracle-verified metric-correctness fixes, the
   identification and failure-modes docs page, slim core install, sklearn-compatible encoders, and
   exact VEI per Mononen (2023) Theorem 1 with a canonical cross-backend vector. The release logs for
   `v0.6.0`, `v0.6.1`, and `v0.6.2` show successful trusted-publisher uploads. After the `v0.6.2`
   tag, the release workflow was reworked to use a bindgen cross sysroot for Linux aarch64 wheels
   (commit 50c84be), then the docs were migrated to the current four-doc structure (commit d1aaea7).
   The Linux aarch64 workflow fix has not yet been proven on a tag, and v0.7 remains unscoped.
3. **Needs you now.** The repo is between cycles. Two benchmark artifacts remain dirty by design
   (`case_studies/benchmarks/datasets_issues.md` and `case_studies/benchmarks/output/results.json`).
   The applied paper batch under `references/papers/applied_revealed_preference/` is local input for
   v0.7 planning and must be committed with that cycle if the roadmap keeps depending on it. Run
   `/goal` to open the next cycle.
<!-- STATUS:END -->

## CURRENT GOAL

*Frozen. Only `/goal` (you) sets this; the agent reads it, works it, and never edits it.*

- **Not set.** The v0.6 cycle closed with 0.6.2 and the prior anchor (the exact-VEI tie-break) is
  resolved, so there is no active goal. Run `/goal` to open the next cycle.
- **Candidate next anchors (your choice).** Verify the aarch64 wheels on the next release tag; or
  scope a v0.7 cycle, for example lifting the v0.6 algorithm freeze for new methods, or the
  API-pruning work the v0.6 review marked P0 and deferred (see `legacy/roadmap-v0.6.md`).

## BACKLOG

- [ ] **`aarch64-verify`** · context: Git and PyPI are current through `v0.6.2`, and the release
  logs for `v0.6.0`, `v0.6.1`, and `v0.6.2` show successful trusted-publisher uploads. The remaining
  release gap is Linux aarch64. The PyPI simple index for `v0.6.2` shows manylinux x86_64 wheels but
  no manylinux aarch64 wheels, and commit 50c84be reworked the Linux aarch64 job after the `v0.6.2`
  tag by adding a bindgen cross sysroot. · ahead: confirm the aarch64 job on the next real `v*` tag.
  If it fails again, read the job log and either fix the cross build or remove Linux aarch64 from the
  wheel matrix rather than letting an unverified target linger. · guard: `continue-on-error` means a
  residual aarch64 failure cannot block x86_64 publication, so success must be checked explicitly.
  · done criteria: PyPI still shows the intended release as latest; a clean environment can
  `pip install prefgraph==<released-version>` and import `prefgraph`; the release workflow outcome is
  recorded; and the Linux aarch64 decision is explicit in STATUS.

- [ ] **`simulation-studies-v0.7`** · goal: turn the applied revealed-preference paper batch into a
  replication-oriented simulation-study cycle without breaking existing examples, benchmarks, or
  case studies. The source inventory is
  `references/papers/applied_revealed_preference/CASE_STUDY_CANDIDATES.md`, and the download audit
  is `references/papers/applied_revealed_preference/DOWNLOAD_MANIFEST.md`. The current input folder
  is untracked, so the first implementation PR for this cycle must either commit the required paper
  artifacts or revise this roadmap not to depend on them.

  **Non-negotiables.** Keep `examples/` as runnable API examples unless a compatibility alias is
  added and tested. Keep existing benchmark and case-study pages reachable, including
  `benchmarks.html` and `benchmarks_ecommerce.html`. Treat these as Simulation Studies, not
  Applications, because the repo's application rule requires real outputs and real use cases. Use
  Polars for data work. Use `Engine.analyze_arrays()` or `Engine.analyze_menus()` for multi-user
  scoring unless the study is a documented single-user walkthrough or a metric with no batch path.
  Do not add per-page References sections unless the repo-wide docs rule is changed first; source
  paper citations should route through `docs/papers.rst`.

  **Phase 0, ingestion gate.** For each study, read the source PDFs, then create Docling markdown
  under `references/papers/applied_revealed_preference/docling_md/<paper-slug>.md`. Each study folder
  must include `replication_targets.md` with the paper tables or figures to mimic, the reported
  headline numbers, the synthetic or loader data needed, and the expected tolerance or qualitative
  match. If the original data are proprietary, say that directly and calibrate the synthetic data to
  reproduce direction, scale, and ranking of results. Do not imply empirical replication when the
  study is only paper-style simulation.

  **Phase 1, shared scaffold.** Add `case_studies/simulation_studies/` with a common runner shape,
  deterministic seeds, small fixture outputs, and one callable per study that returns ready-to-score
  PrefGraph objects or Engine tuples. Promote reusable generators to `src/prefgraph/datasets/` only
  after the case-study-local version is stable. Every study must write machine-readable result
  tables, rendered figures, and a generated summary under
  `case_studies/simulation_studies/<study>/output/`. Each study must have a smoke test that runs
  without network access.

  **Phase 2, budget ground truth.** Build `choi_budget_under_risk` first, because it gives known
  rational, random, and noisy subjects for the rest of the cycle. Source PDFs:
  `public_only/direct_found/AER07-Risk.pdf` and
  `public_only/[2011] - Who Is (More) Rational.pdf`. Required outputs: CCEI distribution or quantile
  table by subject type, Houtman-Maks table, Bronars random-choice benchmark, and one figure showing
  noise moving scores away from 1. Then build `rationality_measure_comparison` on the same fixture
  data. Source PDFs include
  `public_only/from_existing_collection/Mononen2023_ComputingMeasures.pdf`,
  `public_only/[2014] - Consistent Subsets – Computationally Feasible Methods to Compute the
  Houtman-Maks-Index.pdf`, `public_only/[2022] - Testing axioms of revealed preference in Stata.pdf`,
  and optional Bronars, Echenique-Lee-Shum, Dean-Martin, and Smeulders files from
  `public_only/from_existing_collection/`. Required outputs: comparison table for AEI/CCEI, VEI,
  Houtman-Maks, MPI bounds, and Bronars power under sparse errors, diffuse noise, and random choice.
  Dean-Martin MCI is optional after the first version passes.

  **Phase 3, LLM budget study.** Build `chen_llm_budget_rationality`. Source PDF:
  `public_only/[2023] - The Emergence of Economic Rationality of GPT.pdf`; optional extension:
  `public_only/arXiv 2412.04476.pdf`. Required outputs: 25 two-good budget-allocation tasks, a
  deterministic fixture of parsed LLM-style allocations, GARP/CCEI plus optional Houtman-Maks, MPI
  bounds, and Bronars power, a paper-style CCEI table by domain and model or run type, and a
  distribution figure. The target pattern is strong-model fixture scores concentrated near CCEI 1
  and random or noisy controls below it.

  **Phase 4, menu attention study.** Build `random_attention_ram`. Source PDFs:
  `public_only/[2019] - A Random Attention Model.pdf`,
  `public_only/[2012] - Revealed Attention.pdf`, and optionally
  `public_only/[2017] - What Do Consumers Consider Before They Choose- Identification from
  Asymmetric Demand Responses.pdf`. Required outputs: a RAM or logit-attention simulator, known
  data-generating-process table versus recovered restrictions, and a figure comparing RAM-generated
  data against full-attention misspecification. The target pattern is RAM-generated data passing the
  attention checks more often than the full-attention baseline.

  **Phase 5, dominated-menu quality study.** Build `health_plan_dominated_menus`. Source PDF:
  `public_only/direct_found/ChoseLose.pdf`; optional context:
  `public_only/[2009] - Choice Inconsistencies Among the Elderly- Evidence from Plan Choice in the
  Medicare Part D Program.pdf` and
  `public_only/[2013] - Health Insurance for -Humans- Information Frictions, Plan Choice, and
  Consumer Welfare.pdf`. Required outputs: synthetic plan menus with premiums, deductibles, and
  out-of-pocket costs, a deterministic dominated-option checker, chooser types, dominated-choice
  rate table, and avoidable-cost figure. Present this as a choice-quality study, not a welfare proof.

  **Phase 6, docs integration.** Only after at least one study has real outputs, rename the public
  "Examples" navigation concept to "Simulation Studies" or update the repo guidance if "Case
  Studies" is the chosen label instead. Preserve old URLs where possible by retitling files before
  renaming files. Add a Simulation Studies landing page with the common contract, then one page per
  study with this spine: introduction, source-paper context, business problem, data generator or
  loader, data processing, key concepts, analysis tables and figures, interpretation, discussion,
  and conclusion. The first screen should say why the study matters and what result is reproduced.

  **Cycle done criteria.** Required source PDFs are present and have Docling markdown. Every study
  has a read note and `replication_targets.md`. Every study creator runs without network by default.
  Every study has deterministic smoke coverage. Every table and figure rebuilds from code. Docs
  build locally, and any pushed public-docs change is checked on Read the Docs before being called
  done. `rg "Examples" docs README.md roadmap.md` shows only intentional legacy or compatibility
  references. The roadmap, case-study note, docs navigation, and generated outputs agree on study
  names and scope.
- [ ] **`v0.7-cycle`** · context: the v0.6 algorithm and API freezes were cycle-scoped, so with
  `v0.6.2` public they can lift when `/goal` scopes the next cycle. · sources: the deferred items in
  the v0.6 review (the API-pruning P0 and deprecation-policy work) are preserved in
  `legacy/roadmap-v0.6.md`. · ahead: scope the cycle with `/goal`; the prior anchor (exact-VEI) is
  closed. · notes: `src/prefgraph/contrib/` (the MLE estimation package) is deprecated behind
  DeprecationWarning shims and is a removal candidate.

## Done log

- v0.6.0 trust and correctness release: PR-CI gating, maturin as the single build, six oracle-verified
  metric fixes (CCEI supremum, MPI objective, Houtman-Maks ILP, fallback parity, exhaustive
  quasilinearity, honest VEI labeling)  @cf59b07  2026-06-09
- v0.6.1 Tier 2/3 release: identification and failure-modes docs page, slim core install, Linux
  aarch64 wheel target, sklearn-compatible encoders, contributor and security files  @74989c6  2026-06-10
- v0.6.2: exact VEI per Mononen (2023) Theorem 1 with a canonical cross-backend vector  @1f44834  2026-06-10
- aarch64 wheel CI rework: bindgen cross sysroot, tag-gated publish  @50c84be  2026-06-10
- learned rule on version bumps recorded  @fd2dfed  2026-06-10
