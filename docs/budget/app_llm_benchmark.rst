LLM Enterprise Consistency Benchmark
=====================================

SARP, Houtman-Maks, and IIA applied to LLM decision-making across
5 enterprise scenarios. 13,750 decisions, two experiment iterations.

Motivation
----------

Accuracy measures whether the LLM is *right*. SARP measures whether
it has a *coherent policy* — a stable ranking over actions that doesn't
depend on which alternatives happen to be shown. Standard in behavioral
economics (Varian 1982), new to LLM evaluation.

Setup
-----

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Scenario
     - 5 Actions
   * - Support Router
     - auto-reply KB, bug ticket, billing, account mgr, escalate VP
   * - Alert Triage
     - auto-resolve, P3 ticket, page on-call, incident channel, runbook
   * - Content Review
     - approve, warning, hide, remove+strike, suspend+legal
   * - Job Screen
     - reject, hold, phone screen, technical, fast-track
   * - Procurement
     - auto-approve, tag, request quotes, escalate, deny

5 prompts per scenario: *minimal, decision-tree, conservative, aggressive, chain-of-thought*.

v1: Pooled SARP (10,000 decisions)
-----------------------------------

Different vignette per trial. SARP across all 200 trials per group.

- 50/50 groups fail SARP. HM=0.60 everywhere. Zero variation.
- Obs-level HM 0.95 — 95% of individual decisions locally consistent.
- gpt-4o-mini ≈ o4-mini on consistency.
- **Design flaw**: varying the vignette each trial means SARP violations
  = correct classification, not inconsistency.

v2: Per-Vignette SARP (3,750 decisions)
----------------------------------------

Fix vignette, vary menu. 10 curated vignettes per scenario, C(5,2)+5 = 15
menus each. gpt-4o-mini, temp=0.

.. list-table:: SARP pass rates (% of 10 vignettes)
   :header-rows: 1
   :widths: 22 13 13 13 13 13

   * -
     - Min
     - DecTree
     - Conserv
     - Aggress
     - CoT
   * - Support
     - 90
     - 80
     - 100
     - 80
     - 90
   * - Alert
     - 80
     - 100
     - 90
     - 100
     - 90
   * - Content
     - 90
     - 80
     - 70
     - 90
     - 80
   * - Jobs
     - 70
     - 60
     - 80
     - 80
     - 80
   * - Procurement
     - 70
     - 100
     - 70
     - 90
     - 90

.. list-table:: SARP pass rates by vignette tier
   :header-rows: 1
   :widths: 22 16 16 16 16

   * -
     - Clear
     - Binary
     - Ambig.
     - Adversar.
   * - Support
     - 87
     - 93
     - 90
     - 80
   * - Alert
     - 93
     - 100
     - 90
     - 80
   * - Content
     - 60
     - 80
     - 100
     - 100
   * - Jobs
     - 87
     - 67
     - 70
     - 70
   * - Procurement
     - 93
     - 73
     - 90
     - 80

.. list-table:: IIA violations (decoy effects)
   :header-rows: 1
   :widths: 30 15

   * - Scenario
     - Count
   * - Support Router
     - 3
   * - Alert Triage
     - 2
   * - Content Review
     - 9
   * - Job Screen
     - 15
   * - Procurement
     - 8

v2 Stage 2: Stochastic (temp=0.7, K=20)
-----------------------------------------

Same vignettes and menus as Stage 1, but at temp=0.7 with 20 repetitions
per (vignette, menu, prompt). Tests whether stochastic variation changes
the consistency picture.

.. list-table:: Stochastic variation by prompt (% of menus with mixed responses)
   :header-rows: 1
   :widths: 22 13 13 13 13 13

   * -
     - Min
     - DecTree
     - Conserv
     - Aggress
     - CoT
   * - Support
     - 13
     - 24
     - 0
     - 8
     - 12
   * - Alert
     - --
     - --
     - --
     - --
     - --
   * - Content
     - --
     - --
     - --
     - --
     - --
   * - Jobs
     - --
     - --
     - --
     - --
     - --
   * - Procurement
     - --
     - --
     - --
     - --
     - --

*-- = data collection in progress. Updated as scenarios complete.*

Early finding (support_ticket, n=6,745): **88% of menus produce identical
choices across 20 reps at temp=0.7.** gpt-4o-mini is overwhelmingly
deterministic. Decision-tree prompts are 2--3x more stochastic (24% mixed)
than conservative (0%). Mean choice entropy: 0.064 (near zero). Stochastic
IIA violation rate: 1.1% (vs 4% deterministic).

Findings
--------

- **Job screening: 15 IIA violations**, most of any scenario. Showing a
  third candidate flips the ranking between two others.
- **Content review "clear" vignettes: 60% consistent** — lowest of any
  tier. Moderation has no unambiguous inputs.
- **Decision-tree prompts: 60% on jobs** — worst combination. Explicit
  rules create more edge cases than they resolve.
- **Conservative prompts: 100% on support, 70% on content/procurement.**
  No universal "most consistent" prompt.
- **Alert triage: 92% overall.** Clearest ordinal structure = highest
  consistency.
- **Reasoning ≠ consistency.** o4-mini and gpt-4o-mini indistinguishable
  (v1, n=10,000).
- **50% of inputs are prompt-invariant** (v1). Prompt engineering only
  matters on borderline cases.
- **v1=0% pass, v2=60--100% pass.** Same model, same prompts. The
  difference is entirely experimental design.

What ``prefgraph`` adds to LLM evaluation:

- ``validate_menu_sarp()``: per-input consistency — does a stable ranking exist?
- ``compute_menu_efficiency()``: which action pairs cycle?
- IIA detection: does a third option flip a pairwise preference?

Reproduce
---------

.. code-block:: bash

   pip install prefgraph openai
   export OPENAI_API_KEY=your_key
   cd examples

   # v1
   python -m applications.llm_benchmark.run_benchmark --all --trials 200
   python -m applications.llm_benchmark.analyze --all

   # v2
   python -m applications.llm_benchmark.v2.run_benchmark --all --stage 1
   python -m applications.llm_benchmark.v2.analyze --all

Appendix
--------

.. code-block:: python

   from prefgraph import MenuChoiceLog
   from prefgraph.algorithms.abstract_choice import validate_menu_sarp

   log = MenuChoiceLog(
       menus=[frozenset(r["menu"]) for r in records],
       choices=[r["choice"] for r in records],
   )
   result = validate_menu_sarp(log)
   # result.is_consistent, result.violations, result.transitive_closure

.. list-table:: Metrics
   :header-rows: 1
   :widths: 22 12 66

   * - Metric
     - Range
     - Meaning
   * - SARP pass rate
     - 0--100%
     - Fraction of inputs with transitive action ranking
   * - HM efficiency
     - 0--1
     - Fraction of items in largest consistent subset
   * - IIA violations
     - 0--n
     - Third option flips pairwise preference
   * - Max SCC
     - 1--n
     - 1=acyclic, n=all items in one cycle

**Limitations**: No ground truth (consistency ≠ accuracy). Synthetic
vignettes. Deterministic only (stochastic RUM planned). Single model
family.
