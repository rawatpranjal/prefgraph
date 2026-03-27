LLM Enterprise Consistency Benchmark
=====================================

Demonstrated application of PyRevealed's discrete choice axioms (SARP,
Houtman-Maks, IIA) to audit LLM decision-making across 5 enterprise
deployment scenarios. 13,750 API decisions, two experiment iterations.

What PyRevealed Reveals That Other Tools Don't
-----------------------------------------------

Standard LLM evaluation measures **accuracy** (% correct) and **latency**
(ms per call). Neither detects a failure mode that matters in production:
**menu-dependent preferences**.

An LLM might correctly handle 90% of support tickets. But when presented
with {escalate, auto-reply}, it picks escalate — and when presented with
{escalate, auto-reply, bug-ticket}, it switches to auto-reply. The third
option changed the ranking. No accuracy benchmark catches this.

PyRevealed's ``validate_menu_sarp()`` tests exactly this: given a set of
choices from varying menus, does a consistent ranking exist? If not,
``compute_menu_efficiency()`` quantifies how much of the behavior is
rationalizable. These are the same tools economists use to test whether
human consumers are rational (Varian 1982), now applied to LLM deployment.

**What you get that you can't get elsewhere:**

- **SARP pass rate**: Does the LLM have a stable action ranking for this
  input? Binary, per-vignette.
- **Houtman-Maks efficiency**: What fraction of actions can be consistently
  ranked? Tells you which items participate in cycles.
- **IIA violation detection**: Does adding a third option flip the preference
  between two others? The "decoy effect" — classic in behavioral economics,
  unmeasured in LLM evaluation until now.
- **Preference graph structure**: Which action pairs create cycles? Which
  items are in the same strongly connected component? Directly actionable
  for guardrail design.

Scenarios and Prompts
---------------------

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Scenario
     - Input
     - 5 Actions
   * - Support Router
     - Customer ticket
     - auto-reply KB, bug ticket, billing, account mgr, escalate VP
   * - Alert Triage
     - Monitoring alert
     - auto-resolve, P3 ticket, page on-call, incident channel, runbook
   * - Content Review
     - Flagged post
     - approve, warning, hide, remove+strike, suspend+legal
   * - Job Screen
     - Resume + JD
     - reject, hold, phone screen, technical, fast-track
   * - Procurement
     - Purchase request
     - auto-approve, tag, request quotes, escalate, deny

5 prompt strategies per scenario: *minimal*, *decision tree*, *conservative*,
*aggressive*, *chain-of-thought*. Full production prompts (100--300 words).

Experiment v1: Pooled SARP (10,000 decisions)
----------------------------------------------

**Design**: 5 scenarios x 5 prompts x 2 models x 200 vignettes x 1 menu each.
Different vignette per trial. SARP tested across all 200 trials per group.

**Result**: 50/50 groups fail SARP. HM=0.60 everywhere. Zero variation.

**What we learned:**

- At temp=0, both gpt-4o-mini and o4-mini fail SARP on every scenario-prompt
  combination with maximum violations (10/10 pairwise cycles)
- Observation-level bootstrap HM is 0.95 [0.93, 0.97] — 95% of individual
  decisions are locally rationalizable
- Permutation p=1.000 everywhere — LLMs are far more consistent than random
- gpt-4o-mini and o4-mini are indistinguishable on consistency
- Prompts shift choice distributions (KL divergence 0.02--0.84) but never
  eliminate preference cycles
- Conservative prompts flip 6/10 pairwise preferences vs aggressive, but both
  produce 10 SARP violations

**Design flaw identified:** Each trial uses a different vignette. SARP
violations reflect the LLM correctly adapting to different inputs, not
actual menu-dependent inconsistency. EDA confirmed: 50% of vignettes
produce identical choices across all 5 prompts.

Experiment v2: Per-Vignette SARP (3,750 decisions)
----------------------------------------------------

**Design fix**: Hold vignette constant, vary only the menu. For each of 10
curated vignettes per scenario, present all C(5,2)=10 pairwise menus + 5
size-3 menus = 15 menus. Test SARP within each vignette.

**Scale**: 5 scenarios x 5 prompts x 10 vignettes x 15 menus = 3,750 calls.
gpt-4o-mini only, temp=0.

v2 Results
~~~~~~~~~~

SARP pass rates by scenario and prompt:

.. list-table::
   :header-rows: 1
   :widths: 22 13 13 13 13 13

   * - Scenario
     - Minimal
     - Dec. Tree
     - Conserv.
     - Aggress.
     - CoT
   * - Support Router
     - 90%
     - 80%
     - 100%
     - 80%
     - 90%
   * - Alert Triage
     - 80%
     - 100%
     - 90%
     - 100%
     - 90%
   * - Content Review
     - 90%
     - 80%
     - 70%
     - 90%
     - 80%
   * - Job Screen
     - 70%
     - 60%
     - 80%
     - 80%
     - 80%
   * - Procurement
     - 70%
     - 100%
     - 70%
     - 90%
     - 90%

*Percentage of 10 vignettes where SARP is satisfied (deterministic, temp=0).
Higher = more consistent for that prompt-scenario combination.*

SARP pass rates by vignette difficulty tier:

.. list-table::
   :header-rows: 1
   :widths: 22 16 16 16 16

   * - Scenario
     - Clear
     - Binary
     - Ambiguous
     - Adversarial
   * - Support Router
     - 87%
     - 93%
     - 90%
     - 80%
   * - Alert Triage
     - 93%
     - 100%
     - 90%
     - 80%
   * - Content Review
     - 60%
     - 80%
     - 100%
     - 100%
   * - Job Screen
     - 87%
     - 67%
     - 70%
     - 70%
   * - Procurement
     - 93%
     - 73%
     - 90%
     - 80%

*Averaged across 5 prompts per tier. Clear = unambiguous input, Binary = 2
actions compete, Ambiguous = 3+ plausible, Adversarial = designed to trigger
menu effects.*

IIA violations (menu-dependence):

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Scenario
     - IIA violations
     - Interpretation
   * - Support Router
     - 3
     - Adding a third option rarely changes pairwise preference
   * - Alert Triage
     - 2
     - Nearly menu-independent decisions
   * - Content Review
     - 9
     - Moderate menu effects — adding options shifts moderation decisions
   * - Job Screen
     - 15
     - Strongest menu effects — candidate ranking changes with alternatives shown
   * - Procurement
     - 8
     - Moderate menu effects on spending decisions

*IIA = Independence of Irrelevant Alternatives. A violation means adding a
third option to a pairwise menu changed which of two options is preferred.*

Standout Empirical Findings
~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Synthetic vignettes, single model family. Illustrative, not definitive.*

**Decoy effects:**

- **Job screening has 15 IIA violations** — the strongest decoy effects
  of any scenario. Showing a third candidate changes which of two
  candidates the LLM prefers. This is the hiring discrimination vector
  nobody is testing for.
- Content review has 9 IIA violations. Alert triage has 2. The more
  subjective the task, the stronger the menu composition effect.

**Consistency varies where you don't expect it:**

- **Content moderation "clear" vignettes only 60% SARP-consistent** —
  the lowest pass rate of any tier in any scenario. Even posts designed
  to be unambiguous produce menu-dependent severity judgments. Moderation
  has no safe inputs.
- **Alert triage is 92% consistent** — the easiest scenario. Infrastructure
  severity has the clearest natural ordering. SARP pass rate correlates
  with how "ordinal" the action space is.

**Prompt effects are scenario-dependent:**

- **Decision-tree prompts score 60% on job screening** — the worst
  prompt-scenario combination. Explicit if/then rules *create*
  inconsistency by introducing more decision boundaries. Simpler
  prompts (aggressive, CoT) score 80%.
- **Conservative prompts hurt procurement and content review** (70% each)
  but are the *best* prompt for support routing (100%). There is no
  universal "most consistent" prompt strategy.
- **Conservative prompts flip 6/10 pairwise preferences** vs aggressive
  in support routing (v1 data). The largest prompt effect by far — yet
  both produce identical SARP failure rates in pooled testing. Only
  per-vignette testing reveals the difference.

**Model comparison (v1):**

- **gpt-4o-mini and o4-mini are indistinguishable on consistency.**
  Reasoning tokens don't buy coherence. Both models produce the same
  preference graph structure across all 50 group-level tests.

**Methodological:**

- **50% of vignettes produce identical choices across all 5 prompts**
  (v1 data). Prompt engineering only matters on the borderline ~47%.
- **v1 (pooled) = 0% pass; v2 (per-vignette) = 60--100% pass.** The
  difference is entirely design — same model, same prompts. Revealed
  preference theory provides the framework to ask the right question.

What PyRevealed Uniquely Reveals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``validate_menu_sarp()``: per-input consistency score — does the LLM
  have a stable action ranking for *this specific* input?
- ``compute_menu_efficiency()``: which action pairs participate in
  preference cycles? Directly actionable for guardrail design.
- IIA detection (pairwise vs triple menus): does adding a third option
  flip the preference between two others? Unmeasured in standard evals.

What This Means for Practitioners
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you deploy an LLM as a decision system (triage, moderation, screening):

- **Audit before deployment.** Run the per-vignette SARP test on your
  specific inputs. A 70% pass rate means 30% of inputs produce decisions
  that depend on which options you show — not just on the input itself.

- **Identify fragile action pairs.** SARP violations tell you exactly which
  pairs of actions form preference cycles. Add guardrails (confidence
  thresholds, human review) specifically for those pairs.

- **Test prompt candidates on consistency, not just accuracy.** Two prompts
  can have identical accuracy but different SARP pass rates. The one with
  higher consistency produces more predictable behavior in production.

- **Measure IIA before designing menus.** If your system shows the LLM
  different action sets in different contexts (e.g., filtering options by
  eligibility), IIA violations mean the filtered set changes the outcome
  — not just the available options.

Reproduce
---------

.. code-block:: bash

   pip install pyrevealed openai
   export OPENAI_API_KEY=your_key
   cd examples

   # v1 (pooled SARP, 10K decisions)
   python -m applications.llm_benchmark.generate_vignettes --all --trials 200
   python -m applications.llm_benchmark.run_benchmark --all --trials 200
   python -m applications.llm_benchmark.analyze --all

   # v2 (per-vignette SARP, 3.75K decisions)
   python -m applications.llm_benchmark.v2.generate_vignettes --all
   python -m applications.llm_benchmark.v2.run_benchmark --all --stage 1
   python -m applications.llm_benchmark.v2.analyze --all

All stages are resumable. v1 data in ``llm_benchmark/data/``,
v2 data in ``llm_benchmark/v2/data/``.

Appendix: Full Pipeline Documentation
--------------------------------------

Data Generation
~~~~~~~~~~~~~~~

.. code-block:: text

   ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
   │ Vignette Gen      │     │ Menu Gen          │     │ LLM Querying     │
   │                   │     │                   │     │                  │
   │ v1: o4-mini, 200  │     │ v1: 1 random menu │     │ For each         │
   │ random per scen.  │────▶│ per vignette      │────▶│ (vig, menu,      │
   │                   │     │                   │     │  prompt, model): │
   │ v2: gpt-4o-mini,  │     │ v2: ALL 15 menus  │     │  → OpenAI API    │
   │ 10 curated tiers  │     │ per vignette      │     │  → parse choice  │
   │                   │     │ (10 pair + 5 tri) │     │  → append JSONL  │
   └──────────────────┘     └──────────────────┘     └──────────────────┘

Feature Extraction
~~~~~~~~~~~~~~~~~~

Each ``(vignette, prompt)`` group produces a ``MenuChoiceLog``:

.. code-block:: python

   from pyrevealed import MenuChoiceLog
   from pyrevealed.algorithms.abstract_choice import validate_menu_sarp

   log = MenuChoiceLog(
       menus=[frozenset(r["menu"]) for r in records],
       choices=[r["choice"] for r in records],
       item_labels=["auto_reply_kb", "create_bug_ticket", ...],
   )
   result = validate_menu_sarp(log)
   # result.is_consistent, result.violations, result.transitive_closure

Analysis Pipeline
~~~~~~~~~~~~~~~~~

.. code-block:: text

   MenuChoiceLog
       │
       ├─── validate_menu_sarp() ─── SARP pass/fail + cycle list
       ├─── compute_menu_efficiency() ─── HM efficiency (0-1)
       └─── IIA detection ─── compare pairwise vs triple menu choices

Metrics Reference
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Metric
     - Range
     - Interpretation
   * - SARP pass rate
     - [0%, 100%]
     - Fraction of vignettes with transitive action ranking (v2)
   * - HM efficiency
     - [0, 1]
     - Fraction of items in largest consistent subset
   * - IIA violations
     - [0, n_triples]
     - Cases where adding a third option flips a pairwise preference
   * - Max SCC
     - [1, n]
     - Largest strongly connected component. 1=acyclic, n=all items in one cycle

Limitations
~~~~~~~~~~~

1. **No ground truth**: We measure consistency, not accuracy. A perfectly
   consistent but wrong system would score well.

2. **5 synthetic scenarios**: Results may not generalize to all LLM deployments.

3. **Deterministic only (v2 Stage 1)**: SARP assumes deterministic choice.
   Stochastic testing (RUM, K=20 repetitions at temp=0.7) planned for Stage 2.

4. **Single model**: v2 tests gpt-4o-mini only. v1 showed o4-mini is
   indistinguishable, but other model families (Claude, Gemini) untested.
