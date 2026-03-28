LLM Consistency
===============

Do LLMs have stable action rankings, or does the ranking change depending
on which alternatives are shown? We build preference graphs from LLM
decisions and check for cycles.

.. code-block:: text

   Pipeline:
   ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
   │ Vignettes  │───▶│ Query LLM  │───▶│ Build pref │───▶│ Test SARP  │
   │ (10 per    │    │ (15 menus  │    │ graph per  │    │ + IIA per  │
   │ scenario)  │    │ per vig.)  │    │ vignette   │    │ vignette   │
   └────────────┘    └────────────┘    └────────────┘    └────────────┘

   5 scenarios × 5 prompts × 10 vignettes × 15 menus = 3,750 decisions (v2 det.)
   + 20 reps at temp=0.7 = 75,000 decisions (v2 stochastic)

Setup
-----

5 enterprise LLM deployment scenarios, each with 5 actions and 5 system
prompt strategies. gpt-4o-mini at temp=0 (deterministic) and temp=0.7
(stochastic, K=20 reps).

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Scenario
     - 5 Actions
   * - Support
     - auto-reply KB, bug ticket, billing, account mgr, escalate VP
   * - Alert
     - auto-resolve, P3 ticket, page on-call, incident channel, runbook
   * - Content
     - approve, warning, hide, remove+strike, suspend+legal
   * - Jobs
     - reject, hold, phone screen, technical, fast-track
   * - Procurement
     - auto-approve, tag, request quotes, escalate, deny

Prompts: *minimal, decision-tree, conservative, aggressive, chain-of-thought.*

Results
-------

For each vignette, we fix the input and vary only the menu. Preference
graph cycles are genuine — the LLM's ranking depends on which alternatives
are shown.

.. list-table:: SARP pass rate by scenario × prompt (% of 10 vignettes, deterministic)
   :header-rows: 1
   :widths: 18 13 13 13 13 13 13

   * -
     - Min
     - DecTree
     - Conserv
     - Aggress
     - CoT
     - Mean
   * - Support
     - 90
     - 80
     - 100
     - 80
     - 90
     - 88
   * - Alert
     - 80
     - 100
     - 90
     - 100
     - 90
     - 92
   * - Content
     - 90
     - 80
     - 70
     - 90
     - 80
     - 82
   * - Jobs
     - 70
     - 60
     - 80
     - 80
     - 80
     - 74
   * - Procurement
     - 70
     - 100
     - 70
     - 90
     - 90
     - 84

.. list-table:: SARP pass rate by vignette difficulty tier
   :header-rows: 1
   :widths: 18 16 16 16 16 16

   * -
     - Clear
     - Binary
     - Ambig.
     - Advers.
     - Mean
   * - Support
     - 87
     - 93
     - 90
     - 80
     - 88
   * - Alert
     - 93
     - 100
     - 90
     - 80
     - 92
   * - Content
     - 60
     - 80
     - 100
     - 100
     - 82
   * - Jobs
     - 87
     - 67
     - 70
     - 70
     - 74
   * - Procurement
     - 93
     - 73
     - 90
     - 80
     - 84

.. list-table:: IIA violations + stochastic variation (% menus with mixed responses at temp=0.7)
   :header-rows: 1
   :widths: 18 12 12 12 12 12 12

   * -
     - IIA count
     - Min
     - DecTree
     - Conserv
     - Aggress
     - CoT
   * - Support
     - 3
     - 8%
     - 20%
     - 0%
     - 11%
     - 11%
   * - Alert
     - 2
     - 7%
     - 12%
     - 8%
     - 9%
     - 7%
   * - Content
     - 9
     - 19%
     - 17%
     - 23%
     - 5%
     - 27%
   * - Jobs
     - 15
     - --
     - --
     - --
     - --
     - --
   * - Procurement
     - 8
     - --
     - --
     - --
     - --
     - --

*-- = stochastic data collection in progress.*

Do Preference Graphs Add Value?
-------------------------------

Yes. Three things no accuracy benchmark reveals:

1. **Decoy effects exist in LLMs.** Adding a third option changes the
   ranking between two others — 15 times in job screening. The preference
   graph catches this as a cycle. Accuracy testing cannot.

2. **Consistency varies by scenario.** Alert triage (92%) vs job screening
   (74%). This ranking follows from preference graph structure (ordinal
   actions = fewer cycles), not task complexity.

3. **Prompt effects are scenario-dependent.** Decision-tree prompts are
   the most consistent on alert triage (100%) and worst on job screening
   (60%). Conservative is best on support (100%) but worst on content
   review (70%). Only per-vignette SARP testing reveals this.

What the stochastic experiment adds: at temp=0.7, **88-92% of menus
produce identical choices across 20 reps.** gpt-4o-mini is overwhelmingly
deterministic. The 8-18% that vary concentrate on the same action pairs
that create preference graph cycles in the deterministic condition.

Reproduce
---------

.. code-block:: bash

   pip install prefgraph openai
   export OPENAI_API_KEY=your_key
   cd examples

   # v2 deterministic (3,750 calls, ~$2)
   python -m applications.llm_benchmark.v2.generate_vignettes --all
   python -m applications.llm_benchmark.v2.run_benchmark --all --stage 1
   python -m applications.llm_benchmark.v2.analyze --all

   # v2 stochastic (75,000 calls, ~$40)
   python -m applications.llm_benchmark.v2.run_benchmark --all --stage 2 --k 20

Appendix
--------

Pipeline detail
~~~~~~~~~~~~~~~

.. code-block:: text

   1. VIGNETTES: 10 per scenario, curated across 4 tiers
      (clear, binary, ambiguous, adversarial). Generated by gpt-4o-mini.

   2. MENUS: For each vignette, present ALL C(5,2)=10 pairwise menus
      + 5 size-3 menus = 15 menus. Same vignette, different options shown.

   3. QUERY: For each (vignette, menu, prompt), call gpt-4o-mini.
      Deterministic: temp=0, 1 response.
      Stochastic: temp=0.7, K=20 responses.

   4. BUILD GRAPH: Each choice adds directed edges from chosen item
      to all unchosen items in the menu. One graph per (vignette, prompt).

   5. TEST: SARP on the preference graph (is it acyclic?).
      IIA: compare pairwise choice in {A,B} vs A-vs-B in {A,B,C}.

Code
~~~~

.. code-block:: python

   from prefgraph import MenuChoiceLog
   from prefgraph.algorithms.abstract_choice import validate_menu_sarp

   log = MenuChoiceLog(
       menus=[frozenset(r["menu"]) for r in records],
       choices=[r["choice"] for r in records],
   )
   result = validate_menu_sarp(log)
   # result.is_consistent → bool
   # result.violations → list of cycles in preference graph

Metrics
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Metric
     - Range
     - Meaning
   * - SARP pass rate
     - 0--100%
     - % of vignettes where preference graph is acyclic
   * - HM efficiency
     - 0--1
     - Fraction of items in largest acyclic subgraph
   * - IIA violations
     - 0--n
     - Third option flips a pairwise edge direction
   * - % mixed (stoch.)
     - 0--100%
     - % of menus with different choices across K reps

Limitations
~~~~~~~~~~~

No ground truth (consistency ≠ accuracy). Synthetic vignettes. Single
model family. Stochastic data still collecting for 2 scenarios.
