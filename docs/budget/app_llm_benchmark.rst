LLM Consistency
===============

Do LLMs have stable action rankings, or does the ranking change when
different alternatives are shown? We build preference graphs from LLM
decisions and check for cycles.

**TL;DR.** GPT-4o-mini makes structurally consistent decisions most of the
time --- 74--92% of vignettes pass SARP at temp=0, and majority-vote
over 20 stochastic reps barely changes the picture (96--98% agreement).
But the inconsistency that *does* exist is not random: it clusters on
adjacent-severity action pairs, follows predictable compromise and
anchoring patterns borrowed from human behavioral economics (Simonson
1989), and survives temperature averaging. Job screening is the worst
offender (74% pass, 15 IIA violations); alert triage is the cleanest
(92%, 2 violations). No single prompt is universally best ---
decision-tree hits 100% on procurement but 60% on jobs.

.. code-block:: text

   ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
   │ Vignettes  │───▶│ Query LLM  │───▶│ Build pref │───▶│ Test SARP  │
   │ (10 per    │    │ (15 menus  │    │ graph per  │    │ + IIA per  │
   │ scenario)  │    │ per vig.)  │    │ vignette   │    │ vignette   │
   └────────────┘    └────────────┘    └────────────┘    └────────────┘

Setup
-----

5 enterprise scenarios, 5 actions each, 5 system prompts. gpt-4o-mini.
For each vignette: fix the input, present all C(5,2)=10 pairwise menus
+ 5 size-3 menus. Deterministic (temp=0) and stochastic (temp=0.7, K=20).

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Scenario
     - Actions
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

Deterministic Results (temp=0)
------------------------------

Each cell = % of 10 vignettes where the preference graph is acyclic.

.. list-table:: SARP pass rate by prompt
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

.. list-table:: SARP pass rate by vignette difficulty
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

.. list-table:: IIA violations
   :header-rows: 1
   :widths: 20 15 15 20

   * -
     - Deterministic
     - Stochastic
     - Agreement
   * - Support
     - 3
     - 3
     - 95.8%
   * - Alert
     - 2
     - 3
     - 96.6%
   * - Content
     - 9
     - 12
     - 95.5%
   * - Jobs
     - 15
     - 14
     - 97.6%
   * - Procurement
     - 8
     - 6
     - 97.7%

*IIA violation = adding a third option flips the pairwise preference.
Stochastic = majority-vote from K=20 reps. Agreement = % of menus
where temp=0 and temp=0.7 majority match. Procurement stochastic
based on 82% of expected data (41/50 vignette-prompt combos).*

Stochastic Results (temp=0.7, K=20)
------------------------------------

.. list-table:: SARP pass rate (majority-vote)
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
     - 90
     - 90
     - 100
     - 90
   * - Alert
     - 80
     - 90
     - 90
     - 100
     - 90
     - 90
   * - Content
     - 90
     - 80
     - 60
     - 80
     - 70
     - 76
   * - Jobs
     - 80
     - 60
     - 80
     - 80
     - 90
     - 78
   * - Procurement
     - 78
     - 100
     - 75
     - 88
     - 75
     - 83

.. list-table:: % of menus with mixed responses
   :header-rows: 1
   :widths: 18 13 13 13 13 13 13

   * -
     - Min
     - DecTree
     - Conserv
     - Aggress
     - CoT
     - All
   * - Support
     - 8
     - 20
     - 5
     - 11
     - 11
     - 11
   * - Alert
     - 8
     - 11
     - 7
     - 9
     - 7
     - 8
   * - Content
     - 14
     - 7
     - 16
     - 5
     - 17
     - 12
   * - Jobs
     - 9
     - 10
     - 9
     - 5
     - 9
     - 8
   * - Procurement
     - 5
     - 8
     - 24
     - 16
     - 5
     - 12

.. list-table:: SARP pass rate by tier (majority-vote)
   :header-rows: 1
   :widths: 18 16 16 16 16 16

   * -
     - Clear
     - Binary
     - Ambig.
     - Advers.
     - Mean
   * - Support
     - 93
     - 100
     - 80
     - 80
     - 90
   * - Alert
     - 87
     - 100
     - 90
     - 80
     - 90
   * - Content
     - 47
     - 87
     - 80
     - 100
     - 76
   * - Jobs
     - 87
     - 73
     - 80
     - 70
     - 78
   * - Procurement
     - 87
     - 67
     - 100
     - 100\*
     - 83

*\*Procurement adversarial based on 1 observed vignette.*

Patterns
--------

IIA violations are not evenly distributed. Decision-tree is the IIA
hotspot on job screening (6 of 14 stochastic violations), while it
scores 100% SARP on procurement. Conservative leads on content
moderation (4 of 12). The same prompt can be the most and least
consistent depending on the decision domain.

.. list-table:: IIA violations by prompt (stochastic, top entries)
   :header-rows: 1
   :widths: 22 22 16

   * - Scenario
     - Prompt
     - IIA count
   * - Jobs
     - decision_tree
     - 6
   * - Content
     - conservative
     - 4
   * - Content
     - decision_tree
     - 3
   * - Content
     - aggressive
     - 2
   * - Jobs
     - minimal
     - 2
   * - Procurement
     - minimal
     - 2
   * - Procurement
     - conservative
     - 2

**Compromise effect (job screening).** Across 11 of 14 job-screen
IIA violations, the mechanism is the same: adding an extreme option
pushes the choice toward the middle. In vignette v04 under
chain-of-thought, the model prefers *phone_screen* over
*hold_for_review* pairwise. But present {auto_reject, hold,
phone_screen} and *hold* wins --- adding the worst option makes hold
look like the safe middle ground. The reverse also occurs: adding
*fast_track* pushes the choice from *auto_reject* to
*technical_interview*. This is the classic compromise effect from
behavioral economics (Simonson 1989), now confirmed in LLM outputs.

**Severity anchor (content moderation).** In vignette v01 under
decision-tree, the model prefers *remove_and_strike* over
*suspend_and_legal* pairwise. But present {approve, remove, suspend}
and *suspend* wins --- adding "approve" (the lenient end) anchors
judgment toward severity. The reverse also holds: adding "suspend"
(extreme) makes "remove" look moderate and preferable to "approve."
The LLM anchors to the extremes of whatever menu is shown.

**Parse failures as a policy floor.** In content-review v01, 5 of 15
menus return PARSE_FAIL --- all menus containing *only* mild options
(approve, content_warning, hide_from_feed). The model refuses to pick
any mild action for graphic content, outputting an off-menu severe
action instead. This is not noise: it reveals a hard policy constraint
the LLM will not violate even when no on-menu action satisfies it.
SARP counts this as a violation; it is better understood as a revealed
constraint.

Findings
--------

- **Job screening is the least consistent scenario.** 74% deterministic
  SARP pass, 78% stochastic. 15 deterministic + 14 stochastic IIA violations.
  Adding a third candidate changes which of two is preferred.
- **Content moderation "clear" vignettes pass only 47% stochastically.**
  Even unambiguous posts produce menu-dependent severity judgments.
  12 stochastic IIA violations — majority-voting doesn't eliminate
  context effects.
- **Decision-tree is the only prompt to hit 100% on any scenario**
  (procurement stochastic). But it scores 60% on jobs. Conservative
  scores 90% on support, 60% on content. No universal best prompt.
- **Alert triage is the most consistent** (90% stochastic SARP). Actions
  have a clear ordinal severity structure that resists menu effects.
- **Procurement conservative prompt has 24% mixed menus** — the highest
  of any scenario-prompt combination. Spending authority decisions are
  genuinely sensitive to sampling temperature when the prompt emphasizes
  caution.
- **Stochastic sampling barely changes rankings.** 96-98% of menus agree
  between temp=0 and temp=0.7 majority vote. Only 8-12% of menus produce
  mixed responses across 20 reps. Inconsistency is structural, not noise.

Reproduce
---------

.. code-block:: bash

   pip install prefgraph openai
   export OPENAI_API_KEY=your_key
   cd examples

   # Deterministic (3,750 calls)
   python -m applications.llm_benchmark.v2.generate_vignettes --all
   python -m applications.llm_benchmark.v2.run_benchmark --all --stage 1
   python -m applications.llm_benchmark.v2.analyze --all

   # Stochastic (75,000 calls)
   python -m applications.llm_benchmark.v2.run_benchmark --all --stage 2 --k 20
   python -m applications.llm_benchmark.v2.analyze --all --stage 2

Appendix
--------

Pipeline
~~~~~~~~

.. code-block:: text

   1. VIGNETTES: 10 per scenario, 4 tiers (clear, binary, ambiguous,
      adversarial). Generated by gpt-4o-mini.

   2. MENUS: C(5,2)=10 pairwise + 5 size-3 = 15 menus per vignette.
      Same input, different options shown.

   3. QUERY: gpt-4o-mini. temp=0 (1 response) or temp=0.7 (K=20).

   4. GRAPH: Each choice adds edges: chosen → each unchosen item.

   5. TEST: SARP (acyclic?). IIA (pairwise vs triple comparison).

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
   # result.is_consistent, result.violations

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
     - % of vignettes with acyclic preference graph
   * - IIA violations
     - 0--n
     - Adding a third option flips a pairwise preference
   * - % mixed
     - 0--100%
     - % of menus with different choices across K reps

Limitations
~~~~~~~~~~~

No ground truth (consistency ≠ accuracy). Synthetic vignettes. Single
model family. Procurement stochastic based on 82% of expected data.
Stochastic analysis uses majority-vote aggregation over K=20 reps, not
a formal RUM LP test. For stochastic rationality testing (regularity,
Block-Marschak) see :doc:`../menu/theory_stochastic`.
