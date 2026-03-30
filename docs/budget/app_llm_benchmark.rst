Detecting Inconsistency in AI Agents
======================================

**TL;DR.** At temperature 0, between 74 and 92 percent of scenario
configurations pass SARP. Alert Triage is the most consistent domain
(92 percent). Jobs Task is the least (74 percent), with frequent IIA
violations where adding a third option reverses the preference between
two existing ones. At temperature 0.7, we evaluate consistency through
Random Utility Models rather than majority votes.

Example
^^^^^^^

   *A customer has reported frequent service outages that impact their
   ability to access critical features, significantly affecting their
   business operations. They have already tried multiple troubleshooting
   steps from the documentation but the issue persists.*

   *Query.* Route this ticket to the best destination.

   *Menu A.* ``create_bug_ticket`` / ``route_account_mgr``

   *Menu B.* ``create_bug_ticket`` / ``route_account_mgr`` / ``auto_reply_kb``

   *Does adding a third option change the choice between the first two?
   That is the question revealed preference theory answers.*

Do LLMs have stable action rankings, or does the ranking change when
different alternatives are shown? We collected roughly 78,750 GPT-4o-mini
API calls — 3,750 deterministic (temp=0) and 75,000 stochastic (temp=0.7,
20 repetitions per menu) — across 5 enterprise scenarios, 50 vignettes, 5
prompt frameworks, and 15 menus per vignette. We built preference graphs
from these responses and checked for cycles.

.. _llm-setup:


Setup
-----

Each **scenario** is an enterprise decision task with five possible **actions**. A **vignette** is a concrete input (a support ticket, alert payload, or resume) that the LLM must act on. A **prompt** is the system prompt persona framing how the LLM approaches the decision. A **menu** is the subset of actions shown for a given vignette. We present all 10 pairwise menus and 5 size-3 menus per vignette, giving 15 menus total. The model is gpt-4o-mini, tested at temperature 0 (deterministic) and temperature 0.7 with 20 repetitions per menu (stochastic).

.. list-table::
   :header-rows: 1
   :widths: 14 50 36

   * - Scenario
     - Actions
     - Prompts
   * - **Support**
     - auto-reply KB, bug ticket, billing, account mgr, escalate VP
     - **minimal** (bare instruction), **decision-tree** (if/then rules),
       **conservative** (prefer escalation), **aggressive** (prefer automation),
       **chain-of-thought** (numbered reasoning)
   * - **Alert**
     - auto-resolve, P3 ticket, page on-call, incident channel, runbook
     - same 5 prompts, adapted per scenario
   * - **Content Moderation Task**
     - approve, warning, hide, remove+strike, suspend+legal
     - same 5
   * - **Jobs Task**
     - reject, hold, phone screen, technical, fast-track
     - same 5
   * - **Procurement**
     - auto-approve, tag, request quotes, escalate, deny
     - same 5

.. _llm-how-to-read:

How to read the results
-----------------------

**Perfect Consistency (SARP)** tests whether the model operates from a single strict ranking of actions, with no contradictions across all menus. **Menu Independence (IIA)** tests whether adding a third option changes the preference between two existing options. **Probabilistic Consistency (RUM)** takes 20 samples per menu at temperature 0.7 and tests whether the resulting choice frequencies can be explained by a stable distribution of preferences rather than noise.

.. _llm-why-design:

Why this design
---------------

Holding the vignette constant and only changing the menu isolates menu
effects.

.. _llm-det-results:

Results 1: Deterministic (temp=0)
---------------------------------

Each cell shows the percentage of vignette-prompt configurations that pass SARP.

.. list-table:: SARP Pass Rate by Prompt Framework (%)
   :header-rows: 1
   :widths: 18 13 13 13 13 13 13

   * - Scenario
     - Minimal
     - Decision Tree
     - Conservative
     - Aggressive
     - Chain-of-Thought
     - Mean
   * - **Support**
     - 90
     - 80
     - 100
     - 80
     - 90
     - 88
   * - **Alert**
     - 80
     - 100
     - 90
     - 100
     - 90
     - 92
   * - **Content Moderation Task**
     - 90
     - 80
     - 70
     - 90
     - 80
     - 82
   * - **Jobs Task**
     - 70
     - 60
     - 80
     - 80
     - 80
     - 74
   * - **Procurement**
     - 70
     - 100
     - 70
     - 90
     - 90
     - 84

.. list-table:: SARP Pass Rate by Case Difficulty (%)
   :header-rows: 1
   :widths: 18 16 16 16 16 16

   * - Scenario
     - Clear Cases
     - Binary Cases
     - Ambiguous
     - Adversarial
     - Mean
   * - **Support**
     - 87
     - 93
     - 90
     - 80
     - 88
   * - **Alert**
     - 93
     - 100
     - 90
     - 80
     - 92
   * - **Content Moderation Task**
     - 60
     - 80
     - 100
     - 100
     - 82
   * - **Jobs Task**
     - 87
     - 67
     - 70
     - 70
     - 74
   * - **Procurement**
     - 93
     - 73
     - 90
     - 80
     - 84

.. _llm-iia:

.. list-table:: Independence of Irrelevant Alternatives (IIA) Flips
   :header-rows: 1
   :widths: 50 50

   * - Operational Scenario
     - Reversal Frequency (IIA Count)
   * - **Support**
     - 3
   * - **Alert**
     - 2
   * - **Content Moderation Task**
     - 9
   * - **Jobs Task**
     - 15
   * - **Procurement**
     - 8

Structured prompts (Decision Tree, Conservative) generally outperform unguided Minimal prompts, but the effect is domain-specific. Consistency does not uniformly degrade on harder tasks. Content Moderation is less consistent on clear cases than on adversarial ones, while Jobs Task struggles most on binary decisions. The Jobs Task and Content Moderation also produce the most IIA violations, meaning that adding a third option frequently reverses the preference between the original two.

.. _llm-stoch-results:

.. _llm-rum:

Results 2: Stochastic Choice (RUM)
----------------------------------

We ask the model 20 times per menu at temperature 0.7 and test whether the resulting choice frequencies are consistent with a Random Utility Model.

.. list-table:: RUM Pass Rate by Prompt Framework (%)
   :header-rows: 1
   :widths: 18 13 13 13 13 13

   * - Scenario
     - Minimal
     - Decision Tree
     - Conservative
     - Aggressive
     - Chain-of-Thought
   * - **Support**
     - 50
     - 30
     - 70
     - 50
     - 70
   * - **Alert**
     - 70
     - 60
     - 90
     - 70
     - 80
   * - **Content Moderation Task**
     - 50
     - 60
     - 60
     - 80
     - 50
   * - **Jobs Task**
     - 70
     - 40
     - 60
     - 80
     - 60
   * - **Procurement**\*
     - 67
     - 88
     - 50
     - 38
     - 63

.. list-table:: RUM Pass Rate by Case Difficulty (%)
   :header-rows: 1
   :widths: 18 16 16 16 16

   * - Scenario
     - Clear Cases
     - Binary Cases
     - Ambiguous
     - Adversarial
   * - **Support**
     - 53
     - 67
     - 40
     - 50
   * - **Alert**
     - 67
     - 93
     - 70
     - 60
   * - **Content Moderation Task**
     - 13
     - 80
     - 70
     - 90
   * - **Jobs Task**
     - 80
     - 60
     - 50
     - 50
   * - **Procurement**\*
     - 47
     - 47
     - 100
     - 100

RUM pass rates drop 20 to 30 percentage points below the deterministic SARP rates across most scenarios, confirming that stochastic sampling reveals inconsistencies that single-shot testing misses. The same patterns hold. Content Moderation passes only 13 percent of clear cases under stochastic testing, while Alert remains the most consistent domain. No single prompt framework dominates. Procurement results are based on partial stage 2 coverage.

.. _llm-patterns:

Patterns
--------

.. list-table:: IIA Violations by Framework (Top Entries)
   :header-rows: 1
   :widths: 22 22 16

   * - Scenario
     - Prompt Framework
     - Flips (IIA Count)
   * - **Jobs Task**
     - Decision Tree
     - 6
   * - **Content Moderation Task**
     - Conservative
     - 4
   * - **Content Moderation Task**
     - Decision Tree
     - 3
   * - **Content Moderation Task**
     - Aggressive
     - 2
   * - **Jobs Task**
     - Minimal
     - 2
   * - **Procurement**
     - Minimal
     - 2
   * - **Procurement**
     - Conservative
     - 2

Two mechanisms drive most IIA violations. In the Jobs Task, adding an extreme option (auto-reject or fast-track) shifts the choice toward the middle candidate, which is the classic compromise effect from behavioral economics. In Content Moderation, the model anchors to whichever end of the severity spectrum is present in the menu, so adding a lenient option pushes toward strictness and adding an extreme option pushes toward moderation. The same prompt can be the biggest IIA hotspot in one domain and score 100 percent SARP in another. In severe Content Moderation cases, the model occasionally refuses to pick any mild action at all, revealing a safety floor that SARP counts as a violation but is better read as a hard constraint.

.. _llm-findings:

Findings
--------

The Jobs Task is the least consistent scenario at 74 percent deterministic SARP and 15 IIA violations, driven by the compromise effect. Content Moderation shows that clear cases are paradoxically less stable than adversarial ones, with only 13 percent stochastic consistency on clear vignettes. There is no universal best prompt. Decision Tree reaches 100 percent on Procurement but scores 60 percent on Jobs Task. Alert Triage is the most consistent domain overall, likely because its actions form a natural severity ladder. Only 8 to 12 percent of menus produce mixed responses across 20 stochastic repetitions, which means the model stays confident in its ranking even when that ranking contradicts itself across different menus. The inconsistency is structural, not noise.

.. _llm-cost:

Computational Cost
------------------

Measured on Apple M-series (11 cores). The analysis runs SARP + HM on each (vignette, prompt) configuration. 5 scenarios × 10 vignettes × 5 prompts = 250 configurations, of which 150 have sufficient data (≥ 5 menu observations each).

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 15

   * - Stage
     - Configs
     - Wall Time
     - Peak Memory
   * - SARP + HM analysis
     - 150
     - 0.5 s
     - 19 MB

At 3.3 ms per configuration, the consistency analysis is negligible compared to the LLM API calls that generate the data (~3,750 calls for deterministic, ~75,000 for stochastic).

.. _llm-replication:

Replication
-----------

All vignettes and LLM responses are bundled in the repository — no API key needed to reproduce the analysis:

- `Vignettes (JSONL) <https://github.com/rawatpranjal/PrefGraph/tree/main/examples/applications/llm_benchmark/v2/data/vignettes>`_
- `Responses (JSONL) <https://github.com/rawatpranjal/PrefGraph/tree/main/examples/applications/llm_benchmark/v2/data/responses>`_
- `Results (JSON) <https://github.com/rawatpranjal/PrefGraph/tree/main/examples/applications/llm_benchmark/v2/data/results>`_

To regenerate from scratch (requires OpenAI API key):

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

.. _llm-appendix:

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
   :widths: 24 12 64

   * - Metric
     - Range
     - Meaning
   * - Perfect Consistency (SARP)
     - 0--100%
     - % of configurations with completely logical, non-contradictory preferences
   * - Menu Independence (IIA)
     - 0--n
     - Frequency of context-dependent preference reversals
   * - Mixed Choices (%)
     - 0--100%
     - % of menus where probabilistic sampling yielded varied responses

Limitations
~~~~~~~~~~~

Consistency measures internal coherence, not correctness. A model that always picks the same wrong answer scores perfectly on SARP. The vignettes are synthetic and drawn from a single model family (GPT-4o-mini), so we do not know whether these rates generalize across architectures or predict behavior on production tasks. We also have no human baseline for comparison.

Menus in this study contain two to three actions drawn from a five-action set. Production deployments face larger action spaces where consistency may degrade further, but this study does not test that.

For theoretical details on stochastic rationality testing see :doc:`../menu/theory_stochastic`.

Worked Example
~~~~~~~~~~~~~~

One row from the benchmark, showing every concept in action:

- **Scenario:** Procurement Approval
- **Vignette:** *"A department has submitted a procurement request for
  office supplies totaling $150, which is well within the approved budget
  of $500 for the month. All items requested are standard supplies that
  have been pre-approved in the procurement policy without the need for
  further review."* (tier: clear)
- **Prompt:** minimal --- *"Process procurement requests. Pick one action
  from the list."*
- **Actions:** auto-approve, tag, request quotes, escalate, deny
- **Menu:** {auto-approve, tag} (one of 15 menus shown for this vignette)
- **Choice:** auto-approve

This vignette is "clear" --- reasonable people would agree on
auto-approve. The model does too, consistently across all 15 menus
(SARP pass). Now contrast with an adversarial vignette from job
screening:

- **Scenario:** Job Application Screen
- **Vignette:** *"The candidate has 5 years of Python experience building
  data pipelines at a mid-size fintech. No distributed systems work, but
  led a Kafka migration that reduced processing latency by 40%. Bootcamp
  grad, no CS degree."* (tier: binary)
- **Prompt:** decision-tree --- explicit if/then rules for each action
- **Menu 1:** {hold, phone-screen} --- model picks **phone-screen**
- **Menu 2:** {auto-reject, hold, phone-screen} --- model picks **hold**

Adding auto-reject (the worst option) flips the preference from
phone-screen to hold. This is an IIA violation --- the compromise
effect in action. The model gravitates toward the "middle" of
whatever menu it sees.
