Preference Graphs
==================

Build a preference graph from observed choices. Test it for cycles.
Score how rational it is. PrefGraph does this at scale — for shoppers,
recommender clicks, or LLM decisions.

Every choice reveals a preference edge: "A was chosen over B." These
edges form a **preference graph**. If the graph is acyclic, a coherent
ranking exists. If it has cycles, the choices are inconsistent — and
PrefGraph scores *how much* (0 = incoherent, 1 = perfectly rational).

.. raw:: html

   <div class="feature-grid">
     <div class="feature-card">
       <h3>Build Preference Graphs</h3>
       <p>From budget data (prices × quantities) or menu data (choice sets × picks), build the revealed preference graph. Each choice adds a directed edge.</p>
     </div>
     <div class="feature-card">
       <h3>Detect Cycles</h3>
       <p>Test GARP, SARP, WARP — does the preference graph have cycles? If yes, no coherent ranking exists. Floyd-Warshall transitive closure + Tarjan SCC decomposition.</p>
     </div>
     <div class="feature-card">
       <h3>Score Consistency</h3>
       <p>CCEI (0–1), Money Pump Index, Houtman-Maks, Varian Efficiency — all measure how close the preference graph is to acyclic. Each user gets a rationality score.</p>
     </div>
     <div class="feature-card">
       <h3>Two API Layers</h3>
       <p><strong>Engine</strong> — batch scoring via Rust/Rayon (49K users/sec). <strong>Functions</strong> — single-user deep dives with full preference graph output.</p>
     </div>
     <div class="feature-card">
       <h3>Graph Algorithms in Rust</h3>
       <p>Floyd-Warshall, Tarjan SCC, Karp's max-mean cycle, HiGHS LP — all in Rust with Rayon parallelism. Python fallback available.</p>
     </div>
     <div class="feature-card">
       <h3>Post-Estimation</h3>
       <p>From the preference graph: recover utility functions, compute welfare bounds, test separability, detect IIA violations (decoy effects).</p>
     </div>
   </div>

.. code-block:: bash

   pip install prefgraph

Why Preference Graphs?
-----------------------

Most approaches assume a model of preferences first, then fit parameters.
Preference graphs work backwards: take raw choices, build the revealed
preference graph, and ask "is it acyclic?" No assumptions about what
people want — just a consistency check on what they did.

A cycle in the preference graph (A > B > C > A) means no ranking
explains the choices. Detecting and measuring these cycles is what
PrefGraph does — using Floyd-Warshall, Tarjan SCC, and Karp's algorithm
on the preference graph. Afriat (1967), Varian (1982).

.. raw:: html

   <div style="margin: 1.5em 0;"></div>

.. raw:: html

   <div style="display: flex; gap: 20px; justify-content: space-between; flex-wrap: wrap;">
     <div style="width: 48%;">
       <img src="_static/budget_hero.gif" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" alt="Budget Choices">
       <p class="gif-caption"><strong>Budget choices.</strong> A shopper buys goods at given prices. Budget lines show what was affordable. When chosen bundles sit inside each other's budget lines, that's a contradiction — CCEI measures how much you'd need to shrink budgets to fix it. Afriat (1967).</p>
     </div>
     <div style="width: 48%;">
       <img src="_static/menu_hero.gif" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" alt="Menu Choices">
       <p class="gif-caption"><strong>Menu choices.</strong> A user picks one option from a set. Picking Laptop over Tablet in one menu, then Tablet over Laptop in another, is a contradiction — HM counts how many choices to throw out to fix it. Houtman &amp; Maks (1985).</p>
     </div>
   </div>

Two Core Data Types
-------------------

.. code-block:: text

   Budget Data (prices + quantities)          Menu Data (sets + picks)
   ───────────────────────────────            ──────────────────────────
   1. Load → BehaviorLog                     1. Load → MenuChoiceLog
   2. Rational? → validate_consistency()     2. Rational? → validate_menu_sarp()
   3. How much? → compute_integrity_score()  3. How much? → compute_menu_efficiency()
   4. Segment users by score                 4. Segment users by score

Performance
-----------

Benchmarked on synthetic data, T=15 observations, 10 goods, M1 Mac:

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Configuration
     - Throughput
     - Latency (per agent)
     - Complexity
   * - **GARP only**
     - ~49,000 agents/sec
     - 20 μs
     - O(T²)
   * - **GARP + CCEI**
     - ~2,400 agents/sec
     - 420 μs
     - O(T² log T)
   * - **Full suite** (GARP, CCEI, MPI, HARP)
     - ~2,000 agents/sec
     - 500 μs
     - O(T³)
   * - **Menu** (SARP + WARP + HM)
     - ~19,000 agents/sec
     - 50 μs
     - O(N³)

Empirical benchmarks: CCEI ≈ 0.88 in lab experiments (Choi et al., 2014); HM ≈ 0.70–0.85 in recommendation click data.

E-commerce Benchmarks
---------------------

Seven public datasets, 167K users, LightGBM with 5-fold stratified CV.
Full results: :doc:`benchmarks_ecommerce`.

.. list-table::
   :header-rows: 1
   :widths: 22 10 18 12 12

   * - Dataset
     - N
     - Target
     - Baseline
     - +RP
   * - Dunnhumby
     - 2,222
     - High Spender
     - 0.958
     - 0.960
   * - Open E-Commerce
     - 4,694
     - Churn
     - 0.841
     - **0.847**
   * - H&M
     - 46,757
     - High Spender
     - 0.763
     - 0.762
   * - Instacart
     - 50,000
     - High Value
     - 0.967
     - 0.967
   * - REES46
     - 8,832
     - High Engagement
     - 0.996
     - 0.996
   * - Taobao
     - 4,239
     - High Engagement
     - 0.913
     - 0.913
   * - Tenrec
     - 50,000
     - High Engagement
     - 0.993
     - 0.993

RP features add 0–0.7% AUC over strong RFM baselines. The lift is real but modest.

Reading List (2020+)
--------------------

Recent applied work using revealed preference methods on real-world data:

- **Chen et al. (2024)** — "Rationality of LLMs." CCEI testing on GPT budget-allocation tasks. *PNAS*.
- **Cazzola & Daly (2024)** — "Rank-preference consistency in recommender systems." SARP as evaluation metric. *Working paper*.
- **Deb, Kitamura, Quah & Stoye (2023)** — GAPP: price-preference tests + population welfare bounds. *Review of Economic Studies*.
- **Demuynck & Rehbeck (2023)** — Integer programming for goodness-of-fit measures at scale. *Economic Theory*.
- **Smeulders, Crama & Spieksma (2021)** — Collective rationality tests via integer programming. *Operations Research*.
- **Cattaneo, Ma, Masatlioglu & Suleymanov (2020)** — Random Attention Model: stochastic WARP violations from inattention. *Journal of Political Economy*.
- **Echenique, Imai & Saito (2020)** — Testable restrictions on time preferences. *AEJ: Micro*.

See :doc:`references` for the full bibliography.

----

Implements the framework of Chambers & Echenique (2016) `Revealed Preference Theory <https://www.amazon.com/Revealed-Preference-Econometric-Society-Monographs/dp/1107087805>`_ (Cambridge University Press, Econometric Society Monographs). See :doc:`theory_landscape` for the full method map.

.. toctree::
   :maxdepth: 2
   :hidden:

   quickstart
   budget/index
   menu/index
   benchmarks
   algorithms
   performance
   theory_landscape
   api
   references
