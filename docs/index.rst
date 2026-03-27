PyRevealed
==========

PyRevealed is a high-performance computational library designed for the axiomatic analysis of choice behavior. It provides a robust framework for quantifying the consistency of observed decisions with the hypothesis of utility maximization, supporting both budget-constrained and menu-based choice environments.

.. raw:: html

   <div class="feature-grid">
     <div class="feature-card">
       <h3>Axiomatic Scoring</h3>
       <p>Quantify consistency via GARP, SARP, and WARP. Compute efficiency indices including CCEI (0–1), Money Pump Index (MPI), and Houtman-Maks noise fractions.</p>
     </div>
     <div class="feature-card">
       <h3>Heterogeneous Data</h3>
       <p>Support for budget-constrained choices (price-quantity pairs) and discrete menu-based selections (item-set pairs) across large-scale longitudinal datasets.</p>
     </div>
     <div class="feature-card">
       <h3>Computational Engine</h3>
       <p>Optimized Rust backend utilizing graph-theoretic algorithms and HiGHS linear programming, parallelized via Rayon for high-throughput processing.</p>
     </div>
   </div>

.. code-block:: bash

   pip install pyrevealed

Why Revealed Preference?
------------------------

Unlike traditional econometric models that assume specific functional forms for utility, revealed preference analysis is **non-parametric**. It evaluates the internal consistency of data without imposing structural assumptions on preferences, making it an ideal tool for auditing the behavior of both human agents and algorithmic decision-making systems.

.. raw:: html

   <div style="margin: 1.5em 0;"></div>

.. raw:: html

   <div style="display: flex; gap: 20px; justify-content: space-between; flex-wrap: wrap;">
     <img src="_static/budget_hero.gif" style="width: 48%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" alt="Budget Choices">
     <img src="_static/menu_hero.gif" style="width: 48%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" alt="Menu Choices">
   </div>

Integrated Analytical Pipelines
--------------------------------

PyRevealed implements two primary workflows tailored to the structure of the observed data:

.. code-block:: text

   Budget-Constrained Choice (e.g., Grocery)   Discrete Menu-Based Choice (e.g., RecSys)
   ─────────────────────────────────────────   ─────────────────────────────────────────
        BehaviorLog(prices, quantities)             MenuChoiceLog(menus, choices)
                    │                                           │
                    ▼                                           ▼
       validate_consistency() [GARP]               validate_menu_sarp() [SARP]
                    │                                           │
                    ▼                                           ▼
      compute_integrity_score() [CCEI]            compute_menu_efficiency() [HM]
                    │                                           │
                    ▼                                           ▼
          Segmentation / Diagnostics                  Segmentation / Diagnostics
          ─ CCEI histogram by segment                 ─ HM score distribution
          ─ MPI: money-pump exploitability            ─ SARP violation counts
          ─ HARP: homothetic rationality              ─ Churn signal (sliding HM)
          ─ Utility recovery (Afriat LP)              ─ Ordinal preference ranking

Performance Benchmarks
----------------------

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

Empirical benchmarks: CCEI ≈ 0.88 in lab experiments (Choi et al., 2014); HM ≈ 0.70–0.85 in recommendation click data. Memory stays flat regardless of dataset size via bounded streaming chunks.

----

PyRevealed's methodology is grounded in the formal frameworks established in `Chambers & Echenique (2016) <https://www.amazon.com/Revealed-Preference-Econometric-Society-Monographs/dp/1107087805>`_. See :doc:`theory_landscape` for a complete map of all methods organized by data type and output type.

.. toctree::
   :maxdepth: 2
   :hidden:

   quickstart
   budget/index
   menu/index
   applications
   algorithms
   performance
   theory_landscape
   api
   references
