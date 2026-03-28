Menus
=====

Menu-based analysis evaluates consistency when choices come from finite
sets of alternatives without prices. Covers classical WARP/SARP,
limited attention, random utility, and risk preferences.

**3 data subtypes:**

.. list-table::
   :widths: 33 34 33
   :align: center
   :class: gif-grid

   * - .. image:: ../_static/deterministic.gif
          :alt: Deterministic logic
          :width: 100%
     - .. image:: ../_static/stochastic.gif
          :alt: Stochastic logic
          :width: 100%
     - .. image:: ../_static/risk.gif
          :alt: Risk logic
          :width: 100%

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Subtype
     - Input Class
     - Description
   * - **Deterministic**
     - ``MenuChoiceLog``
     - menus → single choices (e.g., which product was clicked)
   * - **Stochastic**
     - ``StochasticChoiceLog``
     - menus → choice frequencies (e.g., 60% A, 30% B, 10% C)
   * - **Risk / Lotteries**
     - ``RiskChoiceLog``
     - lotteries → choices (e.g., gamble A vs gamble B)

.. admonition:: What can you do?

   - **Test**: WARP, SARP, Congruence, WARP-LA, RAM, RUM, IIA, regularity, expected utility
   - **Score**: Menu HM, distance to RUM, predictive success
   - **Recover**: Ordinal utility, consideration sets, attention probabilities, Luce/RUM, risk profile
   - **Structure**: Attention bounds, RAM parameters

.. rubric:: One question, three data formats

All three subtypes test the same core question — does a consistent preference ordering explain the observed choices? The **Deterministic** subtype feeds directly into ``Engine.analyze_menus()`` for batch Rust processing: pass a list of ``(menus, choices, n_items)`` tuples and get back WARP/SARP/HM scores for every user in one call. **Stochastic** and **Risk** data use the per-user Functions API instead (``validate_menu_sarp``, ``fit_luce_model``, ``compute_risk_profile``, etc.) because their inputs — choice frequency tables and lottery matrices — do not map to the simple tuple format the batch engine expects. The axioms and scores are identical across paths; only the entry point differs.

.. code-block:: python

   from prefgraph import MenuChoiceLog, validate_menu_sarp, compute_menu_efficiency

   log = MenuChoiceLog(menus=menus, choices=choices)
   sarp = validate_menu_sarp(log)            # Test: bool
   hm   = compute_menu_efficiency(log)       # Score: 0→1

Theory
------

.. toctree::
   :maxdepth: 1

   theory_abstract
   theory_attention
   theory_stochastic

Examples
--------

.. toctree::
   :maxdepth: 1

   examples
