Efficiency and Power Indices
===========================

In empirical settings where axiomatic consistency (e.g., GARP) is violated, efficiency indices provide a continuous measure to quantify the severity of departures from rationality and the statistical power of the associated tests.

Critical Cost Efficiency Index (CCEI)
-------------------------------------

**Reference Implementation:** ``compute_integrity_score(log)``

The Critical Cost Efficiency Index (CCEI), often referred to as the Afriat Efficiency Index (AEI), quantifies the minimal adjustment required to the agent's budget sets to eliminate all axiomatic violations.

**Formal Definition:**

.. math::

   \text{CCEI} = \sup \left\{ e \in [0,1] : \text{GARP is satisfied with } e \cdot p^i \cdot x^i \geq p^i \cdot x^j \right\}

**Computational Methodology:**

The CCEI is determined via binary search over the efficiency parameter :math:`e`. For a given :math:`e`, the modified revealed preference relation :math:`R_e` is defined as:

.. math::

   x^i \, R_e \, x^j \iff e \cdot (p^i \cdot x^i) \geq p^i \cdot x^j

**Interpretation and Benchmarks:**

The CCEI represents the fraction of wealth the agent "wastes" through inconsistent choices.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - CCEI Value
     - Behavioral Interpretation
   * - 1.00
     - Perfect consistency; behavior is fully reconcilable with utility maximization.
   * - [0.95, 1.00)
     - Near-rational behavior; deviations are often attributed to minor optimization errors or measurement noise.
   * - [0.85, 0.95)
     - Moderate inconsistency; presence of significant preference cycles.
   * - < 0.85
     - Substantial departures from the hypothesis of utility maximization.

.. note::

   **Empirical Context:** Choi et al. (2014) report a mean CCEI of approximately 0.88 in large-scale laboratory experiments. In high-frequency e-commerce data, CCEI values may be lower due to unobserved factors and higher levels of stochastic noise.

**References:** Afriat (1972), Varian (1990, 1991).


Money Pump Index (MPI)
----------------------

**Reference Implementation:** ``compute_confusion_metric(log)``

The Money Pump Index (MPI) provides an alternative measure of inconsistency by quantifying the maximal economic loss an agent could incur by being subjected to a sequence of trades corresponding to their revealed preference cycles.

**Formal Definition:**

For a detected violation cycle :math:`k_1 \to k_2 \to \cdots \to k_m \to k_1`:

.. math::

   \text{MPI} = \frac{\sum_{i=1}^{m} \max(0, p^{k_i} \cdot x^{k_i} - p^{k_i} \cdot x^{k_{i+1}})}{\sum_{i=1}^{m} p^{k_i} \cdot x^{k_i}}

**Interpretation:**

The MPI measures the "exploitability" of the agent. A higher MPI indicates that the agent's inconsistencies could lead to substantial wealth extraction in a market environment.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - MPI Value
     - Interpretation
   * - 0.00
     - No preference cycles detected; zero exploitability.
   * - (0.00, 0.10]
     - Marginal exploitability; minor transitivity violations.
   * - (0.10, 0.30]
     - Significant exploitability; noticeable behavioral contradictions.
   * - > 0.30
     - Severe exploitability; fundamental failure of transitive preferences.

**Reference:** Echenique, Lee, & Shum (2011).


Houtman-Maks Index (HM)
-----------------------

**Reference Implementation:** ``compute_minimal_outlier_fraction(log)``

The Houtman-Maks Index identifies the maximal subset of observations that are mutually consistent with GARP. It is typically expressed as the fraction of observations that must be discarded to achieve rationalizability.

**Formal Definition:**

.. math::

   \text{HM} = \min \left\{ \frac{|S|}{T} : \text{The subset of observations } \{1, \ldots, T\} \setminus S \text{ satisfies GARP} \right\}

This index is particularly useful for identifying agents who are "mostly" rational but exhibit a small number of anomalous choices.

**Reference:** Houtman & Maks (1985).


Granular Efficiency (Varian's Index)
------------------------------------

**Reference Implementation:** ``compute_granular_integrity(log)``

Varian's Index provides observation-specific efficiency scores by solving a constrained optimization problem. It identifies the minimal perturbation to each individual budget set required to satisfy GARP.

**Optimization Problem:**

.. math::

   \min \sum_{i=1}^{T} (1 - e_i)

Subject to:

.. math::

   e_i \cdot (p^i \cdot x^i) \geq p^i \cdot x^j \quad \forall \, (i,j) : x^i \, R^* \, x^j

.. math::

   0 \leq e_i \leq 1

This granular approach allows analysts to pinpoint specific temporal periods or choice environments where the agent's behavior deviates from the model.

**Reference:** Varian (1990).


Statistical Test Power (Bronars' Index)
---------------------------------------

**Reference Implementation:** ``compute_test_power(log)``

The validity of any revealed preference test depends on its ability to reject the null hypothesis of random behavior. Bronars' Index quantifies the statistical power of the test given the observed budget sets.

**Methodology:**

Power is estimated by simulating a cohort of "synthetic agents" who make random choices (typically drawn from a uniform distribution on the budget hyperplane) and calculating the frequency with which these random choices violate GARP.

.. math::

   \text{Power} = \mathbb{P}(\text{Random Behavior Violates GARP} \mid \text{Observed Budget Sets})

**Interpretation:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Power Value
     - Statistical Interpretation
   * - > 0.90
     - High Power: The observed budget sets provide a rigorous test; passing GARP is highly informative.
   * - [0.70, 0.90]
     - Sufficient Power: The test is capable of discriminating between rational and random behavior.
   * - < 0.50
     - Low Power: The budget sets do not overlap sufficiently; passing GARP may be a trivial result of the data structure rather than a reflection of agent rationality.

.. warning::

   A high CCEI (or AEI) observed in a low-power environment must be interpreted with caution, as it may be a spurious artifact of non-overlapping budget sets.

**Reference:** Bronars (1987).
