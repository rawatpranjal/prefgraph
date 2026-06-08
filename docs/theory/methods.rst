Method Landscape
================

Every method in PrefGraph, organized by **input data type** and **output type**.
This table is the roadmap for choosing the right theory page before choosing an
API call.

.. list-table::
   :header-rows: 1
   :widths: 15 22 20 22 21
   :class: method-landscape

   * - **Data Type**
     - **Test** *(pass/fail)*
     - **Score** *(0 to 1)*
     - **Recover** *(vector/function)*
     - **Structure** *(property)*

   * - | **Budget Choice**
       | ``BehaviorLog``
       | *(prices x quantities)*
       |
       | :doc:`Theory <../budget/theory_consistency>`
       | :doc:`Tutorial <../budget/tutorial>`
     - | GARP - Varian (1982)
       | WARP - Samuelson (1938)
       | SARP - Richter (1966)
       | HARP - Varian (1983)
       | GAPP - Deb et al. (2023)
       | Acyclical P - Varian (1982)
       | Quasilinearity - C&E Ch 9
       | Integrability - Slutsky
       | Additive separability - C&E Ch 9
       | Gross substitutes - C&E Ch 10
       | Separability - C&E (2016)
       | Law of demand - C&E Ch 10
     - | CCEI/AEI - Afriat (1967)
       | VEI - Varian (1990)
       | MPI - Echenique et al. (2011)
       | Houtman-Maks - H&M (1985)
       | Swaps - A&B (2015)
       | Min cost - Dean & Martin (2016)
       | Bronars power - Bronars (1987)
     - | Utility - Afriat (1967)
       | Value function - Afriat (1967)
       | Demand prediction
       | CV - Vartia (1983)
       | EV - Vartia (1983)
       | Expenditure function
       | Cost function
       | Slutsky matrix - C&E Ch 6
       | Hicksian demand - C&E Ch 10
       | Substitution matrix
     - | Separability partitions
       | Quasilinear structure
       | Additive structure
       | Spatial/ideal point - C&E Ch 11

   * - | **Menu Choice**
       | *(menus x choices)*
       |
       | *3 subtypes by data:*
       | 1. Deterministic
       |    ``MenuChoiceLog``
       | 2. Stochastic
       |    ``StochasticChoiceLog``
       | 3. Risk/Lotteries
       |    ``RiskChoiceLog``
       |
       | :doc:`Theory <../menu/theory_abstract>`
       | :doc:`Tutorial <../menu/tutorial_menu_choice>`
     - | *Deterministic*
       | WARP - Samuelson (1938)
       | SARP - Richter (1966)
       | Congruence - Richter (1966)
       | WARP-LA - MNO (2012)
       | Attention filter - MNO (2012)
       | Attention overload
       | Status quo bias
       |
       | *Stochastic*
       | RAM - Cattaneo et al. (2020)
       | RUM - Block & Marschak (1960)
       | McFadden axioms
       | IIA - Luce (1959)
       | Regularity - Debreu (1960)
       | Stochastic transitivity
       | Context effects
       |
       | *Risk*
       | Expected utility - vNM (1944)
       | Rank-dependent utility
     - | Menu HM - H&M (1985)
       | Distance to RUM
       | Predictive success
     - | *Deterministic*
       | Ordinal utility
       | Consideration sets - MNO (2012)
       | Preference with attention
       |
       | *Stochastic*
       | Attention probabilities
       | Salience weights
       | Choice probabilities
       | Luce model - Luce (1959)
       | RUM distribution
       | Bradley-Terry
       |
       | *Risk*
       | Risk profile
     - | Attention bounds
       | RAM parameters

   * - | **Production**
       | ``ProductionLog``
       | *(inputs x outputs)*
     - | Profit maximization - Varian (1984)
       | Cost minimization
     - | Technical efficiency
     - | -
     - | Returns to scale

   * - | **Intertemporal**
       | *(dated amounts)*
     - | Exponential discounting - EIS (2020)
       | Quasi-hyperbolic - Laibson (1997)
       | Present bias
     - | -
     - | Discount factor
     - | -

Abbreviations
-------------

- **C&E**: Chambers & Echenique (2016), *Revealed Preference Theory*
- **MNO**: Masatlioglu, Nakajima & Ozbay (2012, *AER*)
- **EIS**: Echenique, Imai & Saito (2020, *AEJ: Micro*)
- **A&B**: Apesteguia & Ballester (2015, *JPE*)
- **H&M**: Houtman & Maks (1985)
- **vNM**: von Neumann & Morgenstern (1944)

See :doc:`../papers` for papers cited in the implementation.
