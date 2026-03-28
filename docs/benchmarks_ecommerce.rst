E-commerce Benchmarks
=====================

*Last updated: 2026-03-28. Instacart now uses aisle-level menu construction.
H&M results with per-customer realized prices (v0.5.8).*

Eight public datasets, 217K users, 42 RP features. CatBoost (H&M) or LightGBM
(others) with 80/20 user holdout + bootstrap CI. Results split by data type:

- **Menu datasets**: RP features are **competitive with baselines**. Taobao
  RP-only AUC (0.925) beats the engagement baseline (0.913). Graph features
  (``menu_transitivity``, ``menu_pref_density``) and choice entropy carry
  real signal that engagement stats miss.
- **Budget datasets**: RP adds ~0% marginal lift over strong RFM baselines.
  Spending features already capture the signal; CCEI/MPI are correlated.
- **Instacart**: Aisle-level menu construction with trailing-3 menus.
  RP features show real structure (83.8% SARP violations) but add near-zero
  lift over baseline — consistent with habit-heavy grocery reordering.

All targets use top-tercile thresholds for consistency.

Assumptions
-----------

**Dunnhumby.** 2,222 households, 104 weeks, 10 commodity groups (~$19/week of
a ~$100--150 weekly grocery basket). Budget-based RP. Global median price oracle
per commodity per week, shared across all households (Dean & Martin 2016).
Individual price exposure (coupons, regional variation) is not captured.
Within-commodity substitution is invisible.

**Open E-Commerce.** 4,694 users, category-level quantities. Budget-based RP.
Median price per category per month, forward-filled for missing periods. Shared
oracle across users. Within-category product switching is invisible.

**H&M.** 46,757 customers, 31.8M transactions (2018-09 to 2020-09). Budget-based
RP with per-customer realized prices.

*Observation unit.* Customer × month. Each customer-month is one budget
observation consisting of a price vector and a quantity vector over 20 product
groups.

*Goods.* Product groups are the first 2 digits of article_id, top 20 by
transaction frequency. Coarse but necessary — finer grouping destroys repeated
support across months.

*Quantities.* Article counts per group per month. Each raw CSV row is one
purchased article unit (confirmed by duplicate (date, customer, article) rows
representing distinct units). If a customer bought 5 items in group "06" in June,
quantity = 5.

*Prices.* Per-customer realized prices, not a shared oracle. For groups the
customer purchased in a given month: their own average paid price across all
articles in that group-month. For groups they did not purchase: imputed via
period-group median → group median → global median. This three-tier fallback
ensures every observation has a full price vector (required for RP tests) while
preserving individual price variation where it exists. Prices are normalized 0--1
(Kaggle competition); relative variation is real, absolute dollar values are not.

*Panel eligibility.* ≥ 6 active months, ≥ 10 total observations, ≥ 5 in the
feature window, ≥ 3 in the target window. 46,757 of ~50,000 most-active
customers pass these filters.

*Time toggle.* Default is month. Week and quarter are available as robustness
checks but are not used in the main benchmark (week is too sparse, quarter too
coarse).

*Ignored.* Sales channel (online vs in-store) is dropped for simplicity. Any
resulting price heterogeneity is treated as noise.

*Targets.* Three targets from the last 30% of each customer's history: (1) High
Spender — top tercile of target-window total spend (classification); (2) Future
Spend — mean spend per period in target window (regression); (3) Spend Change —
target mean spend minus train mean spend (regression). "Churn" was dropped
because the design excludes customers with no future window, making it
spend-decline-among-survivors rather than true disappearance. "LTV" was renamed
to "Future Spend" for the same reason.

*Features.* Baseline: 10 spending/concentration features (total spend, mean
spend, std, basket size, Herfindahl, top group share, spend slope, spend CV,
active groups, observation count). RP: GARP, CCEI, MPI, HARP, Houtman-Maks, VEI
distributions, utility recovery, graph structure (~42 features via Engine +
extended per-user algorithms). A dual-baseline comparison (10 core vs 17 full
features) confirmed the extra 7 baseline features add < 0.002 AUC.

*Model.* CatBoost with default parameters, 80/20 user holdout, bootstrap CI on
lift (1000 iterations), grouped permutation importance.

*Key finding.* RP features add negligible signal for classification (−0.1% AUC
lift on High Spender) but show small consistent gains for regression (+0.003 R²
on Future Spend, +0.005 R² on Spend Change). RP grouped importance (0.005) is
55× smaller than baseline importance (0.276). Spending history already captures
the predictive signal; CCEI and MPI are correlated with spending features and
do not carry independent information for this dataset.

**Instacart.** 50,000 users, 134 aisles. Menu-based RP (no prices in raw data).
Observation = user × order × aisle with exactly one reordered SKU. Menu =
trailing-3 order products in the same aisle (familiarity set). Filters: menu
size ≥ 2, (user, aisle) pairs with ≥ 3 valid events. Yields 4.5M events from
120K users across 715K user-aisle pairs. Habit-heavy: 58.6% of repeated
user-aisle pairs never switch; 83.8% of users have SARP violations.

**REES46.** 8,832 users, click-to-purchase sessions. Menu-based RP.
Server-defined session IDs (gold standard). Menus contain only items the user
clicked; unviewed items are invisible. Median menu size ~5 items. No prices —
choices reveal preference orderings only.

**Taobao.** 4,239 users, 100M raw events. Menu-based RP. Session boundaries
defined by 30-minute inactivity gaps (84% of inter-event gaps < 30 min). Median
menu size 4 items. Menus contain only items the user viewed or purchased within
a session. No prices.

**Tenrec.** 50,000 users, NeurIPS 2022 QQ Browser dataset. Menu-based RP.
Click-to-like windows with positional feedback tracking; median ~5 clicks
between likes. Menus reflect algorithmic recommendations, not organic browsing.
Items shown but not clicked are invisible. No prices.

Results
-------

.. list-table::
   :header-rows: 1
   :widths: 18 8 15 10 10 10 8

   * - Dataset
     - N
     - Target
     - Baseline
     - +RP
     - RP-only
     - Lift%
   * - Dunnhumby
     - 2,222
     - High Spender
     - 0.960
     - 0.960
     - —
     - -0.0%
   * - Dunnhumby
     - 2,222
     - Churn
     - 0.752
     - 0.740
     - —
     - -1.5%
   * - Open E-Commerce
     - 4,694
     - High Spender
     - 0.950
     - 0.951
     - —
     - +0.0%
   * - Open E-Commerce
     - 4,694
     - Churn
     - 0.846
     - 0.846
     - 0.769
     - -0.0%
   * - H&M
     - 46,757
     - High Spender
     - 0.784
     - 0.783
     - 0.720
     - -0.1%
   * - H&M
     - 46,757
     - Future Spend (R²)
     - 0.337
     - 0.340
     - —
     - +0.003
   * - H&M
     - 46,757
     - Spend Change (R²)
     - 0.290
     - 0.295
     - —
     - +0.005
   * - Instacart
     - 50,000
     - Low Loyalty
     - 0.968
     - 0.969
     - —
     - +0.0%
   * - Instacart
     - 50,000
     - High Novelty
     - 0.765
     - 0.767
     - 0.762
     - +0.3%
   * - REES46
     - 8,832
     - High Engagement
     - 0.996
     - 0.996
     - 0.990
     - +0.0%
   * - Taobao
     - 4,239
     - High Engagement
     - 0.913
     - **0.915**
     - **0.925**
     - **+0.2%**
   * - Tenrec
     - 50,000
     - High Engagement
     - 0.993
     - 0.993
     - **0.993**
     - +0.0%

*Baseline = LightGBM on 13 RFM features. +RP = same model with 42 RP features
added. RP-only = RP features without baseline. On Taobao, RP-only (0.925)
outperforms the engagement baseline (0.913) — graph transitivity and choice
entropy capture patterns that session counts miss.*

Top Features
------------

Across all classification tasks (LightGBM feature importance, combined model):

.. list-table::
   :header-rows: 1
   :widths: 5 30 10 55

   * - #
     - Feature
     - Type
     - Interpretation
   * - 1
     - total_spend
     - Baseline
     - Total expenditure in training period
   * - 2
     - spend_slope
     - Baseline
     - Spending trend (increasing/decreasing)
   * - 3
     - n_sessions
     - Baseline
     - Number of menu presentations (menu datasets)
   * - 4
     - n_obs
     - Baseline
     - Number of observations (frequency)
   * - 5
     - mean_basket_size
     - Baseline
     - Average items per observation
   * - 6
     - max_choice_freq
     - Baseline
     - Most-chosen item frequency
   * - 7
     - util_gini
     - **RP**
     - Gini inequality of recovered Afriat utility values
   * - 8
     - spend_cv
     - Baseline
     - Spending variability (coefficient of variation)
   * - 9
     - items_per_session
     - Baseline
     - Item diversity per session
   * - 10
     - choice_entropy_norm
     - **RP**
     - Normalized Shannon entropy of choice distribution

Menu-dataset top features (Taobao + Tenrec):

.. list-table::
   :header-rows: 1
   :widths: 5 30 10 55

   * - #
     - Feature
     - Type
     - Interpretation
   * - 1
     - std_menu_size
     - Baseline
     - Variability of menu sizes
   * - 2
     - menu_transitivity
     - **RP**
     - Item graph transitivity ratio
   * - 3
     - n_sessions
     - Baseline
     - Number of sessions
   * - 4
     - menu_pref_density
     - **RP**
     - Item graph edge density
   * - 5
     - choice_entropy_norm
     - **RP**
     - Normalized choice entropy
   * - 8
     - menu_util_range
     - **RP**
     - Ordinal utility spread (max - min recovered rank)

Four of the top 8 menu features are RP-derived. Item graph
structure and choice entropy carry signal that engagement statistics do not capture.

Reproduce
---------

.. code-block:: bash

   pip install prefgraph lightgbm scikit-learn
   python case_studies/benchmarks/runner.py --datasets all

Datasets require ``kaggle`` CLI. See ``case_studies/benchmarks/`` for details.

----

Appendix: Pipeline
------------------

.. code-block:: text

   Raw CSV
     -> Loader (prefgraph.datasets)
     -> BehaviorPanel / MenuChoiceLog per user
     -> Temporal split: first 70% -> features, last 30% -> targets
     -> Feature extraction:
          Baseline (13): RFM, category concentration, temporal trends
          RP Engine (14): CCEI, MPI, HM, VEI, GARP, HARP, SCC, n_scc, harp_severity
          RP Extended (28): VEI distribution, utility recovery (Gini, CV),
              graph network (density, transitivity, cycles), MPI cycle costs,
              choice reversals, choice entropy, congruence, ordinal utility
     -> LightGBM (num_leaves=15, lr=0.03, reg_alpha=1.0, reg_lambda=5.0)
     -> 5-fold stratified CV
     -> Metrics: AUC-ROC, AUC-PR, Log Loss, F1

**Three models per target**: (a) Baseline only, (b) RP only, (c) Baseline + RP.

**Targets**: High Spender (top tercile spend), Future Spend (regression),
Spend Change (regression), High Engagement (top tercile sessions).

**Output**: ``case_studies/benchmarks/output/results.json`` (full metrics),
``summary_table.csv``, ``figures/``.

