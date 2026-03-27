E-commerce Benchmarks
=====================

We test whether revealed preference features (CCEI, MPI, Houtman-Maks, VEI)
improve ML models on standard e-commerce prediction tasks. Six public
datasets, 120K+ users, LightGBM with 5-fold stratified CV. **Result: RP
features add 0--1% AUC over strong RFM baselines.** The signal is real but
modest — revealed preference scores capture behavioral patterns correlated
with, but not independent of, standard spending features.

Results
-------

.. list-table::
   :header-rows: 1
   :widths: 18 7 15 12 12 8 10

   * - Dataset
     - N
     - Target
     - Baseline
     - +RP
     - Lift%
     - AUC-PR
   * - Dunnhumby
     - 2,222
     - High Spender
     - 0.958
     - 0.959
     - +0.1%
     - 0.929
   * - Dunnhumby
     - 2,222
     - Churn
     - 0.748
     - 0.730
     - -2.4%
     - 0.254
   * - Open E-Commerce
     - 4,694
     - High Spender
     - 0.950
     - 0.951
     - +0.2%
     - 0.907
   * - Open E-Commerce
     - 4,694
     - Churn
     - 0.841
     - 0.847
     - +0.7%
     - 0.362
   * - Instacart
     - 50,000
     - Spend Drop
     - 0.665
     - 0.666
     - +0.2%
     - 0.210
   * - Instacart
     - 50,000
     - High Value
     - 0.967
     - 0.967
     - +0.0%
     - 0.932
   * - REES46
     - 8,832
     - High Engagement
     - 0.942
     - 0.941
     - -0.1%
     - 0.855
   * - H&M
     - 50,000
     - Churn
     - *pending*
     - *pending*
     - *pending*
     - *pending*

*"Baseline" = LightGBM on RFM + spending features. "+RP" = same model with RP features added. Lift = (Combined − Baseline) / Baseline × 100.*

Price Assumptions
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Dataset
     - Price Source
     - RP Type
     - Caveat
   * - Dunnhumby
     - Median price oracle
     - Budget
     - Shared prices across households (Dean & Martin 2016 standard)
   * - Open E-Commerce
     - Median per category/month
     - Budget
     - Forward-filled for missing periods
   * - Online Retail II
     - Actual transaction prices
     - Budget
     - Cleanest budget-based RP signal
   * - Instacart
     - Uniform ($1/unit)
     - Budget
     - **Quantity-only consistency** — CCEI/MPI do not test price-quantity tradeoffs
   * - REES46
     - N/A
     - Menu
     - Click → purchase sessions; no price assumptions
   * - H&M
     - Actual transaction prices
     - Budget
     - Normalized prices from competition data

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
     - spend_cv
     - Baseline
     - Spending variability (coefficient of variation)
   * - 4
     - herfindahl
     - Baseline
     - Category concentration (higher = fewer categories)
   * - 5
     - scc_ratio
     - **RP**
     - Fraction of observations in largest violation cycle
   * - 6
     - hm_ratio
     - **RP**
     - Houtman-Maks consistency fraction
   * - 7
     - mpi
     - **RP**
     - Money Pump Index (exploitability)
   * - 8
     - ccei
     - **RP**
     - Critical Cost Efficiency Index

Reproduce
---------

.. code-block:: bash

   pip install pyrevealed lightgbm scikit-learn
   python case_studies/benchmarks/runner.py --datasets all

Datasets download automatically via ``kaggle`` CLI. See ``case_studies/benchmarks/README.md``.
