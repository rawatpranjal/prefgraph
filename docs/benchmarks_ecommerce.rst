Case Study 2: Predicting Customer Spend & Engagement
=====================================================

Across 11 datasets and 27 prediction targets, revealed preference features add between zero and two percent marginal lift over standard baselines. The signal is real but small. Baseline spend and engagement features already capture most of the predictive power.

Setup
-----

We test whether revealed preference graph features improve user-level predictions of spend, churn, engagement, and loyalty. The 11 datasets span grocery, e-commerce, fashion, video, news, and classifieds platforms. Each user's first 70 percent of observations produce 42 RP features alongside a 13-feature RFM baseline. The RP features include GARP, CCEI, MPI, HM, VEI, graph density, transitivity, and choice entropy. We train a regularized LightGBM and an L1-penalized logistic regression on an 80/20 user holdout. All results below are out-of-sample AUC-ROC from the regularized LightGBM. Every run uses SEED 42 and is fully reproducible.

.. _eco-results:

Results
-------

.. list-table::
   :header-rows: 1
   :widths: 14 7 7 16 8 8 7 7 7

   * - Dataset
     - Type
     - N
     - Target
     - Base
     - +RP
     - Δ
     - Lift %
     - Engine
   * - Dunnhumby
     - Budget
     - 2,222
     - Spend Drop
     - .683
     - **.694**
     - +.011
     - **+1.6%**
     - 6m 6s
   * - Dunnhumby
     -
     -
     - High Spender
     - .962
     - .961
     - -.001
     - -0.1%
     -
   * - Dunnhumby
     -
     -
     - Future LTV (R²)
     - .582
     - .587
     - +.005
     - +0.9%
     -
   * - Amazon
     - Budget
     - 4,668
     - Spend Drop
     - .756
     - **.761**
     - +.005
     - **+0.7%**
     - 12m
   * - Amazon
     -
     -
     - High Spender
     - .940
     - .939
     - -.001
     - -0.1%
     -
   * - H&M
     - Budget
     - 46,757
     - Spend Change (R²)
     - .299
     - .302
     - +.003
     - +1.0%
     - 16m
   * - H&M
     -
     -
     - High Spender
     - .787
     - .785
     - -.002
     - -0.3%
     -
   * - Instacart
     - Menu
     - 50,000
     - Low Loyalty
     - .969
     - .970
     - +.001
     - +0.1%
     - 8m
   * - REES46
     - Menu
     - 50,000
     - Low Loyalty
     - .883
     - .882
     - -.001
     - -0.1%
     - 8m
   * - Taobao
     - Menu
     - 50,000
     - Engagement
     - .938
     - .938
     - .000
     - 0.0%
     - 3m
   * - Taobao BW
     - Menu
     - 29,519
     - Low Loyalty
     - .984
     - .986
     - +.002
     - +0.2%
     - 0.1s
   * - Tenrec
     - Menu
     - 50,000
     - Engagement
     - .992
     - .992
     - .000
     - 0.0%
     - 15m
   * - MIND
     - Menu
     - 5,000
     - High CTR
     - .678
     - .667
     - -.011
     - -1.6%
     - 4m
   * - FINN
     - Menu
     - 1,869
     - Low Loyalty
     - .957
     - .957
     - .000
     - 0.0%
     - 13m

All values are out-of-sample AUC-ROC from regularized LightGBM. Base refers to 13 RFM features. +RP adds 42 revealed preference features on top of Base. One representative target per dataset is shown. Full results for all targets are in ``output/results.json``. Engine times are Rust batch scoring on Apple M-series hardware.

Findings
--------

The only targets with consistent positive lift across sample sizes and models are the two Spend Drop predictions. On Dunnhumby the lift is 1.6 percent and on Amazon it is 0.7 percent. On these targets the VEI metric captures declining purchase efficiency before spending actually drops. Everywhere else the lift is within plus or minus 0.3 percent. The mean lift across all 27 targets is 0.03 percent.

Despite near-zero aggregate lift, RP features rank highly in model importance. The table below shows that ``menu_transitivity`` is the third most important feature overall, appearing in the top 10 for 18 of 19 menu targets. The model uses RP features but they do not improve predictions because the baseline features already capture overlapping information through a different path.

.. _eco-features:

Feature Importance
------------------

.. list-table::
   :header-rows: 1
   :widths: 5 30 8 15 15

   * - Rank
     - Feature
     - Group
     - Mean importance
     - Top-10 in
   * - 1
     - ``n_sessions``
     - Base
     - 0.144
     - 15/19 targets
   * - 2
     - ``std_menu_size``
     - Base
     - 0.112
     - 15/18
   * - 3
     - ``menu_transitivity``
     - **RP**
     - 0.101
     - 18/19
   * - 4
     - ``mean_basket_size``
     - Base
     - 0.092
     - 12/13
   * - 5
     - ``spend_slope``
     - Base
     - 0.080
     - 13/13
   * - 6
     - ``mean_menu_size``
     - Base
     - 0.072
     - 16/18
   * - 7
     - ``choice_entropy_norm``
     - **RP**
     - 0.072
     - 17/19
   * - 8
     - ``menu_pref_density``
     - **RP**
     - 0.072
     - 15/18
   * - 9
     - ``total_spend``
     - Base
     - 0.070
     - 11/13
   * - 10
     - ``mean_spend``
     - Base
     - 0.064
     - 11/13

Three of the top ten features are RP-derived. All three are graph-structural measures of how consistently a user makes decisions. They are not measures of volume or frequency, which is what the baseline already captures.

.. _eco-replication:

Replication
-----------

.. code-block:: bash

   pip install prefgraph lightgbm scikit-learn

   # Full run on all 11 validated datasets with both models
   python case_studies/benchmarks/runner.py --datasets validated --model both

   # Quick smoke test at 250 users
   python case_studies/benchmarks/runner.py --datasets validated --max-users 250 --model both

   # Single dataset
   python case_studies/benchmarks/runner.py --datasets dunnhumby

   # Regenerate summary and plots from cached results
   python case_studies/benchmarks/runner.py --replot

All results are deterministic with SEED 42. Per-dataset JSON files are saved to ``case_studies/benchmarks/output/``. For datasets on external drives, set the environment variable ``PYREVEALED_DATA_DIR`` to the data path.

.. _eco-appendix:

Appendix: Dataset Summary
--------------------------

The table below shows standardized summary statistics for all 11 datasets. T is the median number of choice occasions per user. K is the number of alternatives. Repeat is the fraction of observations where the chosen item was chosen previously. Uniq is the median fraction of distinct items per user.

.. list-table::
   :header-rows: 1
   :widths: 16 7 7 7 5 5 5 6 6 6

   * - Dataset
     - Type
     - N
     - Total obs
     - T med
     - K
     - %T≥10
     - Repeat
     - Uniq%
     - Domain
   * - Dunnhumby
     - Budget
     - 2,222
     - 55,388
     - 30
     - 10
     - 93%
     - 28%
     - 24%
     - Grocery
   * - Amazon
     - Budget
     - 4,668
     - 54,356
     - 31
     - 15
     - 95%
     - 73%
     - 13%
     - E-commerce
   * - H&M
     - Budget
     - 46,757
     - 29,132
     - 15
     - 9
     - 97%
     - 68%
     - 21%
     - Fashion
   * - Instacart
     - Menu
     - 50,000
     - 59,649
     - 16
     - 18
     - 64%
     - 57%
     - 50%
     - Grocery
   * - REES46
     - Menu
     - 50,000
     - 14,922
     - 7
     - 24
     - 15%
     - 19%
     - 86%
     - E-commerce
   * - Taobao
     - Menu
     - 50,000
     - 11,061
     - 5
     - 30
     - 2%
     - 2%
     - 100%
     - E-commerce
   * - Taobao BW
     - Menu
     - 29,519
     - 2,593
     - 4
     - 25
     - 3%
     - 6%
     - 100%
     - E-commerce
   * - RetailRocket
     - Menu
     - 47
     - 356
     - 5
     - 57
     - 19%
     - 0%
     - 100%
     - E-commerce
   * - Tenrec
     - Menu
     - 50,000
     - 12,847
     - 5
     - 24
     - 16%
     - 0%
     - 100%
     - Video
   * - MIND
     - Menu
     - 5,000
     - 855
     - 4
     - 86
     - 2%
     - 1%
     - 100%
     - News
   * - FINN
     - Menu
     - 1,869
     - 15,850
     - 9
     - 57
     - 39%
     - 6%
     - 100%
     - Classifieds

Budget datasets have rich purchase histories with 15 to 31 median observations per user and repeat rates between 28 and 73 percent. This structure is ideal for revealed preference testing because users revisit the same goods under varying prices. Menu datasets are thinner, with 4 to 9 median observations and repeat rates below 6 percent on most platforms. Recommendation systems surface novel items, so users rarely face the same choice set twice. The exceptions are Instacart with 57 percent repeat rate and REES46 with 19 percent. These are the menu datasets where RP features show the most signal. Revealed preference testing needs repeated choices from overlapping sets, and most menu platforms do not provide that.

Appendix: Feature Correlation
-------------------------------

RP features are largely orthogonal to baseline features. The median absolute cross-correlation is 0.12 on budget datasets and 0.28 on menu datasets. The utility-recovery features are internally redundant with pairwise correlations above 0.95. The genuinely independent RP features are ``choice_entropy``, ``menu_transitivity``, ``sarp_violation_density``, ``hm_ratio``, and ``vei_mean``. These are the same features that survive L1 selection in the Lasso model and appear in the LGBM top-10 importance list.

Appendix: Null Rates
---------------------

50 of 59 RP features are always populated on budget data. 25 of 27 are always populated on menu data. The exceptions are utility-recovery features that require sufficient intersecting choices to solve the Afriat LP. All null values are imputed with train-set medians before model training.
