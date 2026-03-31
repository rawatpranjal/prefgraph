Examples
========

Real-world applications of PrefGraph.

`Detecting Inconsistency in AI Agents <budget/app_llm_benchmark.html>`_
   Do LLMs keep a stable action ranking across menus? We build preference graphs from model choices and test for cycles (SARP, IIA), then quantify minimal edits (HM) to restore consistency.

`Descriptive Study of Classifieds Choice <menu/app_finn_slates.html>`_
   How consistent are 70,000 users on Norway's largest classifieds platform? We map 1.3M unique listings to 290 category-geography groups, run SARP, HM, and RUM tests, and compare search versus recommendation behavior.

`Predicting Customer Spend & Engagement <benchmarks_ecommerce.html>`_
   Do RP features improve predictive models? We benchmark GARP, CCEI, MPI, HM, and VEI features against spend/engagement baselines on churn, high-spender, engagement, and LTV tasks across 10+ datasets.

`Performance Benchmarks <performance.html>`_
   Throughput and scaling metrics for the Rust engine across dataset sizes, user counts, and choice dimensions.

.. toctree::
   :hidden:
   :maxdepth: 1

   budget/app_llm_benchmark
   menu/app_finn_slates
   benchmarks_ecommerce
   performance
