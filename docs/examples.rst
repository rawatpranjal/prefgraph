Examples
========

.. contents:: On this page
   :local:
   :depth: 1

Budget Choices (Engine)
-----------------------

Score many users in one call. The Engine uses Rust for graph algorithms
and LP solving, parallelized across CPU cores.

.. code-block:: python

   from pyrevealed.engine import Engine
   import numpy as np

   # Simulate 1000 users: 20 purchase occasions, 5 product categories
   rng = np.random.RandomState(42)
   users = [
       (rng.rand(20, 5).astype(np.float64) + 0.1,
        rng.rand(20, 5).astype(np.float64) + 0.1)
       for _ in range(1000)
   ]

   engine = Engine(metrics=["garp", "ccei", "mpi", "harp", "hm"])
   results = engine.analyze_arrays(users)

   # Summary statistics
   n_consistent = sum(1 for r in results if r.is_garp)
   avg_ccei = np.mean([r.ccei for r in results])
   avg_mpi = np.mean([r.mpi for r in results])

   print(f"Users: {len(results)}")
   print(f"Consistent: {n_consistent} ({100*n_consistent/len(results):.1f}%)")
   print(f"Avg CCEI: {avg_ccei:.3f}")
   print(f"Avg MPI: {avg_mpi:.3f}")
   print(f"Backend: {engine.backend}")

.. code-block:: text

   Users: 1000
   Consistent: 0 (0.0%)
   Avg CCEI: 0.823
   Avg MPI: 0.239
   Backend: rust

Each ``EngineResult`` has: ``is_garp``, ``ccei``, ``mpi``, ``is_harp``,
``hm_consistent``, ``hm_total``, ``utility_success``, ``vei_mean``, ``vei_min``.

Available metrics: ``"garp"``, ``"ccei"``, ``"mpi"``, ``"harp"``, ``"hm"``,
``"utility"``, ``"vei"``.

Menu Choices (SARP)
-------------------

When there are no prices --- just menus and chosen items --- use
``MenuChoiceLog`` with SARP and Houtman-Maks:

.. code-block:: python

   from pyrevealed import MenuChoiceLog, validate_menu_sarp, compute_menu_efficiency
   import numpy as np

   rng = np.random.RandomState(42)
   n_items = 8

   for uid in range(5):
       menus, choices = [], []
       for _ in range(25):
           size = rng.randint(2, min(5, n_items + 1))
           menu = frozenset(rng.choice(n_items, size, replace=False).tolist())
           menus.append(menu)
           choices.append(rng.choice(list(menu)))

       log = MenuChoiceLog(menus=menus, choices=choices)
       sarp = validate_menu_sarp(log)
       hm = compute_menu_efficiency(log)
       print(f"user {uid}  violations={sarp.num_violations}"
             f"  removed={len(hm.removed_observations)}/25")

.. code-block:: text

   user 0  violations=28  removed=12/25
   user 1  violations=21  removed=14/25
   user 2  violations=28  removed=9/25
   user 3  violations=28  removed=13/25
   user 4  violations=28  removed=13/25

``validate_menu_sarp`` checks for preference cycles across item choices.
``compute_menu_efficiency`` (Houtman-Maks) finds the minimum observations
to remove to make choices consistent.

Production Data
---------------

Test whether a firm's input/output decisions are consistent with profit
maximization:

.. code-block:: python

   from pyrevealed import ProductionLog
   from pyrevealed.algorithms.production import test_profit_maximization
   import numpy as np

   rng = np.random.RandomState(42)
   log = ProductionLog(
       input_prices=rng.rand(15, 3) + 0.5,
       input_quantities=rng.rand(15, 3) + 0.1,
       output_prices=rng.rand(15, 2) + 1.0,
       output_quantities=rng.rand(15, 2) + 0.1,
   )

   result = test_profit_maximization(log)
   print(f"Consistent: {result.is_consistent}")
   print(f"Violations: {result.num_violations}")

.. code-block:: text

   Consistent: False
   Violations: 2

Single-User Analysis
--------------------

For drilling into one user without the Engine:

.. code-block:: python

   from pyrevealed import BehaviorLog, check_garp, compute_aei, compute_mpi
   import numpy as np

   prices = np.array([[2.0, 1.0], [1.0, 2.0], [1.5, 1.5]])
   quantities = np.array([[3.0, 2.0], [2.0, 3.0], [2.5, 2.5]])

   log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

   garp = check_garp(log)
   aei = compute_aei(log)
   mpi = compute_mpi(log)

   print(f"GARP consistent: {garp.is_consistent}")
   print(f"Violations: {len(garp.violations)}")
   print(f"CCEI: {aei.efficiency_index:.3f}")
   print(f"MPI: {mpi.mpi_value:.3f}")

.. code-block:: text

   GARP consistent: False
   Violations: 2
   CCEI: 0.889
   MPI: 0.100

Loading Datasets
----------------

Built-in panel datasets (require separate download):

.. code-block:: python

   from pyrevealed.datasets import load_dunnhumby, load_open_ecommerce, load_uci_retail

   panel = load_dunnhumby()           # 2,500 households, 10 goods, 104 weeks
   panel = load_open_ecommerce()      # 4,700 consumers, 50 categories, 66 months
   panel = load_uci_retail()          # 1,800 customers, 50 products, 13 months

From a pandas DataFrame:

.. code-block:: python

   from pyrevealed import BehaviorPanel

   panel = BehaviorPanel.from_dataframe(
       df,
       user_col="customer_id",
       cost_cols=["price_a", "price_b", "price_c"],
       action_cols=["qty_a", "qty_b", "qty_c"],
   )

From individual logs:

.. code-block:: python

   from pyrevealed import BehaviorLog, BehaviorPanel
   import numpy as np

   logs = [
       BehaviorLog(prices, quantities, user_id=f"user_{i}")
       for i, (prices, quantities) in enumerate(user_data)
   ]
   panel = BehaviorPanel.from_logs(logs)
