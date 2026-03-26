PyRevealed
==========

Consistency and efficiency scores from choice data.

.. raw:: html

   <div class="feature-grid">
     <div class="feature-card">
       <h3>Scores</h3>
       <p>GARP consistency, CCEI efficiency (0-1), MPI exploitability, Houtman-Maks noise fraction. Per user.</p>
     </div>
     <div class="feature-card">
       <h3>Data</h3>
       <p>Budget choices (prices + quantities) and menu choices (items + selections). No prices needed for menus.</p>
     </div>
     <div class="feature-card">
       <h3>Engine</h3>
       <p>Rust backend. Graph algorithms + HiGHS LP, parallelized via Rayon. ~10K users/sec.</p>
     </div>
   </div>

.. code-block:: bash

   pip install pyrevealed

Budget Choice Scoring
---------------------

Score how consistently each user's purchases align with utility maximization.
Each user has T observations of prices and quantities across K goods.

.. code-block:: python

   from pyrevealed.engine import Engine
   import numpy as np

   # Simulate 10 users: 20 purchase occasions, 5 product categories
   rng = np.random.RandomState(42)
   users = [
       (rng.rand(20, 5).astype(np.float64) + 0.1,
        rng.rand(20, 5).astype(np.float64) + 0.1)
       for _ in range(10)
   ]

   engine = Engine(metrics=["garp", "ccei", "mpi", "harp", "hm"])
   results = engine.analyze_arrays(users)

   for i, r in enumerate(results):
       hm = f"{r.hm_consistent}/{r.hm_total}"
       print(f"user {i}  garp={str(r.is_garp):5s}  ccei={r.ccei:.3f}"
             f"  mpi={r.mpi:.3f}  harp={str(r.is_harp):5s}  hm={hm}")

.. code-block:: text

   user 0  garp=False  ccei=0.755  mpi=0.285  harp=False  hm=13/20
   user 1  garp=False  ccei=0.799  mpi=0.241  harp=False  hm=15/20
   user 2  garp=False  ccei=0.910  mpi=0.257  harp=False  hm=16/20
   user 3  garp=False  ccei=0.785  mpi=0.318  harp=False  hm=14/20
   user 4  garp=False  ccei=0.716  mpi=0.356  harp=False  hm=13/20
   user 5  garp=False  ccei=0.791  mpi=0.236  harp=False  hm=16/20
   user 6  garp=False  ccei=0.770  mpi=0.244  harp=False  hm=14/20
   user 7  garp=False  ccei=0.795  mpi=0.244  harp=False  hm=9/20
   user 8  garp=False  ccei=0.902  mpi=0.187  harp=False  hm=14/20
   user 9  garp=False  ccei=0.699  mpi=0.312  harp=False  hm=12/20

Each row is one user. ``garp`` = are their choices rationalizable?
``ccei`` = how close to perfectly rational (1.0 = perfect).
``mpi`` = how exploitable. ``hm`` = how many choices are consistent.
See :doc:`theory` for the economics.

Menu Choice Scoring
-------------------

When there are no prices --- just menus and chosen items (surveys, app clicks,
recommendations) --- use ``MenuChoiceLog`` with SARP:

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
       print(f"user {uid}  sarp={sarp.num_violations} violations"
             f"  removed={len(hm.removed_observations)}/25")

.. code-block:: text

   user 0  sarp=28 violations  removed=12/25
   user 1  sarp=21 violations  removed=14/25
   user 2  sarp=28 violations  removed=9/25
   user 3  sarp=28 violations  removed=13/25
   user 4  sarp=28 violations  removed=13/25

SARP checks for preference cycles across item choices (no prices needed).
Houtman-Maks tells you how many choices to discard to make the rest consistent.

Scale
-----

The Rust engine scores users in parallel via Rayon. Memory stays bounded
via streaming chunks.

.. code-block:: text

     1,000 users |  0.1s |  14,968 users/s
    10,000 users |  0.9s |  10,602 users/s
   100,000 users |  9.2s |  10,853 users/s

Single-User Analysis
--------------------

For drilling into one user:

.. code-block:: python

   from pyrevealed import BehaviorLog, check_garp, compute_aei, compute_mpi

   log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

   garp = check_garp(log)          # GARP consistency
   aei = compute_aei(log)          # Afriat efficiency index
   mpi = compute_mpi(log)          # Money pump index

   print(f"Consistent: {garp.is_consistent}")
   print(f"Efficiency: {aei.efficiency_index:.3f}")
   print(f"Exploitability: {mpi.mpi_value:.3f}")

----

Based on `Chambers & Echenique (2016) <https://www.amazon.com/Revealed-Preference-Econometric-Society-Monographs/dp/1107087805>`_.
See :doc:`theory` for the economics behind each score.

.. toctree::
   :maxdepth: 1
   :hidden:

   examples
   theory
   performance
   api
   references
