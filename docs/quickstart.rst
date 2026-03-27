Quickstart
==========

Install
-------

.. code-block:: bash

   pip install pyrevealed

Score 100 Users in 5 Lines
--------------------------

.. code-block:: python

   from pyrevealed.datasets import load_demo
   from pyrevealed.engine import Engine

   users = load_demo()  # 100 synthetic consumers, no download
   engine = Engine(metrics=["garp", "ccei", "mpi", "hm"])
   results = engine.analyze_arrays(users)

   for r in results[:5]:
       print(r)

.. code-block:: text

   EngineResult: [+] GARP-consistent  ccei=1.0000  hm=15/15  (25us)
   EngineResult: [+] GARP-consistent  ccei=1.0000  hm=15/15  (19us)
   EngineResult: [-] 2 violations  ccei=0.9920  mpi=0.0084  hm=14/15  (32us)
   EngineResult: [+] GARP-consistent  ccei=1.0000  hm=15/15  (20us)
   EngineResult: [+] GARP-consistent  ccei=1.0000  hm=15/15  (23us)

Read the Results
----------------

Each ``EngineResult`` is one user's rationality profile:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Range
     - Meaning
   * - ``is_garp``
     - bool
     - Are choices rationalizable? True = consistent with utility maximization.
   * - ``ccei``
     - (0, 1]
     - Afriat efficiency. 1.0 = perfectly rational. Lower = more waste.
   * - ``mpi``
     - [0, 1)
     - Money Pump Index. 0.0 = unexploitable. Higher = more inconsistent.
   * - ``hm_consistent / hm_total``
     - [0, 1]
     - Houtman-Maks fraction of rationalizable observations.

Call ``r.summary()`` for a formatted report, or ``r.to_dict()`` for serialization.

Which API?
----------

PyRevealed has two APIs. Use whichever fits your task:

**Engine** — batch scoring (thousands of users, Rust backend):

.. code-block:: python

   from pyrevealed.engine import Engine
   from pyrevealed.datasets import load_demo

   results = Engine(metrics=["garp", "ccei", "mpi"]).analyze_arrays(load_demo())
   # list[EngineResult] — flat scores, ready for pandas

**Function API** — deep single-user analysis (violation details, preference graphs):

.. code-block:: python

   from pyrevealed import BehaviorLog, validate_consistency, compute_integrity_score
   import numpy as np

   prices = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
   quantities = np.array([[4.0, 1.0], [1.0, 4.0], [2.0, 2.0]])
   session = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

   garp = validate_consistency(session)     # GARPResult: violations, preference matrices
   ccei = compute_integrity_score(session)  # AEIResult: binary search details

Your Own Data
-------------

Your data needs to be ``(prices T*K, quantities T*K)`` per user, where T = observations and K = goods.

**From a DataFrame:**

.. code-block:: python

   import numpy as np
   from pyrevealed.engine import Engine

   # Suppose df has columns: user_id, obs_id, price_good_0, ..., qty_good_0, ...
   price_cols = [c for c in df.columns if c.startswith("price_")]
   qty_cols = [c for c in df.columns if c.startswith("qty_")]

   users = []
   for uid, group in df.groupby("user_id"):
       prices = group[price_cols].values.astype(np.float64)
       quantities = group[qty_cols].values.astype(np.float64)
       users.append((prices, quantities))

   results = Engine(metrics=["garp", "ccei", "mpi"]).analyze_arrays(users)

**Into pandas:**

.. code-block:: python

   import pandas as pd
   df = pd.DataFrame([r.to_dict() for r in results])
   print(df[["is_garp", "ccei", "mpi"]].describe())

Next Steps
----------

- :doc:`budget/tutorial` — budget choice analysis in depth
- :doc:`menu/tutorial_menu_choice` — menu/discrete choice analysis
- :doc:`api` — full API reference
- :doc:`performance` — benchmarks and scaling
