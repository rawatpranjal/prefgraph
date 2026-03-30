Loading Data
============

Consistency scores are only meaningful when the input represents genuine feasible choices. For budget data, prices must be positive and quantities non-negative. For menu data, reconstruct menus from what the user actually saw, keep only sessions with exactly one purchase, ensure the chosen item appears in the menu, and remap item IDs to contiguous ``0..N-1`` indices. The Engine validates types, shapes, value ranges, and menu structure before scoring. If your data passes validation but the menus or budgets do not approximate real choice sets, the resulting scores will measure data artifacts rather than behavior.

Synthetic data (Rust-parallel generators)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PrefGraph ships four Rayon-parallel generators that produce data in the exact format the Engine expects. Each accepts a ``rationality`` parameter (0.0 = random, 1.0 = utility-maximizing) and a ``seed`` for reproducibility. Generation runs entirely in Rust with the GIL released, so 100K users take roughly one second.

.. code-block:: python

   from prefgraph import generate_random_budgets, generate_random_menus
   from prefgraph.engine import Engine, results_to_dataframe

   # Budget data: each user has 15 shopping trips across 5 goods.
   # rationality=0.7 means 70% of choices follow Cobb-Douglas utility,
   # 30% are random — simulating realistic noisy behaviour.
   budget_data = generate_random_budgets(
       n_users=100_000, n_obs=15, n_goods=5,
       functional_form="cobb_douglas",   # also "ces" or "leontief"
       rationality=0.7, noise_scale=0.3, seed=42,
   )
   # Score all 100K users in one Rust-parallel batch call
   engine = Engine(metrics=["garp", "ccei", "hm"])
   df = results_to_dataframe(engine.analyze_arrays(budget_data))
   print(df[["is_garp", "ccei", "hm_consistent", "hm_total"]].head())

.. code-block:: text

      is_garp  ccei  hm_consistent  hm_total
   0     True   1.0             15        15
   1     True   1.0             15        15
   2     True   1.0             15        15
   3     True   1.0             15        15
   4     True   1.0             15        15

.. code-block:: python

   # Menu data: each user picks one item from a variable-size menu (2–5 items).
   # logit choice model adds realistic substitution noise.
   menu_data = generate_random_menus(
       n_users=100_000, n_obs=10, n_items=5,
       menu_size=(2, 5), choice_model="logit",  # also "fixed_ranking" or "uniform"
       temperature=1.0, rationality=0.7, seed=42,
   )
   engine2 = Engine(metrics=["hm"])
   df2 = results_to_dataframe(engine2.analyze_menus(menu_data))
   print(df2[["is_sarp", "n_sarp_violations", "hm_consistent", "hm_total"]].head())

.. code-block:: text

      is_sarp  n_sarp_violations  hm_consistent  hm_total
   0    False                  6              3         5
   1    False                  3              4         5
   2    False                  6              3         5
   3    False                  3              4         5
   4    False                  6              3         5

Production and intertemporal generators follow the same pattern:

.. code-block:: python

   from prefgraph import generate_random_production, generate_random_intertemporal

   # Production: 10K firms, 3 inputs + 2 outputs
   prod_data = generate_random_production(
       n_users=10_000, n_obs=15, n_inputs=3, n_outputs=2,
       functional_form="cobb_douglas", rationality=0.7, seed=42,
   )

   # Intertemporal: 10K agents, 5 time periods, discount factor 0.8-0.99
   inter_data = generate_random_intertemporal(
       n_users=10_000, n_obs=10, n_periods=5,
       discount_factor=(0.8, 0.99), rationality=0.7, seed=42,
   )

   # prod_data:  10,000 users, each (15, 5) — 15 observations × (3 inputs + 2 outputs)
   # inter_data: 10,000 users, each (10, 5) — 10 observations × 5 periods

Both ``n_obs`` and ``menu_size`` accept an ``int`` for fixed counts or a ``(min, max)`` tuple for variable counts per user. A pure-NumPy fallback runs automatically if the Rust extension is unavailable.

Budget data from Parquet (wide format)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wide format means one row per observation with separate price and quantity columns for each good.

.. code-block:: python

   import polars as pl
   from prefgraph.engine import Engine

   # Wide format: one row per observation, separate columns for each good's price/qty.
   # Example schema: user_id, t, p_milk, p_bread, q_milk, q_bread
   path = "my_budget_wide.parquet"

   # analyze_parquet reads the file, groups by user, and scores in Rust
   engine = Engine(metrics=["garp", "ccei", "mpi", "hm"])
   results_df = engine.analyze_parquet(
       path,
       user_col="user_id",
       cost_cols=["p_milk", "p_bread"],      # price columns (one per good)
       action_cols=["q_milk", "q_bread"],    # quantity columns (one per good)
   )
   print(results_df[["is_garp", "ccei", "mpi", "hm_consistent", "hm_total"]].head())

.. code-block:: text

            is_garp      ccei       mpi  hm_consistent  hm_total
   user_id
   0          False  0.880033  0.253057              8        10
   1          False  0.676351  0.423827              8        10
   2           True  1.000000  0.000000             10        10
   3          False  0.937556  0.129736              8        10
   4          False  0.942781  0.084905              9        10

Budget data from Parquet (long format)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Long format means one row per (user, time, item) with columns for item id, price, and quantity.

.. code-block:: python

   import polars as pl
   from prefgraph.engine import Engine

   # Long format: one row per (user, time, item) — the engine pivots internally.
   # Example schema: user_id, t, item, price, quantity
   path = "my_budget_long.parquet"

   # Passing item_col + time_col tells the engine to pivot to wide format per user
   engine = Engine(metrics=["garp", "ccei", "mpi", "hm"])
   results_df = engine.analyze_parquet(
       path,
       user_col="user_id",
       item_col="item",        # identifies the good
       time_col="t",           # identifies the observation
       cost_col="price",       # single price column
       action_col="quantity",  # single quantity column
   )
   print(results_df[["is_garp", "ccei", "mpi", "hm_consistent", "hm_total"]].head())

.. code-block:: text

            is_garp      ccei       mpi  hm_consistent  hm_total
   user_id
   0          False  0.722626  0.322987              7        10
   1          False  0.853556  0.349261              7        10
   2          False  0.747051  0.262645              9        10
   3          False  0.850580  0.210591              8        10
   4          False  0.842826  0.151106              9        10

Budget data from a DataFrame (per‑user arrays)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have a DataFrame in memory, build per‑user price/quantity matrices and pass them to ``analyze_arrays``.

.. code-block:: python

   import polars as pl
   import numpy as np
   from prefgraph.engine import Engine

   # When you need full control over the pivot step (e.g., custom imputation),
   # build per-user (prices, quantities) arrays yourself.
   df = pl.read_parquet("my_budget_long.parquet")

   users: list[tuple[np.ndarray, np.ndarray]] = []
   for uid, g in df.group_by("user_id", maintain_order=True):
       # Pivot from long to wide: rows = observations, columns = goods
       price_wide = g.pivot(values="price", index="t", on="item").sort("t").drop("t")
       qty_wide   = g.pivot(values="quantity", index="t", on="item").sort("t").drop("t")
       P = price_wide.to_numpy()                  # shape: (n_obs, n_goods)
       Q = qty_wide.fill_null(0).to_numpy()       # missing qty → 0 (not purchased)
       users.append((P, Q))

   # analyze_arrays accepts list[tuple[ndarray, ndarray]] directly
   engine = Engine(metrics=["garp", "ccei", "mpi", "hm"])
   results = engine.analyze_arrays(users)
   print(results[0])

.. code-block:: text

   EngineResult: [-] 13 violations  ccei=0.7226  mpi=0.3230  hm=7/10  (50905us)

Menu data from Parquet (events → menus)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For clickstream data, build menus from what the user actually saw (e.g., viewed items) and use the purchased/clicked item as the choice. ``analyze_menus`` expects per‑user tuples ``(menus, choices, n_items)`` where menus are lists of item indices.

.. code-block:: python

   import polars as pl
   from prefgraph.engine import Engine

   # Clickstream events: "view" = item appeared on screen, "purchase" = user bought it.
   # The goal is to reconstruct menus (what the user saw) and choices (what they picked).
   ev = pl.read_parquet("my_events.parquet").filter(
       pl.col("event_type").is_in(["view", "purchase"])
   )

   user_batches: list[tuple[list[list[int]], list[int], int]] = []

   for uid, ug in ev.group_by("user_id", maintain_order=True):
       # Step 1: group views by session to build menus
       views = ug.filter(pl.col("event_type") == "view").group_by("session_id").agg(
           pl.col("product_id").unique().alias("viewed")
       )
       # Step 2: keep only sessions with exactly one purchase (clean single-choice)
       buys = ug.filter(pl.col("event_type") == "purchase").group_by("session_id").agg(
           pl.col("product_id").n_unique().alias("n_buy"),
           pl.col("product_id").first().alias("choice")
       ).filter(pl.col("n_buy") == 1)

       # Step 3: join views + purchases; ensure the chosen item is in the menu
       sess = buys.join(views, on="session_id", how="inner")
       sess = sess.with_columns(
           pl.concat_list([pl.col("viewed"), pl.col("choice").map_elements(lambda x: [x])])
             .list.unique()
             .alias("menu")
       ).with_columns(pl.col("menu").list.len().alias("m"))
       sess = sess.filter((pl.col("m") >= 2) & (pl.col("m") <= 50))  # drop trivial menus

       # Step 4: remap product IDs to 0..N-1 (Engine expects contiguous indices)
       all_items = sorted({int(i) for ms in sess["menu"] for i in ms})
       to_local = {pid: i for i, pid in enumerate(all_items)}
       menus   = [[to_local[int(i)] for i in ms] for ms in sess["menu"]]
       choices = [to_local[int(c)] for c in sess["choice"]]
       n_items = len(all_items)

       if menus:
           user_batches.append((menus, choices, n_items))

   # analyze_menus expects list[tuple[menus, choices, n_items]] — one tuple per user
   engine = Engine(metrics=["hm"])
   results = engine.analyze_menus(user_batches)
   print(results[:3])

.. code-block:: text

   [MenuResult: [+] SARP-consistent  hm=17/17  (759us),
    MenuResult: [-] 6 SARP violations  hm=16/18  (560us),
    MenuResult: [-] 4 SARP violations  hm=16/18  (398us)]
