Installation
============

.. code-block:: bash

   pip install prefgraph

This installs the core library with NumPy, SciPy, NetworkX, Numba, and Polars. The Rust engine compiles automatically if a Rust toolchain is available. If not, a pure-Python fallback handles GARP, CCEI, MPI, and HM.

Extras
------

Some workflows need additional packages. Install them with bracket syntax.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Extra
     - Install command
     - What it adds
   * - ``parquet``
     - ``pip install "prefgraph[parquet]"``
     - PyArrow for reading and writing Parquet files
   * - ``datasets``
     - ``pip install "prefgraph[datasets]"``
     - Pandas for loading real-world dataset loaders
   * - ``viz``
     - ``pip install "prefgraph[viz]"``
     - Matplotlib for plotting
   * - ``dev``
     - ``pip install "prefgraph[dev]"``
     - Pytest, Mypy, Ruff for development
   * - ``all``
     - ``pip install "prefgraph[all]"``
     - Everything above plus Jupyter and Sphinx

The core ``pip install prefgraph`` is enough for ``load_demo``, ``Engine``, ``analyze_arrays``, and ``analyze_menus``. You only need extras when working with Parquet files, real-world datasets, or visualization.

Choose Your Workflow
--------------------

PrefGraph has three main entry points depending on what data you have.

**I already have per-user NumPy arrays.** Call ``Engine.analyze_arrays()`` directly. Each user is a tuple of ``(prices, quantities)`` arrays with shape ``(T, K)`` where T is the number of observations and K is the number of goods. See the :doc:`Loading Data <quickstart>` guide for examples.

**I have a Parquet file or DataFrame.** Call ``Engine.analyze_parquet()`` with column names for user ID, prices, and quantities. The engine groups by user and scores in one call. Wide format needs ``cost_cols`` and ``action_cols``. Long format needs ``item_col``, ``time_col``, ``cost_col``, and ``action_col``. Requires ``pip install "prefgraph[parquet]"``.

**I have clickstream or event logs.** Build menus from your events first, then call ``Engine.analyze_menus()``. Each user is a tuple of ``(menus, choices, n_items)`` where menus are lists of integer item indices. The :doc:`Loading Data <quickstart>` guide shows the full pipeline from raw events to scored results.
