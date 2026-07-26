VS Code and Coding-Agent Onboarding
===================================

Purpose
-------

This guide helps a new ws3 contributor set up a practical local VS Code
workflow and collaborate effectively with a local coding agent working in the
same checkout.

It is written for real project work, not as a generic AI-tools overview.
The goal is to help a newcomer become productive without losing track of the
repo/runtime rules that matter in ws3.

Use This Guide For
------------------

Use this guide when you want to:

- open ws3 in VS Code and do day-to-day development from a local checkout;
- work with a local coding agent that can read and edit files in the repo;
- understand what work can be delegated safely and what still needs active
  human review;
- onboard a new student or collaborator who is comfortable with code but has
  not yet learned the ws3 workflow.

This guide assumes you are working from a local ws3 checkout, not from a
read-only browser view of the repo.

Minimum Local Setup
-------------------

Before thinking about prompts or agent workflow, get the local environment into
a known-good state.

1. Install the normal local tools:

   - Git
   - Python 3.9+
   - VS Code

2. Open the ws3 repo root in VS Code.

3. In the integrated terminal, follow the canonical bootstrap:

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate
      pip install -e ".[dev,docs]"

4. Confirm the repo can pass the minimum shell checks from the active
   ``.venv`` before starting model work:

   - ``python -c "import ws3; print(ws3.__version__)"``
   - ``pytest --version``
   - ``sphinx-build --version``

Quick Contract
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Seam
     - Contract
   * - Canonical repo root
     - Use the active checkout root as the canonical repository root for
       commands, patches, and file references. Prefer repo-relative examples
       in published docs rather than machine-specific absolute paths.
   * - Python environment
     - Use a repo-local ``.venv`` and install ``.[dev,docs]`` before ws3
       development or docs work.
   * - Source layout
     - Package code lives in ``ws3/`` at the repo root (NOT ``src/ws3/``).
       Modules: ``common.py``, ``core.py``, ``forest.py``, ``forest_helper.py``,
       ``financial.py``, ``opt.py``, ``spatial.py``.
   * - Tests
     - Live in ``tests/``. Run with ``pytest`` from the repo root.
   * - Docs
     - Live in ``docs/source/``. Build with ``sphinx-build -b html
       docs/source _build/html``. Deployed at
       https://ubc-fresh.github.io/ws3/.
   * - Examples
     - Jupyter notebooks in ``examples/``. Some reference data in
       ``examples/data/``.

Module Map
----------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Responsibility
   * - ``ws3.common``
     - Global constants (HORIZON_DEFAULT, PERIOD_LENGTH_DEFAULT, MIN_AGE_DEFAULT,
       MAX_AGE_DEFAULT), utility functions, rasterio integration
   * - ``ws3.core``
     - ``Interpolator`` class, ``Curve`` class with arithmetic operators
       (__add__, __sub__, __mul__, __truediv__)
   * - ``ws3.forest``
     - ``ForestModel`` (main model class), ``DevelopmentType``, ``Action``,
       ``GreedyAreaSelector``, parallel worker functions
   * - ``ws3.forest_helper``
     - ``PersistentWorkerPool``, batch utilities, worker initialization
   * - ``ws3.financial``
     - Financial analysis functions (NPV, rotation economics)
   * - ``ws3.opt``
     - ``Problem``, ``Variable``, ``Constraint`` classes; solver bindings
       (Gurobi, PuLP/HiGHS)
   * - ``ws3.spatial``
     - ``ForestRaster`` class for spatial schedule allocation

Class Hierarchy
---------------

.. mermaid::

   graph TD
     FM["ForestModel"] --> DT["DevelopmentType"]
     FM --> ACT["Action"]
     FM --> CURVE["Curve"]
     FM --> SEL["AreaSelector"]
     FM --> OPT["Problem"]
     CURVE --> INTERP["Interpolator"]
     ACT --> TRANS["Transition"]
     FM --> FIN["Financial functions"]
     FM --> SPAT["ForestRaster"]

Data Flow
---------

The typical ws3 workflow follows this data flow:

.. mermaid::

   graph LR
     INV["Forest Inventory<br/>(spatial data)"] --> AGG["Aggregation<br/>(strata/age classes)"]
     AGG --> FM["ForestModel<br/>development types"]
     FM --> ACT["Actions defined"]
     FM --> CURVE["Growth curves defined"]
     FM --> SIM["Simulation<br/>(period-by-period)"]
     SIM --> SCHED["Activity schedule<br/>(aspatial output)"]
     SCHED --> OPT["Optimization<br/>(if applicable)"]
     OPT --> FINAL["Optimal schedule"]
     FINAL --> SPAT["Spatial allocation<br/>(if applicable)"]

Common Patterns
---------------

Building a Model From Scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ws3.forest import ForestModel

   # Create a model with default settings
   model = ForestModel()

   # Define development types from inventory data
   model.add_development_types(inventory_df)

   # Define growth curves
   model.add_curves(curve_data)

   # Define actions
   model.add_action("harvest", descr="Clearcut harvest")

   # Run simulation
   results = model.run_simulation()

Running Optimization
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ws3.opt import Problem

   # Build optimization problem
   prob = Problem()

   # Add variables and constraints
   prob.add_variable("harvest_area", vtype="continuous", lb=0)
   prob.add_constraint("area_limit", sense="<=", rhs=1000)

   # Solve
   prob.solve(solver="highs")

   # Extract solution
   solution = prob.get_solution()

Extending ws3
~~~~~~~~~~~~~

To extend ws3, subclass the relevant base classes:

.. code-block:: python

   from ws3.forest import ForestModel, AreaSelector

   class MyAreaSelector(AreaSelector):
       def operate(self, period, acode, target_area, mask=None,
                   commit_actions=True, verbose=False):
           # Custom selection logic
           pass

   class MyForestModel(ForestModel):
       def __init__(self):
           super().__init__()
           self.area_selector = MyAreaSelector(self)

Platform Notes
--------------

Linux/macOS
~~~~~~~~~~~

The canonical development platform. All examples assume POSIX shell.

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,docs]"

Windows
~~~~~~~

ws3 runs on Windows but some geospatial dependencies (rasterio, fiona) may
require additional setup. Use conda for geospatial packages:

.. code-block:: powershell

   conda create -n ws3 python=3.12
   conda activate ws3
   pip install -e ".[dev,docs]"

Known Issues
~~~~~~~~~~~~

- PaCal library has compatibility issues with newer numpy versions. The
  ``ws3.common`` module sets ``PACAL_BROKEN = True`` to work around this.
  Functions that depend on PaCal will not work without a patched version.