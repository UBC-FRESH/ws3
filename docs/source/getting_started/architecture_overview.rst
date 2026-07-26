Architecture Overview
=====================

This page explains how ws3's components fit together and how data flows
through the system. Understanding this architecture will help you extend
ws3, debug issues, and write efficient code.

High-Level Architecture
-----------------------

ws3 follows a modular architecture where each component has a single
responsibility. The :py:class:`ws3.forest.ForestModel` class acts as the
central coordinator.

.. mermaid::

   graph TD
     subgraph "Core"
       FM["ForestModel<br/>Central coordinator"]
       DT["DevelopmentType<br/>Forest strata"]
       ACT["Action<br/>Management interventions"]
       CURVE["Curve<br/>Growth trajectories"]
     end

     subgraph "Optimization"
       PROB["Problem<br/>Optimization formulation"]
       VAR["Variable<br/>Decision variables"]
       CONST["Constraint<br/>Problem constraints"]
     end

     subgraph "Spatial"
       RASTER["ForestRaster<br/>Spatial allocation"]
     end

     subgraph "Financial"
       FIN["Financial functions<br/>NPV, rotation economics"]
     end

     subgraph "Helpers"
       POOL["PersistentWorkerPool<br/>Parallel workers"]
       SEL["AreaSelector<br/>Harvest targeting"]
     end

     FM --> DT
     FM --> ACT
     FM --> CURVE
     FM --> PROB
     FM --> RASTER
     FM --> FIN
     FM --> POOL
     FM --> SEL

Data Flow
---------

The typical ws3 workflow follows this data flow:

.. mermaid::

   graph LR
     DATA["Inventory Data<br/>(CSV, GeoJSON, etc.)"] --> AGG["Aggregation<br/>to development types"]
     AGG --> FM["ForestModel<br/>initialized"]
     FM --> ACT["Actions defined"]
     FM --> CURVE["Curves defined"]
     FM --> SIM["Simulation<br/>period-by-period"]
     SIM --> OPT["Optimization<br/>(if applicable)"]
     OPT --> FINAL["Optimal schedule"]
     FINAL --> SPAT["Spatial allocation<br/>(if applicable)"]

Component Responsibilities
--------------------------

ForestModel
~~~~~~~~~~~

The :py:class:`ws3.forest.ForestModel` class is the central hub. It:

- Stores development types, actions, and curves
- Coordinates simulation and optimization
- Provides methods for adding/removing components
- Tracks area and volume across the planning horizon

.. code-block:: python

   model = ForestModel()
   model.add_development_type(...)
   model.add_action(...)
   model.add_curve(...)
   results = model.run_simulation(horizon=20)

DevelopmentType
~~~~~~~~~~~~~~~

A :py:class:`ws3.forest.DevelopmentType` represents a homogeneous group
of forest stands. Each DT has:

- A unique code (e.g., "DF-SI50")
- Current area (hectares)
- Current age
- Species, site index, and other attributes

Development types are the fundamental unit of inventory tracking. The
model moves area between development types as actions are applied.

.. code-block:: python

   dt = DevelopmentType(
       code="DF-SI50",
       area=500.0,
       age=20,
       species="Pseudotsuga menziesii",
       site_index=50
   )

Action
~~~~~~

An :py:class:`ws3.forest.Action` is a management intervention. Each action:

- Has a code (e.g., "HARV", "THIN")
- Has a description
- Specifies which components are affected (volume, basal area, etc.)
- Defines transitions to new development types

.. code-block:: python

   action = Action(
       code="HARV",
       descr="Clearcut harvest",
       components=["volume"],
       transitions={"DF-SI50": "Bare"}
   )

Curve
~~~~~

A :py:class:`ws3.core.Curve` defines a growth trajectory. Curves:

- Map age (x-axis) to attribute values (y-axis)
- Support interpolation between defined points
- Support arithmetic operations (add, subtract, multiply, divide)
- Are used to calculate volume, biomass, value at any age

.. code-block:: python

   curve = Curve(
       x=[0, 10, 20, 30, 40, 50],
       y=[0, 5, 25, 65, 120, 200],
       name="DF_volume"
   )
   volume_at_age_25 = curve(25)  # Interpolated value

Problem (Optimization)
~~~~~~~~~~~~~~~~~~~~~~

The :py:class:`ws3.opt.Problem` class formulates an optimization problem.
It:

- Manages decision variables (:py:class:`ws3.opt.Variable`)
- Manages constraints (:py:class:`ws3.opt.Constraint`)
- Supports multiple solvers (HiGHS, Gurobi, PuLP)
- Returns solution values and objective function values

.. code-block:: python

   prob = Problem()
   x = prob.add_variable("harvest_area", vtype="continuous", lb=0)
   prob.add_constraint("limit", expr=x <= 100)
   prob.set_objective(x, sense="maximize")
   prob.solve(solver="highs")

ForestRaster (Spatial)
~~~~~~~~~~~~~~~~~~~~~~

The :py:class:`ws3.spatial.ForestRaster` class handles spatial allocation.
It:

- Reads forest inventory as raster data (using rasterio)
- Allocates harvest targets to specific pixels
- Enforces spatial constraints (contiguity, adjacency)
- Outputs spatially explicit harvest maps

.. code-block:: python

   raster = ForestRaster("inventory.tif")
   raster.allocate_harvest(target_area=100)

PersistentWorkerPool (Parallel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :py:class:`ws3.forest_helper.PersistentWorkerPool` class manages
parallel computation. It:

- Uses :py:class:`concurrent.futures.ProcessPoolExecutor`
- Serializes workers with dill for cross-process communication
- Provides automatic batching for large workloads
- Manages worker lifecycle (creation, reuse, shutdown)

.. code-block:: python

   pool = PersistentWorkerPool(n_workers=4)
   results = pool.map(process_function, work_items)

AreaSelector
~~~~~~~~~~~~

The :py:class:`ws3.forest.AreaSelector` class (and its subclass
:py:class:`ws3.forest.GreedyAreaSelector`) determines which development
types to harvest when a target area is specified. The greedy selector
always harvests from the oldest stands first.

.. code-block:: python

   selector = GreedyAreaSelector(model)
   selector.operate(period=0, acode="HARV", target_area=50)

Module Map
----------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Key Classes/Functions
   * - ``ws3.common``
     - Global constants, utility functions, rasterio integration
   * - ``ws3.core``
     - :py:class:`Interpolator`, :py:class:`Curve`
   * - ``ws3.forest``
     - :py:class:`ForestModel`, :py:class:`DevelopmentType`,
       :py:class:`Action`, :py:class:`GreedyAreaSelector`
   * - ``ws3.forest_helper``
     - :py:class:`PersistentWorkerPool`, batch utilities
   * - ``ws3.financial``
     - NPV calculation, rotation economics functions
   * - ``ws3.opt``
     - :py:class:`Problem`, :py:class:`Variable`, :py:class:`Constraint`
   * - ``ws3.spatial``
     - :py:class:`ForestRaster`

Design Principles
-----------------

ws3 follows these design principles:

1. **Single Responsibility**: Each class has one clear purpose
2. **Composition over Inheritance**: Use composition to combine behavior
3. **Explicit Data Flow**: Data moves through the system in a predictable order
4. **Extensibility**: Subclass base classes to add custom behavior
5. **Testability**: Each component can be tested in isolation

Common Pitfalls
---------------

1. **Forgetting transitions**: Every action must define transitions for
   all affected development types. Missing transitions cause errors.

2. **Mismatched curve lengths**: Curves must have the same x-values
   (ages) as the model's age classes.

3. **Ignoring area conservation**: The total area in the model should
   remain constant (area moves between development types, never disappears).

4. **Solver selection**: HiGHS is the default and works for most problems.
   Use Gurobi only if you need its specific features (quadratic objectives,
   MIP variables, etc.).