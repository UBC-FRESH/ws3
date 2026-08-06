Chapter 1: Forest Estate Models
================================

Learning Objectives
-------------------

After reading this chapter, you should be able to:

- Explain what a wood supply model (WSM) is and why forest managers use them
- Describe the key components of a forest estate model: area of interest,
  development types, age classes, actions, and growth curves
- Build a minimal ws3 model from scratch
- Run a simulation and interpret the output
- Understand how ws3 fits into the broader forest planning workflow

What Is a Wood Supply Model?
----------------------------

Imagine you manage a 50,000-hectare forest. You want to know: *What happens
if we harvest 500 hectares per year for the next 100 years?* Or: *How much
wood can we sustainably harvest while keeping the forest healthy?*

These are **wood supply planning problems** (WSPP). A **wood supply model**
(WSM) is a computer program that simulates how a forest changes over time
given a set of management activities.

Think of it like a flight simulator for forests. Just as a flight simulator
lets pilots practice scenarios without risking real aircraft, a wood supply
model lets forest managers explore "what if" scenarios without committing
to expensive real-world experiments.

Why Do We Need Them?
--------------------

Forest management operates on timescales that dwarf human lifespans. A
typical timber rotation lasts 50-200 years. Decisions made today affect
the forest for generations.

Human beings are terrible at intuitively reasoning about compound growth,
nonlinear dynamics, and multi-decade consequences. We tend to:

- **Underestimate exponential growth**: A forest that doubles every 20
  years seems manageable until you realize it will cover the Earth in
  400 years.
- **Overlook feedback loops**: Harvesting changes future growth rates,
  which changes future harvest capacity, which changes future revenue.
- **Ignore spatial constraints**: You can't harvest a hectare today if
  it will be needed for watershed protection next year.

Wood supply models handle this complexity by:

1. **Tracking inventory explicitly**: Every hectare is classified by
   species, age class, and site quality. The model knows exactly what
   grows where.
2. **Simulating growth realistically**: Growth curves capture how volume,
   biomass, and value change with age for each species-site combination.
3. **Enforcing constraints**: Minimum rotation ages, habitat requirements,
   and sustainable yield targets are built into the simulation.
4. **Optimizing decisions**: Given multiple possible actions, the model
   finds the schedule that maximizes your objective (profit, sustained
   yield, carbon sequestration, etc.).

Key Concepts
------------

Before we dive into ws3, let's define the core vocabulary.

Area of Interest (AOI)
~~~~~~~~~~~~~~~~~~~~~~

The **area of interest** is the geographic boundary of your modelling
study. It could be:

- A entire forest management unit (FMU)
- A single timber harvest block
- A watershed
- A private landholding

Everything outside the AOI is ignored. Everything inside is modelled.

.. figure:: /examples/images/aoi_example.png
   :alt: Example area of interest
   :align: center

   An AOI might be defined by administrative boundaries, ownership, or
   natural features.

Development Types
~~~~~~~~~~~~~~~~~

A **development type** (DT) is a unique combination of forest attributes
that determines how the stand will grow and respond to management. Common
attributes include:

- **Species**: Douglas-fir, western red cedar, spruce, etc.
- **Site Index (SI)**: A measure of site productivity (how tall trees
  grow on this site)
- **Stocking**: Number of stems per hectare
- **Canopy cover**: Closed, open, or bare

Each development type represents a homogeneous group of stands that will
respond identically to the same management action.

In ws3, development types are the fundamental unit of inventory. The model
tracks area (in hectares) for each development type at each time step.

Age Classes
~~~~~~~~~~~

**Age classes** divide the planning horizon into discrete time steps. Each
age class represents a period (typically 5-20 years) during which growth
and management actions are assumed constant.

For example, with a 100-year horizon and 10-year age classes:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Age Class
     - Years
   * - 1
     - 0-10
   * - 2
     - 10-20
   * - 3
     - 20-30
   * - ...
     - ...
   * - 10
     - 90-100

Actions
~~~~~~~

An **action** is a management intervention applied to a development type
in a specific age class. Common actions include:

- **Harvest**: Clearcut, selection cut, shelterwood cut
- **Thinning**: Commercial thin, pre-commercial thin
- **Planting**: Replant after harvest
- **Prescribe**: Do nothing (let nature take its course)

Each action has:

- A **code** (e.g., "HARV", "THIN", "PLNT")
- A **description**
- A set of **components** (what attributes change)
- A set of **transitions** (what happens after the action)

Transitions
~~~~~~~~~~~

A **transition** defines what happens to a development type after an action
is applied. For example:

- A clearcut harvest (action "HARV") on a Douglas-fir stand (DT "DF-SI50")
  transitions to a bare site (DT "BARE")
- A thinning (action "THIN") on a young Douglas-fir stand reduces stem
  density but keeps the stand as Douglas-fir

Transitions are the bridge between actions and growth. They answer the
question: *After I do X to this stand, what does it become?*

Growth Curves
~~~~~~~~~~~~~

**Growth curves** describe how forest attributes change with age. The most
common curves are:

- **Volume curves**: Total merchantable volume (m³/ha) vs. age
- **Basal area curves**: Cross-sectional area of tree trunks vs. age
- **Height curves**: Dominant height vs. age
- **Value curves**: Dollar value per unit volume vs. age

Curves are typically fitted to sample plot data or derived from provincial
growth-and-yield tables. In ws3, the :py:class:`ws3.core.Curve` class
handles curve definition, interpolation, and algebra.

.. figure:: /examples/images/growth_curve_example.png
   :alt: Example volume curve
   :align: center

   A typical volume curve shows sigmoidal growth: slow initially, rapid
   in middle ages, then asymptoting.

How ws3 Implements This
-----------------------

Now let's see how these concepts map to ws3 classes.

.. mermaid::

   graph TD
     FM["ForestModel"] --> DT["DevelopmentType"]
     FM --> ACT["Action"]
     FM --> CURVE["Curve"]
     FM --> SEL["AreaSelector"]
     CURVE --> INTERP["Interpolator"]

The :py:class:`ws3.forest.ForestModel` class is the central hub. It
contains:

- A collection of :py:class:`ws3.forest.DevelopmentType` objects
- A collection of :py:class:`ws3.forest.Action` objects
- Growth curves (via :py:class:`ws3.core.Curve`)
- An area selector for distributing harvest targets

A Minimal ws3 Model
~~~~~~~~~~~~~~~~~~~

Let's build a simple model. Suppose we have:

- 1,000 hectares of Douglas-fir on Site Index 50
- We want to harvest 50 hectares per year for 20 years
- Growth follows a standard Douglas-fir volume curve

.. code-block:: python

   from ws3.forest import ForestModel
   from ws3.core import Curve

   # Create model with required parameters
   model = ForestModel(
       model_name="example",
       model_path="/path/to/data",
       base_year=2024,
       horizon=20,
       period_length=10,
       max_age=200
   )

   # Development types are loaded from Woodstock-format data files,
   # not constructed individually. The typical workflow is:
   #
   #   model.import_areas_section()       # loads areas into dtypes
   #   model.import_yields_section()      # loads yield curves
   #   model.import_actions_section()     # loads action definitions
   #   model.import_transitions_section() # loads transition rules
   #
   # For programmatic curve registration, use register_curve():
   #   volume_curve = Curve(label="DF-SI50_volume", is_volume=True,
   #                        points=[(0,0),(10,5),(20,20),...,(100,490)])
   #   model.register_curve(volume_curve)

   # Define a volume curve (simplified)
   volume_curve = Curve(
       label="DF-SI50_volume",
       is_volume=True,
       points=[(0, 0), (10, 5), (20, 20), (30, 50), (40, 100),
               (50, 180), (60, 280), (70, 380), (80, 450),
               (90, 480), (100, 490)]
   )
   model.register_curve(volume_curve)

   # Actions are defined in Woodstock-format ACTION section files
   # and imported via model.import_actions_section()
   # Transitions are defined in the TRANSITION section and imported
   # via model.import_transitions_section()

   # Simulation proceeds by:
   # 1. Resetting actions: model.reset_actions()
   # 2. Setting applied actions for each period
   # 3. Growing the model: model.grow(start_period=1)
   # For optimization, build a ws3.opt.Problem, solve it, then
   # compile the schedule via model.compile_schedule(problem)

The output will show how area moves between development types over time
as the harvest action is applied.

Summary
-------

In this chapter, we covered:

- **Wood supply models** simulate forest growth and management over
  multi-decade horizons
- **Development types** are homogeneous groups of stands that respond
  identically to management
- **Age classes** divide time into discrete periods
- **Actions** are management interventions (harvest, thin, plant)
- **Transitions** define what happens after an action
- **Growth curves** describe how attributes change with age
- **ForestModel** is ws3's central class that ties everything together

Exercises
---------

**Exercise 1 (Easy)**: Install ws3 and run the following code to verify
your setup:

.. code-block:: python

   import ws3
   print(f"ws3 version: {ws3.__version__}")

**Exercise 2 (Medium)**: Extend the model to include Spruce:

1. Add Spruce-SI40 to the AREAS section file (500 hectares)
2. Add a separate yield curve for Spruce-SI40
3. Run the simulation and compare the volume trajectories

**Exercise 3 (Hard)**: The current model assumes all 50 hectares harvested
each year come from the same development type. Modify the code to:

1. Add a greedy area selector that always harvests from the oldest stands
2. Track which development type was harvested each period
3. Plot the area remaining in each development type over time

.. hint::

   Look at the :py:class:`ws3.forest.GreedyAreaSelector` class and the
   :meth:`ws3.forest.ForestModel.operate` method.

Further Reading
---------------

- :doc:`ch02_forest_inventory` — How to prepare forest inventory data for ws3
- :doc:`ch03_growth_and_yield` — Growth curve fitting and interpolation
- :doc:`/getting_started/quickstart` — Hands-on quickstart tutorial