Quickstart Tutorial
===================

This tutorial gets you up and running with ws3 in under 10 minutes.
You'll build a simple wood supply model, run a simulation, and inspect
the output.

Prerequisites
-------------

- ws3 installed (see :doc:`installation`)
- Python 3.9+ available in your terminal

Step 1: Import ws3
------------------

Open a Python interpreter or Jupyter notebook and import ws3:

.. code-block:: python

   import ws3
   print(f"ws3 version: {ws3.__version__}")

Step 2: Create a Forest Model
------------------------------

The :py:class:`ws3.forest.ForestModel` class is the central hub for
building a wood supply model.

.. code-block:: python

   from ws3.forest import ForestModel

   # Create a model with default settings
   model = ForestModel()

   print(f"Model created: {model}")

Step 3: Add Development Types
-----------------------------

Development types represent homogeneous groups of forest stands. Each
type has a code, area (hectares), age, and attributes.

.. code-block:: python

   # Add three development types representing different forest conditions
   model.add_development_type(
       code="DF-SI50",
       area=500.0,
       age=20,
       species="Pseudotsuga menziesii",
       site_index=50
   )

   model.add_development_type(
       code="Spruce-SI40",
       area=300.0,
       age=40,
       species="Picea sitchensis",
       site_index=40
   )

   model.add_development_type(
       code="Bare",
       area=0.0,
       age=0,
       species="",
       site_index=0
   )

   print(f"Added 3 development types")
   print(f"Total area: {model.total_area()} hectares")

Step 4: Define Growth Curves
----------------------------

Growth curves describe how forest attributes change with age.

.. code-block:: python

   from ws3.core import Curve

   # Define a volume curve for Douglas-fir on SI 50
   df_volume = Curve(
       x=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
       y=[0, 5, 25, 65, 120, 200, 300, 400, 470, 500, 510],
       name="DF-SI50_volume"
   )

   # Define a volume curve for Spruce on SI 40
   spruce_volume = Curve(
       x=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
       y=[0, 3, 15, 40, 80, 140, 210, 280, 340, 380, 400],
       name="Spruce-SI40_volume"
   )

   model.add_curve("volume", df_volume)
   model.add_curve("volume", spruce_volume)

   print("Added growth curves")

Step 5: Define Actions
----------------------

Actions are management interventions. Each action has a code, description,
and transitions.

.. code-block:: python

   # Define a harvest action
   model.add_action(
       code="HARV",
       descr="Clearcut harvest",
       components=["volume"],
       transitions={
           "DF-SI50": "Bare",
           "Spruce-SI40": "Bare"
       }
   )

   print("Added harvest action")

Step 6: Run a Simulation
-------------------------

Now we can simulate the forest over time.

.. code-block:: python

   # Run simulation for 20 periods (100 years with 5-year periods)
   results = model.run_simulation(horizon=20)

   # Print summary
   print(results.summary())

   # Access specific results
   print(f"Total volume at end: {results.total_volume():.1f} m³")
   print(f"Total harvest: {results.total_harvest():.1f} m³")

Step 7: Inspect the Output
--------------------------

The simulation results contain detailed information about each period:

.. code-block:: python

   # Get area by development type for each period
   area_data = results.area_by_development_type()
   print(area_data.head())

   # Get harvest volumes by period
   harvest_data = results.harvest_by_period()
   print(harvest_data.head())

What's Next?
------------

- :doc:`first_model` — Build a more complete model with optimization
- :doc:`architecture_overview` — Understand how ws3 components fit together
- :doc:`/textbook/ch01_forest_estate_models` — Learn the theory behind wood supply models
- :doc:`/howto/data-preparation` — Prepare real forest inventory data for ws3