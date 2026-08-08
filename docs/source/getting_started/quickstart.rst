Quickstart Tutorial
===================

This tutorial gets you up and running with ws3 in under 10 minutes.
You'll load a Woodstock model, run an optimization, and inspect
the output.

Prerequisites
-------------

- ws3 installed (see :doc:`installation`)
- Python 3.10+ available in your terminal
- A Woodstock model directory (see :doc:`/howto/loading-a-woodstock-model` in the
  How-To guides for details on the expected file layout)

Step 1: Import ws3
------------------

Open a Python interpreter or Jupyter notebook and import ws3:

.. code-block:: python

   import ws3
   print(f"ws3 version: {ws3.__version__}")

Step 2: Create a ForestModel
------------------------------

The :py:class:`ws3.forest.ForestModel` class is the central hub for
building a wood supply model. It requires a model name, a path to the
input data directory, and a base year.

.. code-block:: python

   from ws3.forest import ForestModel

   fm = ForestModel(
       model_name="my_model",
       model_path="path/to/model",
       base_year=2020,
       horizon=10,
       period_length=10
   )

Step 3: Import Sections
------------------------

Load the model data from the Woodstock section files:

.. code-block:: python

   fm.import_areas_section()
   fm.import_yields_section()
   fm.import_actions_section()
   fm.import_transitions_section()

Step 4: Initialize
------------------

.. code-block:: python

   fm.initialize_areas()
   fm.add_null_action()
   fm.reset_actions()

Step 5: Verify
--------------

.. code-block:: python

   print(f"Development types: {len(fm.dtypes)}")
   print(f"Actions: {list(fm.actions.keys())}")
   print(f"Yield names: {fm.ynames}")

Step 6: Run Optimization
--------------------------

.. code-block:: python

   from ws3.opt import Problem

   problem = Problem(
       name="base_scenario",
       sense=1,  # SENSE_MAXIMIZE
       solver="highs"
   )
   # Add variables, constraints, objective...
   problem.solve(verbose=True)
   solution = problem.solution()

Step 7: Inspect Results
-------------------------

.. code-block:: python

   print(f"Objective value: {problem.z()}")
   print(f"Variables: {len(solution)}")

   # Get harvest volumes by period
   harvest_data = results.harvest_by_period()
   print(harvest_data.head())

What's Next?
------------

- :doc:`first_model` — Build a more complete model with optimization
- :doc:`architecture_overview` — Understand how ws3 components fit together
- :doc:`/textbook/ch01_forest_estate_models` — Learn the theory behind wood supply models
- :doc:`/howto/loading-a-woodstock-model` — Prepare real forest inventory data for ws3