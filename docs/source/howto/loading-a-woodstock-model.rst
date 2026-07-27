.. _howto-loading-model:

=============================
Loading a Woodstock Model
=============================

Goal
----

Load a Woodstock-formatted model into ws3 and verify it loaded correctly.

Prerequisites
-------------

* A Woodstock model directory with standard section files (AREAS, YIELDS, ACTIONS, TRANSITIONS)
* ws3 installed

Step-by-Step Instructions
-------------------------

**Step 1: Create ForestModel Instance**

.. code-block:: python

   from ws3.forest import ForestModel

   fm = ForestModel(
       model_name="my_model",
       model_path="path/to/model",
       base_year=2020,
       horizon=10,
       period_length=10
   )

**Step 2: Import Sections**

.. code-block:: python

   fm.import_areas_section()
   fm.import_yields_section()
   fm.import_actions_section()
   fm.import_transitions_section()

**Step 3: Initialize**

.. code-block:: python

   fm.initialize_areas()
   fm.add_null_action()
   fm.reset_actions()

**Step 4: Verify**

.. code-block:: python

   print(f"Development types: {len(fm.dtypes)}")
   print(f"Actions: {list(fm.actions.keys())}")
   print(f"Yield names: {fm.ynames}")

Expected Output
---------------

* Model loaded with development types, actions, and yield curves
* No errors during import

Troubleshooting
---------------

**Issue: Missing section files**

* Check that model_path contains standard Woodstock section files
* Verify file naming convention (model_name.are, model_name.yld, etc.)

**Issue: Empty development types**

* Check that AREAS section contains non-zero area values
* Verify area_epsilon threshold is appropriate

Next Steps
----------

* :doc:`defining-growth-curves` — Learn to define custom growth curves
* :doc:`running-optimization` — Run your first optimization scenario