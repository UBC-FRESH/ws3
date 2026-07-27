.. _howto-loading-model:

=============================
Loading a Woodstock Model
=============================

This guide shows how to load a Woodstock-formatted model into ws3.

Prerequisites
-------------

* A Woodstock model directory containing standard section files
  (AREAS, YIELDS, ACTIONS, TRANSITIONS)
* ws3 installed

Procedure
---------

**1. Create a ForestModel instance**

.. code-block:: python

   from ws3.forest import ForestModel

   fm = ForestModel(
       model_name="my_model",
       model_path="path/to/model",
       base_year=2020,
       horizon=10,
       period_length=10
   )

**2. Import sections**

.. code-block:: python

   fm.import_areas_section()
   fm.import_yields_section()
   fm.import_actions_section()
   fm.import_transitions_section()

**3. Initialize**

.. code-block:: python

   fm.initialize_areas()
   fm.add_null_action()
   fm.reset_actions()

**4. Verify the load**

.. code-block:: python

   print(f"Development types: {len(fm.dtypes)}")
   print(f"Actions: {list(fm.actions.keys())}")
   print(f"Yield names: {fm.ynames}")

Troubleshooting
---------------

* **Missing section files** — confirm ``model_path`` contains standard
  Woodstock section files and that file naming follows the convention
  ``<model_name>.<section>``.
* **Empty development types** — check that the AREAS section contains
  non-zero area values and that ``area_epsilon`` is appropriate.