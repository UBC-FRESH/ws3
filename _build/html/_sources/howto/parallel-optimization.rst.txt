.. _howto-parallel-optimization:

=============================
Parallel Optimization
=============================

This guide shows how to run multiple optimization scenarios in parallel.

Prerequisites
-------------

* A loaded :ref:`ForestModel <howto-loading-model>`
* Understanding of parallel computing concepts

Procedure
---------

**1. Load the model**

.. code-block:: python

   from ws3.forest import ForestModel

   fm = ForestModel(
       model_name="my_model",
       model_path="path/to/model",
       base_year=2020,
       horizon=10,
       period_length=10
   )
   fm.import_areas_section()
   fm.import_yields_section()
   fm.import_actions_section()
   fm.import_transitions_section()
   fm.initialize_areas()
   fm.add_null_action()
   fm.reset_actions()

**2. Define scenarios**

.. code-block:: python

   scenarios = [
       {"name": "base", "objective": "maximize_npv"},
       {"name": "conservation", "objective": "maximize_carbon"},
       {"name": "timber", "objective": "maximize_volume"}
   ]

**3. Run parallel optimization**

.. code-block:: python

   from ws3.forest_helper import PersistentWorkerPool

   pool = PersistentWorkerPool(n_workers=4)

   results = pool.map(
       lambda scenario: run_scenario(fm, scenario),
       scenarios
   )

**4. Collect results**

.. code-block:: python

   for scenario, result in zip(scenarios, results):
       print(f"Scenario {scenario['name']}: {result}")

Notes
-----

* Install ``dill`` for worker serialization.
* Reset the model between scenarios to avoid state leakage.
* Set random seeds consistently if reproducibility matters.