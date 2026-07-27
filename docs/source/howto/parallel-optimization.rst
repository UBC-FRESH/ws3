.. _howto-parallel-optimization:

=============================
Parallel Optimization
=============================

Goal
----

Run multiple optimization scenarios in parallel to speed up analysis.

Prerequisites
-------------

* Completed :doc:`running-optimization`
* Understanding of parallel computing concepts

Step-by-Step Instructions
-------------------------

**Step 1: Load Model**

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

**Step 2: Define Scenarios**

.. code-block:: python

   scenarios = [
       {"name": "base", "objective": "maximize_npv"},
       {"name": "conservation", "objective": "maximize_carbon"},
       {"name": "timber", "objective": "maximize_volume"}
   ]

**Step 3: Run Parallel Optimization**

.. code-block:: python

   from ws3.forest_helper import PersistentWorkerPool

   pool = PersistentWorkerPool(n_workers=4)

   results = pool.map(
       lambda scenario: run_scenario(fm, scenario),
       scenarios
   )

**Step 4: Collect Results**

.. code-block:: python

   for scenario, result in zip(scenarios, results):
       print(f"Scenario {scenario['name']}: {result}")

Expected Output
---------------

* Multiple optimization scenarios run in parallel
* Results collected and compared

Troubleshooting
---------------

**Issue: Parallel execution fails**

* Check that dill is installed
* Verify that worker processes can access model data
* Check memory usage

**Issue: Results are inconsistent**

* Ensure model is reset between scenarios
* Check that random seeds are set consistently
* Verify that parallel execution doesn't introduce race conditions

Next Steps
----------

* :doc:`running-optimization` — Run single-objective scenarios
* :doc:`multi-objective-optimization` — Run multi-objective scenarios