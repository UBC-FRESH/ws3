.. _howto-multi-objective:

=============================
Multi-Objective Optimization
=============================

Goal
----

Run multi-objective optimization scenarios with ws3.

Prerequisites
-------------

* Completed :doc:`running-optimization`
* Understanding of multi-objective optimization concepts

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

**Step 2: Create Multi-Objective Problem**

.. code-block:: python

   from ws3.advanced_modeling import MultiObjectiveOptimizer

   optimizer = MultiObjectiveOptimizer(fm)

**Step 3: Define Objectives**

.. code-block:: python

   objectives = [
       {"name": "npv", "weight": 0.5, "direction": "maximize"},
       {"name": "even_flow", "weight": 0.3, "direction": "minimize_deviation"},
       {"name": "carbon", "weight": 0.2, "direction": "maximize"}
   ]

**Step 4: Run Optimization**

.. code-block:: python

   pareto_front = optimizer.optimize(objectives)

**Step 5: Inspect Results**

.. code-block:: python

   print(f"Pareto front size: {len(pareto_front)}")
   for solution in pareto_front:
       print(f"NPV: {solution['npv']}, Even Flow: {solution['even_flow']}, Carbon: {solution['carbon']}")

Expected Output
---------------

* Pareto front with multiple trade-off solutions
* Ability to select preferred solution based on weights

Troubleshooting
---------------

**Issue: No Pareto solutions found**

* Check that objectives are feasible
* Verify constraint ranges are appropriate
* Try different weight combinations

**Issue: Optimization takes too long**

* Reduce number of objectives
* Simplify constraints
* Check model size

Next Steps
----------

* :doc:`running-optimization` — Run single-objective scenarios
* :doc:`spatial-allocation` — Allocate harvest spatially