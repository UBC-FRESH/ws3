.. _howto-multi-objective:

=============================
Multi-Objective Optimization
=============================

This guide shows how to run multi-objective optimization with ws3.

Prerequisites
-------------

* A loaded :ref:`ForestModel <howto-loading-model>`
* Understanding of multi-objective optimization concepts

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

**2. Create a MultiObjectiveOptimizer**

.. code-block:: python

   from ws3.advanced_modeling import MultiObjectiveOptimizer

   optimizer = MultiObjectiveOptimizer(fm)

**3. Define objectives**

.. code-block:: python

   objectives = [
       {"name": "npv", "weight": 0.5, "direction": "maximize"},
       {"name": "even_flow", "weight": 0.3, "direction": "minimize_deviation"},
       {"name": "carbon", "weight": 0.2, "direction": "maximize"}
   ]

**4. Run optimization**

.. code-block:: python

   pareto_front = optimizer.optimize(objectives)

**5. Inspect results**

.. code-block:: python

   print(f"Pareto front size: {len(pareto_front)}")
   for solution in pareto_front:
       print(f"NPV: {solution['npv']}, Even Flow: {solution['even_flow']}, Carbon: {solution['carbon']}")

Notes
-----

* If no Pareto solutions are found, check that objectives are feasible and
  constraint ranges are appropriate.
* Reduce the number of objectives or simplify constraints if optimization
  is too slow.