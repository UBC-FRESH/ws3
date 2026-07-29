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

**2. Create a Problem and add objectives**

.. code-block:: python

   from ws3.opt import Problem
   from ws3.advanced_modeling import MultiObjectiveOptimizer

   problem = Problem(name=\"multi_objective\", sense=1, solver=\"highs\")
   # Add your variables and constraints here...

   optimizer = MultiObjectiveOptimizer(problem)
   optimizer.add_objective(name=\"npv\", weight=0.5, direction=\"maximize\")
   optimizer.add_objective(name=\"even_flow\", weight=0.3, direction=\"minimize_deviation\")

**3. Solve using weighted sum**

.. code-block:: python

   result = optimizer.solve_weighted_sum(weights={\"npv\": 0.5, \"even_flow\": 0.3})

**4. Inspect results**

.. code-block:: python

   print(f\"Method: {result['method']}\")
   print(f\"Weights: {result['weights']}\")
   solution = problem.solution()

Notes
-----

* The :py:class:`ws3.advanced_modeling.MultiObjectiveOptimizer` currently
  supports weighted sum and epsilon-constraint methods.
* For Pareto frontier computation, run multiple optimizations with
  different weight combinations.