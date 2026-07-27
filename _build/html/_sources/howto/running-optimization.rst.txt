.. _howto-running-optimization:

=============================
Running Optimization
=============================

This guide shows how to run an optimization scenario with ws3.

Prerequisites
-------------

* A loaded :ref:`ForestModel <howto-loading-model>`
* Understanding of linear programming concepts

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

**2. Create a Problem**

.. code-block:: python

   from ws3.opt import Problem

   problem = Problem(
       name="base_scenario",
       sense=1,  # SENSE_MAXIMIZE
       solver="highs"
   )

**3. Define the objective**

.. code-block:: python

   coeffs = {var_name: 1.0 for var_name in problem.var_names()}
   problem.z(coeffs)

**4. Add constraints**

.. code-block:: python

   problem.add_constraint(
       name="even_flow",
       coeffs={var_name: 1.0 if "period_0" in var_name else -1.2
               for var_name in problem.var_names()},
       sense="<=",
       rhs=0.0
   )

**5. Solve**

.. code-block:: python

   problem.solve(verbose=True)

**6. Inspect results**

.. code-block:: python

   solution = problem.solution()
   print(f"Objective value: {problem.z()}")
   print(f"Variables: {len(solution)}")

Troubleshooting
---------------

* **Solver fails to converge** — verify all development types and actions are
  defined and constraint ranges are feasible.
* **No harvest in schedule** — check that actions are applicable to
  development types and that area constraints allow harvest.
* **Solver takes too long** — reduce the planning horizon or simplify
  constraints.