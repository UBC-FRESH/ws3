.. _howto-running-optimization:

=============================
Running Optimization
=============================

Goal
----

Run your first optimization scenario with ws3.

Prerequisites
-------------

* Completed :doc:`loading-a-woodstock-model` and :doc:`defining-growth-curves`
* Understanding of optimization concepts

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

**Step 2: Create Problem**

.. code-block:: python

   from ws3.opt import Problem

   problem = Problem(
       name="base_scenario",
       sense=1,  # SENSE_MAXIMIZE
       solver="highs"
   )

**Step 3: Define Objective**

.. code-block:: python

   # Define objective coefficients
   # Example: maximize volume harvest
   coeffs = {var_name: 1.0 for var_name in problem.var_names()}
   problem.z(coeffs)

**Step 4: Add Constraints**

.. code-block:: python

   # Example: even-flow constraint
   # Sum of harvest in period 0 <= 1.2 * Sum of harvest in period 1
   problem.add_constraint(
       name="even_flow",
       coeffs={var_name: 1.0 if "period_0" in var_name else -1.2 for var_name in problem.var_names()},
       sense="<=",
       rhs=0.0
   )

**Step 5: Solve**

.. code-block:: python

   problem.solve(verbose=True)

**Step 6: Inspect Results**

.. code-block:: python

   solution = problem.solution()
   print(f"Objective value: {problem.z()}")
   print(f"Variables: {len(solution)}")

Expected Output
---------------

* Optimization solution found
* Harvest schedule with period-by-period prescriptions
* Summary statistics (total volume, NPV, etc.)

Troubleshooting
---------------

**Issue: Solver fails to converge**

* Check that all development types and actions are defined
* Verify constraint ranges are feasible
* Try simpler objective function first

**Issue: No harvest in schedule**

* Check that actions are applicable to development types
* Verify area constraints allow harvest
* Ensure growth curves are defined

**Issue: Solver takes too long**

* Reduce planning horizon
* Simplify constraints
* Check model size (number of development types)

Next Steps
----------

* :doc:`spatial-allocation` — Allocate harvest spatially
* :doc:`multi-objective-optimization` — Run multi-objective scenarios