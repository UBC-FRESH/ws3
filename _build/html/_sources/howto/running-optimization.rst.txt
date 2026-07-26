.. _howto-running-optimization:

=================
Running Optimization
=================

Goal
----

Run your first optimization scenario with ws3:

* Configure optimization parameters
* Solve the harvest scheduling problem
* Interpret the results
* Export results for analysis

Prerequisites
-------------

* Completed :doc:`data-preparation`, :doc:`curve-definition`, and :doc:`action-definition`
* Familiarity with optimization concepts from :doc:`../textbook/ch05_optimization`
* A working ws3 installation with sample data

Step-by-Step Instructions
-------------------------

**Step 1: Load Your Model**

.. code-block:: python

   from ws3.forest import ForestModel
   import pandas as pd

   # Load model from previous steps
   model = ForestModel()

   # Add development types, curves, and actions
   # (see previous how-to guides)

**Step 2: Configure Optimization Parameters**

.. code-block:: python

   # Define planning horizon
   horizon = 5

   # Define time periods
   periods = list(range(horizon))

   # Set optimization objective
   objective = 'maximize_volume'  # or 'maximize_npv', 'area_control'

**Step 3: Define Constraints**

.. code-block:: python

   # Flow constraints
   flow_constraints = [
       {
           'type': 'flow',
           'periods': [0, 1],
           'min_ratio': 0.8,
           'max_ratio': 1.2
       }
   ]

   # Area constraints
   area_constraints = [
       {
           'type': 'area',
           'period': 0,
           'min_area': 50.0,
           'max_area': 200.0
       }
   ]

**Step 4: Run Optimization**

.. code-block:: python

   from ws3.opt import solve_optimization

   # Solve the optimization problem
   solution = solve_optimization(
       model=model,
       horizon=horizon,
       objective=objective,
       flow_constraints=flow_constraints,
       area_constraints=area_constraints
   )

**Step 5: Inspect Results**

.. code-block:: python

   # Get harvest schedule
   schedule = solution.get_schedule()

   # Print results
   print(schedule.head())

   # Get summary statistics
   summary = solution.get_summary()
   print(summary)

**Step 6: Export Results**

.. code-block:: python

   # Export to CSV
   schedule.to_csv('harvest_schedule.csv', index=False)

   # Export to Excel
   schedule.to_excel('harvest_schedule.xlsx', index=False)

Expected Output
---------------

* Optimization solution object
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

* :doc:`parallel-optimization` — Run multiple scenarios in parallel
* :doc:`spatial-schedule-allocation` — Allocate harvest spatially
* :doc:`libcbm-callbacks` — Integrate with libCBM for carbon