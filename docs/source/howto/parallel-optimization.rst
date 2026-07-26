.. _howto-parallel-optimization:

=================
Parallel Optimization
=================

Goal
----

Run multiple optimization scenarios in parallel to compare outcomes:

* Sensitivity analysis across parameter ranges
* Scenario comparison (different objectives, constraints)
* Monte Carlo simulation with randomized inputs

Prerequisites
-------------

* Completed :doc:`running-optimization`
* Familiarity with parallel processing concepts
* A working ws3 installation with sample data

Step-by-Step Instructions
-------------------------

**Step 1: Define Scenario Parameters**

.. code-block:: python

   import itertools

   # Define parameter ranges
   horizon_options = [3, 5, 7]
   flow_min_options = [0.7, 0.8, 0.9]
   flow_max_options = [1.1, 1.2, 1.3]

   # Generate all combinations
   scenarios = list(itertools.product(
       horizon_options,
       flow_min_options,
       flow_max_options
   ))

   print(f"Total scenarios: {len(scenarios)}")

**Step 2: Define Scenario Function**

.. code-block:: python

   from ws3.opt import solve_optimization

   def run_scenario(params):
       horizon, flow_min, flow_max = params

       # Configure constraints
       flow_constraints = [
           {
               'type': 'flow',
               'periods': [0, 1],
               'min_ratio': flow_min,
               'max_ratio': flow_max
           }
       ]

       # Run optimization
       solution = solve_optimization(
           model=model,
           horizon=horizon,
           objective='maximize_volume',
           flow_constraints=flow_constraints
       )

       # Extract results
       total_volume = solution.get_total_volume()
       npv = solution.get_npv()

       return {
           'horizon': horizon,
           'flow_min': flow_min,
           'flow_max': flow_max,
           'total_volume': total_volume,
           'npv': npv
       }

**Step 3: Run Scenarios in Parallel**

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor, as_completed

   results = []

   with ThreadPoolExecutor(max_workers=4) as executor:
       futures = {
           executor.submit(run_scenario, scenario): scenario
           for scenario in scenarios
       }

       for future in as_completed(futures):
           result = future.result()
           results.append(result)
           print(f"Completed: horizon={result['horizon']}, "
                 f"vol={result['total_volume']:.0f} m3")

**Step 4: Analyze Results**

.. code-block:: python

   import pandas as pd

   # Convert to DataFrame
   df = pd.DataFrame(results)

   # Sort by total volume
   df_sorted = df.sort_values('total_volume', ascending=False)

   print(df_sorted.head())

   # Plot results
   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(10, 6))
   ax.bar(range(len(df)), df['total_volume'])
   ax.set_xlabel('Scenario')
   ax.set_ylabel('Total Volume (m3)')
   ax.set_title('Scenario Comparison')
   plt.tight_layout()
   plt.show()

Expected Output
---------------

* Multiple optimization solutions
* Comparison table or DataFrame
* Visual comparison of outcomes

Troubleshooting
---------------

**Issue: Memory errors**

* Reduce number of parallel workers
* Process scenarios in batches
* Clear intermediate results

**Issue: Some scenarios fail**

* Wrap run_scenario in try/except
* Log failed scenarios for investigation
* Skip invalid parameter combinations

**Issue: Results inconsistent**

* Check that model state is reset between runs
* Verify random seeds if using stochastic elements
* Ensure all scenarios use same base model

Next Steps
----------

* :doc:`running-optimization` — Run single optimization
* :doc:`spatial-schedule-allocation` — Allocate harvest spatially
* :doc:`libcbm-callbacks` — Integrate with libCBM for carbon