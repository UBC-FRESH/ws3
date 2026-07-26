.. _howto-model-validation:

=================
Model Validation
=================

Goal
----

Validate your ws3 model against observed data and expectations:

* Compare simulated vs observed inventory
* Check growth curve fit
* Validate optimization results
* Identify and fix model errors

Prerequisites
-------------

* Completed :doc:`running-optimization`
* Familiarity with validation concepts
* A working ws3 installation with sample data

Step-by-Step Instructions
-------------------------

**Step 1: Compare Inventory**

.. code-block:: python

   import pandas as pd

   # Load observed inventory
   observed = pd.read_csv('observed_inventory.csv')

   # Load model inventory
   model_inventory = model.get_development_types()

   # Compare area
   observed_area = observed.groupby('stratum_code')['area_ha'].sum()
   model_area = model_inventory.groupby('code')['area'].sum()

   # Calculate differences
   comparison = pd.DataFrame({
       'observed': observed_area,
       'model': model_area
   })
   comparison['difference'] = comparison['observed'] - comparison['model']
   comparison['pct_diff'] = (comparison['difference'] / comparison['observed']) * 100

   print(comparison)

**Step 2: Validate Growth Curves**

.. code-block:: python

   import matplotlib.pyplot as plt

   # Load observed growth data
   observed_growth = pd.read_csv('observed_growth.csv')

   # Plot observed vs model
   fig, ax = plt.subplots(figsize=(10, 6))

   ax.scatter(observed_growth['age'], observed_growth['volume'],
              label='Observed', alpha=0.7)

   # Plot model curve
   ages = np.arange(10, 101, 1)
   volumes = [curve.get_value(age, 'volume') for age in ages]
   ax.plot(ages, volumes, label='Model', linewidth=2)

   ax.set_xlabel('Age')
   ax.set_ylabel('Volume (m3/ha)')
   ax.set_title('Growth Curve Validation')
   ax.legend()
   plt.tight_layout()
   plt.show()

**Step 3: Check Optimization Results**

.. code-block:: python

   # Run optimization
   solution = solve_optimization(model=model, horizon=5, objective='maximize_volume')

   # Get schedule
   schedule = solution.get_schedule()

   # Check feasibility
   total_area = schedule['area_ha'].sum()
   available_area = model.get_total_area()

   print(f"Total scheduled area: {total_area:.1f} ha")
   print(f"Available area: {available_area:.1f} ha")
   print(f"Utilization: {(total_area / available_area) * 100:.1f}%")

**Step 4: Validate Against Historical Data**

.. code-block:: python

   # Load historical harvest data
   historical = pd.read_csv('historical_harvest.csv')

   # Compare period-by-period
   for period in schedule['period'].unique():
       scheduled = schedule[schedule['period'] == period]['area_ha'].sum()
       historical_vol = historical[historical['period'] == period]['area_ha'].sum()

       if historical_vol > 0:
           ratio = scheduled / historical_vol
           print(f"Period {period}: scheduled={scheduled:.0f}, historical={historical_vol:.0f}, ratio={ratio:.2f}")

**Step 5: Identify and Fix Issues**

.. code-block:: python

   # Common issues to check:
   # 1. Development types with zero area
   zero_area = model.get_development_types(model.area == 0)
   if len(zero_area) > 0:
       print(f"Warning: {len(zero_area)} development types have zero area")

   # 2. Growth curves outside expected range
   for dt_code, curve in model.growth_curves.items():
       max_vol = curve.get_value(100, 'volume')
       if max_vol > 1500:  # Adjust threshold as needed
           print(f"Warning: {dt_code} curve may be too high (max={max_vol:.0f})")

Expected Output
---------------

* Validation report with comparisons
* Graphical validation plots
* Identified issues and fixes

Troubleshooting
---------------

**Issue: Large inventory differences**

* Check data loading and transformation
* Verify stratum code matching
* Ensure area units are consistent

**Issue: Growth curve doesn't fit**

* Check data quality
* Try different curve parameters
* Consider site-specific calibration

**Issue: Optimization infeasible**

* Check constraint feasibility
* Verify area calculations
* Simplify constraints

Next Steps
----------

* :doc:`running-optimization` — Run optimization
* :doc:`data-preparation` — Prepare inventory data
* :doc:`curve-definition` — Define growth curves