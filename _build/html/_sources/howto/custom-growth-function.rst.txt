.. _howto-custom-growth-function:

=================
Custom Growth Function
=================

Goal
----

Implement custom growth functions beyond standard curves:

* User-defined growth models
* Machine learning predictions
* Site-specific calibration
* Dynamic growth adjustment

Prerequisites
-------------

* Completed :doc:`curve-definition`
* Familiarity with growth modeling concepts
* A working ws3 installation

Step-by-Step Instructions
-------------------------

**Step 1: Define Custom Growth Function**

.. code-block:: python

   import numpy as np

   def custom_growth_function(age, site_index, volume_at_age_10=20.0):
       """Custom growth function using logistic model."""

       # Logistic growth parameters
       k = 0.08  # growth rate
       v_max = 1200.0  # asymptotic volume
       t0 = 15.0  # inflection point

       # Logistic function
       volume = v_max / (1 + np.exp(-k * (age - t0)))

       # Scale by site index
       si_factor = site_index / 50.0
       volume *= si_factor

       return volume

**Step 2: Create GrowthCurve with Custom Function**

.. code-block:: python

   from ws3.common import GrowthCurve

   ages = np.arange(10, 101, 5)

   # Generate volumes using custom function
   volumes = [
       custom_growth_function(age, 50) for age in ages
   ]

   curve = GrowthCurve(
       species='SP',
       site_index=50,
       ages=ages,
       volumes=volumes,
       components=['volume']
   )

**Step 3: Register with Model**

.. code-block:: python

   from ws3.forest import ForestModel

   model = ForestModel()

   model.add_development_type(
       code='DT001',
       species='SP',
       site_index=50,
       age=10,
       area=100.0
   )

   model.add_growth_curve(curve)

**Step 4: Test Custom Function**

.. code-block:: python

   # Compare custom vs standard
   custom_vol = custom_growth_function(50, 50)
   standard_vol = curve.get_value(50, component='volume')

   print(f"Custom: {custom_vol:.1f} m3/ha")
   print(f"Standard: {standard_vol:.1f} m3/ha")

**Step 5: Use in Optimization**

.. code-block:: python

   from ws3.opt import solve_optimization

   solution = solve_optimization(
       model=model,
       horizon=5,
       objective='maximize_volume'
   )

Expected Output
---------------

* Custom growth function implemented
* GrowthCurve created with custom values
* Optimization uses custom growth

Troubleshooting
---------------

**Issue: Growth values unrealistic**

* Check function parameters
* Compare with published yield tables
* Verify site index scaling

**Issue: Function doesn't converge**

* Ensure function is smooth and differentiable
* Check for discontinuities
* Try simpler function first

**Issue: Performance issues**

* Profile function execution time
* Consider pre-computing values
* Use vectorized operations

Next Steps
----------

* :doc:`curve-definition` — Define standard growth curves
* :doc:`data-preparation` — Prepare inventory data
* :doc:`running-optimization` — Run optimization scenarios