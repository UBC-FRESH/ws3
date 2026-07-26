.. _howto-curve-definition:

=================
Growth Curve Definition
=================

Goal
----

Define custom growth curves for forest development types, including:

* Volume curves by age
* Basal area curves
* Stem density curves
* Multi-component curves (volume, BA, SD)

Prerequisites
-------------

* Completed :doc:`data-preparation`
* Familiarity with growth-and-yield concepts from :doc:`../textbook/ch03_growth_and_yield`
* A working ws3 installation

Step-by-Step Instructions
-------------------------

**Step 1: Prepare Age-Volume Data**

Collect or estimate age-volume pairs for your stand. Data can come from:

* Field measurements
* Yield tables
* Existing growth models
* Literature values

Example data:

.. code-block:: python

   ages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
   volumes = [15.2, 45.8, 95.3, 165.7, 258.4, 368.2, 485.9, 602.1, 715.3, 820.5]

**Step 2: Create GrowthCurve Object**

Use the ws3 GrowthCurve class:

.. code-block:: python

   from ws3.common import GrowthCurve

   curve = GrowthCurve(
       species='SP',
       site_index=50,
       ages=ages,
       volumes=volumes,
       components=['volume']
   )

**Step 3: Add Multiple Components**

For volume, basal area, and stem density:

.. code-block:: python

   ba = [5.2, 12.8, 22.1, 31.5, 39.8, 46.2, 50.8, 54.1, 56.3, 57.8]
   sd = [1850, 1420, 1150, 980, 850, 740, 650, 580, 520, 470]

   curve = GrowthCurve(
       species='SP',
       site_index=50,
       ages=ages,
       volumes=volumes,
       basal_areas=ba,
       stem_densities=sd,
       components=['volume', 'basal_area', 'stem_density']
   )

**Step 4: Register Curve with Model**

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

**Step 5: Validate Curve**

Check that the curve behaves as expected:

.. code-block:: python

   # Get volume at age 45
   vol_45 = curve.get_value(45, component='volume')
   print(f"Volume at age 45: {vol_45:.1f} m3/ha")

   # Plot the curve (requires matplotlib)
   curve.plot(component='volume')

Expected Output
---------------

* GrowthCurve object created and validated
* Curve registered with ForestModel
* Ability to query volume/BA/SD at any age

Troubleshooting
---------------

**Issue: Interpolation errors**

* Ensure ages are in ascending order
* Check for NaN or negative values
* Verify age range covers your planning horizon

**Issue: Curve doesn't match expectations**

* Compare with published yield tables
* Check species and site index codes
* Verify data source reliability

**Issue: Multiple curves conflict**

* Ensure each species/site_index combination has only one curve
* Check for duplicate curve registrations

Next Steps
----------

* :doc:`data-preparation` — Prepare inventory data
* :doc:`action-definition` — Define management actions
* :doc:`running-optimization` — Run optimization scenarios