.. _howto-defining-curves:

=============================
Defining Growth Curves
=============================

Goal
----

Define custom growth curves for forest development types.

Prerequisites
-------------

* Completed :doc:`loading-a-woodstock-model`
* Understanding of growth-and-yield concepts

Step-by-Step Instructions
-------------------------

**Step 1: Prepare Age-Volume Data**

.. code-block:: python

   # Age-volume pairs as list of tuples
   points = [
       (10, 15.2), (20, 45.8), (30, 95.3), (40, 165.7),
       (50, 258.4), (60, 368.2), (70, 485.9), (80, 602.1),
       (90, 715.3), (100, 820.5)
   ]

**Step 2: Create Curve Object**

.. code-block:: python

   from ws3.core import Curve

   curve = Curve(
       label="SP_SI50_Volume",
       is_volume=True,
       points=points,
       period_length=10
   )

**Step 3: Register with Model**

.. code-block:: python

   fm.register_curve(curve)

**Step 4: Query Curve**

.. code-block:: python

   # Get volume at age 45
   vol_45 = curve.lookup(45)
   print(f"Volume at age 45: {vol_45:.1f} m3/ha")

Expected Output
---------------

* Curve object created and registered
* Ability to query volume at any age

Troubleshooting
---------------

**Issue: Interpolation errors**

* Ensure ages are in ascending order
* Check for NaN or negative values

**Issue: Curve doesn't match expectations**

* Compare with published yield tables
* Verify species and site index codes

Next Steps
----------

* :doc:`loading-a-woodstock-model` — Load a Woodstock model
* :doc:`running-optimization` — Run optimization scenarios