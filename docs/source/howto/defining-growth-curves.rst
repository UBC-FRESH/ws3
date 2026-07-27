.. _howto-defining-curves:

=============================
Defining Growth Curves
=============================

This guide shows how to define custom growth curves for forest development
types.

Prerequisites
-------------

* A loaded :ref:`ForestModel <howto-loading-model>`
* Age-volume data as pairs ``(age, volume)``

Procedure
---------

**1. Prepare age-volume data**

.. code-block:: python

   points = [
       (10, 15.2), (20, 45.8), (30, 95.3), (40, 165.7),
       (50, 258.4), (60, 368.2), (70, 485.9), (80, 602.1),
       (90, 715.3), (100, 820.5)
   ]

**2. Create a Curve object**

.. code-block:: python

   from ws3.core import Curve

   curve = Curve(
       label="SP_SI50_Volume",
       is_volume=True,
       points=points,
       period_length=10
   )

**3. Register with the model**

.. code-block:: python

   fm.register_curve(curve)

**4. Query the curve**

.. code-block:: python

   vol_45 = curve.lookup(45)
   print(f"Volume at age 45: {vol_45:.1f} m3/ha")

Notes
-----

* Ages must be in ascending order.
* ``Curve.lookup()`` performs linear interpolation between data points.