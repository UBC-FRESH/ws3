Chapter 3: Growth and Yield
===========================

Learning Objectives
-------------------

After reading this chapter, you should be able to:

- Explain how growth curves are fitted to forest inventory data
- Use the :py:class:`ws3.core.Curve` class to define and manipulate growth curves
- Interpolate between curve points
- Perform arithmetic operations on curves (add, subtract, multiply, divide)
- Validate growth curves for biological plausibility

What Are Growth Curves?
-----------------------

**Growth curves** describe how forest attributes change over time. They
are the engine of wood supply models — without them, the model would
just track area without understanding how the forest grows.

Common attributes tracked by growth curves:

- **Volume**: Total merchantable volume (m³/ha)
- **Basal area**: Cross-sectional area of tree trunks (m²/ha)
- **Height**: Dominant height (m)
- **Biomass**: Total above-ground biomass (tonnes/ha)
- **Value**: Dollar value per unit volume ($/m³)

Growth curves typically follow a sigmoidal (S-shaped) pattern:

.. mermaid::

   graph TD
     YOUNG["Young stands<br/>Slow growth"] --> MATURE["Mature stands<br/>Rapid growth"] --> OLD["Old stands<br/>Asymptoting"]

The curve starts slowly (young trees growing), accelerates (rapid
middle-age growth), and then plateaus (trees reach physiological limits).

The Curve Class
---------------

ws3's :py:class:`ws3.core.Curve` class provides a flexible interface
for defining and manipulating growth curves.

Defining a Curve
~~~~~~~~~~~~~~~~

.. code-block:: python

   from ws3.core import Curve

   # Define a volume curve for Douglas-fir on Site Index 50
   volume_curve = Curve(
       x=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
       y=[0, 5, 25, 65, 120, 200, 300, 400, 470, 500, 510],
       name="DF-SI50_volume"
   )

The ``x`` values are ages (years), and the ``y`` values are attribute
values (e.g., volume in m³/ha).

Interpolation
~~~~~~~~~~~~~

Curves support interpolation between defined points:

.. code-block:: python

   # Get volume at age 25 (interpolated between ages 20 and 30)
   vol_25 = volume_curve(25)
   print(f"Volume at age 25: {vol_25:.1f} m³/ha")

   # Get volume at age 55
   vol_55 = volume_curve(55)
   print(f"Volume at age 55: {vol_55:.1f} m³/ha")

Arithmetic Operations
~~~~~~~~~~~~~~~~~~~~~

Curves support arithmetic operations, which is useful for combining
curves or calculating differences:

.. code-block:: python

   # Create a value curve (volume * price)
   price_curve = Curve(
       x=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
       y=[0, 250, 1250, 3250, 6000, 10000, 15000, 20000, 23500, 25000, 25500],
       name="DF-SI50_value"
   )

   # Net value = value curve - harvesting cost curve
   cost_curve = Curve(
       x=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
       y=[0, 100, 400, 900, 1600, 2500, 3600, 4900, 6400, 8100, 10000],
       name="harvest_cost"
   )

   net_value = price_curve - cost_curve
   print(f"Net value at age 50: ${net_value(50):,.0f}")

Curve Algebra
~~~~~~~~~~~~~

The Curve class supports all basic arithmetic operations:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Operator
     - Description
   * - ``curve1 + curve2``
     - Element-wise addition
   * - ``curve1 - curve2``
     - Element-wise subtraction
   * - ``curve1 * curve2``
     - Element-wise multiplication
   * - ``curve1 / curve2``
     - Element-wise division
   * - ``curve * scalar``
     - Scale all values by a constant
   * - ``curve / scalar``
     - Divide all values by a constant

Fitting Growth Curves
---------------------

Growth curves are typically fitted to field data using statistical methods.
Common approaches include:

1. **Polynomial regression**: Simple but can produce unrealistic curves
2. **Logistic growth**: Captures sigmoidal shape well
3. **Richards function**: Flexible sigmoidal model
4. **Provincial yield tables**: Government-published curves for common species

Using Provincial Yield Tables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In British Columbia, the Ministry of Forests publishes growth-and-yield
tables for common species. These tables provide volume estimates for
different species, site indices, and ages.

.. code-block:: python

   # Example: Load a provincial yield table
   # (This is a simplified example — actual tables are more complex)

   yield_data = {
       "species": "Pseudotsuga menziesii",
       "site_index": 50,
       "ages": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
       "volumes": [0, 5, 25, 65, 120, 200, 300, 400, 470, 500, 510]
   }

   curve = Curve(
       x=yield_data["ages"],
       y=yield_data["volumes"],
       name=f"{yield_data['species']}-SI{yield_data['site_index']}_volume"
   )

Validating Growth Curves
------------------------

Before using a growth curve in a model, validate it:

.. code-block:: python

   def validate_curve(curve, min_age=0, max_age=500):
       """Validate a growth curve for biological plausibility."""

       # Check that x values are increasing
       x = curve.x
       if not all(x[i] < x[i+1] for i in range(len(x)-1)):
           raise ValueError("x values must be strictly increasing")

       # Check that y values are non-negative
       if any(y < 0 for y in curve.y):
           raise ValueError("y values must be non-negative")

       # Check that curve starts at zero (or near zero)
       if curve.y[0] > 10:
           raise ValueError("Curve should start near zero volume")

       # Check for reasonable maximum values
       max_vol = max(curve.y)
       if max_vol > 10000:  # 10,000 m³/ha is unrealistically high
           raise ValueError(f"Maximum volume {max_vol} m³/ha is unrealistic")

       # Check that curve eventually plateaus (optional)
       if len(curve.y) > 10:
           recent_growth = curve.y[-1] - curve.y[-2]
           if recent_growth > 100:  # More than 100 m³/ha growth in one period
               print("Warning: Curve may not be plateauing")

   validate_curve(volume_curve)

Common Growth Curve Shapes
--------------------------

Different species have characteristic growth curve shapes:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Species
     - Growth Pattern
   * - Douglas-fir
     - Fast initial growth, peaks around age 80-100, then declines
   * - Western red cedar
     - Slow initial growth, continues increasing slowly past age 200
   * - Sitka spruce
     - Very rapid growth, peaks early (age 40-60), then declines sharply
   * - Lodgepole pine
     - Moderate growth, peaks around age 60-80, relatively flat plateau

Exercises
---------

**Exercise 1 (Easy)**: Create a volume curve for Sitka spruce on Site
Index 45 and interpolate to find the volume at age 35.

**Exercise 2 (Medium)**: Write a function that takes two curves (volume
and price) and returns a net value curve (price * volume - cost).

**Exercise 3 (Hard)**: Fit a logistic growth curve to the following
data points using scipy.optimize.curve_fit:

.. code-block:: python

   ages = [10, 20, 30, 40, 50, 60, 70, 80]
   volumes = [10, 40, 90, 150, 210, 260, 300, 330]

Further Reading
---------------

- :doc:`ch02_forest_inventory` — Preparing inventory data
- :doc:`/howto/curve-definition` — Detailed curve definition guide
- :doc:`/reference/modules/core` — Curve and Interpolator API reference