Chapter 9: Advanced Topics
==========================

Learning Objectives
-------------------

After reading this chapter, you should be able to:

- Extend ws3 with custom growth functions and area selectors
- Integrate ws3 with other forest planning tools
- Implement parallel computation for large models
- Understand the limitations and boundaries of ws3

Custom Growth Functions
-----------------------

While ws3 provides basic curve functionality, you may need custom growth
functions for specific species or regions.

Subclassing Curve
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ws3.core import Curve

   class CustomCurve(Curve):
       """Custom growth curve with additional methods."""

       def __init__(self, points, growth_rate=0.1, **kwargs):
           super().__init__(points=points, **kwargs)
           self.growth_rate = growth_rate

       def relative_growth_rate(self, age):
           """Calculate relative growth rate at a given age."""
           vol = self.lookup(age)
           if vol == 0:
               return 0
           return (self.lookup(age + 1) - self.lookup(age)) / vol

       def time_to_double(self):
           """Estimate time to double current volume."""
           current_vol = self.lookup(self.xmax)
           target_vol = 2 * current_vol
           for age in self.x:
               if self.lookup(age) >= target_vol:
                   return age
           return None

   # Use the custom curve
   custom = CustomCurve(
       points=[(0, 0), (10, 5), (20, 25), (30, 65), (40, 120), (50, 200)],
       label=\"custom_DF\",
       is_volume=True,
       growth_rate=0.15
   )

   print(f\"Relative growth rate at age 30: {custom.relative_growth_rate(30):.3f}\")
   print(f\"Time to double: {custom.time_to_double()} years\")

Custom Area Selectors
---------------------

The :py:class:`ws3.forest.GreedyAreaSelector` class can be subclassed to
implement custom harvest targeting logic.

.. code-block:: python

   from ws3.forest import GreedyAreaSelector

   class HabitatAreaSelector(GreedyAreaSelector):
       """Select harvest areas based on habitat requirements."""

       def __init__(self, model, min_habitat_area=100):
           super().__init__(model)
           self.min_habitat_area = min_habitat_area

       def operate(self, period, acode, target_area, mask=None,
                   commit_actions=True, verbose=False):
           # Get all development types
           dts = self.model.development_types

           # Sort by habitat value (highest first)
           sorted_dts = sorted(
               dts,
               key=lambda dt: self.get_habitat_value(dt),
               reverse=True
           )

           # Harvest from highest habitat value first
           harvested = 0
           for dt in sorted_dts:
               if harvested >= target_area:
                   break
               available = self.model.get_development_type_area(dt)
               harvest = min(available, target_area - harvested)
               self.model.operate(period, acode, harvest, dt)
               harvested += harvest

           return harvested

       def get_habitat_value(self, dt_code):
           """Calculate habitat value for a development type."""
           # Example: older stands have higher habitat value
           age = self.model.get_development_type_age(dt_code)
           return age / 100.0

   # Use the custom selector
   selector = HabitatAreaSelector(model, min_habitat_area=50)
   selector.operate(period=0, acode="HARV", target_area=100)

Integrating with Other Tools
----------------------------

ws3 can be integrated with other forest planning tools:

.. mermaid::

   graph LR
     WS3["ws3<br/>Wood supply model"] --> OUTPUT["Schedule output"]
     OUTPUT --> GIS["GIS software<br/>(QGIS, ArcGIS)"]
     GIS --> MAP["Harvest maps"]
     WS3 --> FIN["Financial software<br/>(Excel, Python)"]
     FIN --> REPORT["Financial reports"]

Exporting Results
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd

   # Export simulation results to CSV
   # Simulation results are accessed via dtype.area(period) and
   # yield curve lookups for each period.
   # Example:
   results_rows = []
   for period in model.periods:
       for dtype_key, dtype in model.dtypes.items():
           area = dtype.area(period)
           age = dtype.age(period) if hasattr(dtype, 'age') else period * model.period_length
           results_rows.append({
               "period": period,
               "dtype": str(dtype_key),
               "area": area,
               "age": age
           })
   df = pd.DataFrame(results_rows)
   df.to_csv("simulation_results.csv", index=False)

   # Export optimization solution
   solution = prob.solution()
   sol_df = pd.DataFrame(list(solution.items()), columns=["variable", "value"])
   sol_df.to_csv("optimization_solution.csv", index=False)

Importing External Data
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load external growth curves from a CSV file
   import pandas as pd

   curve_data = pd.read_csv("growth_curves.csv")
   for _, row in curve_data.iterrows():
       ages = [int(a) for a in row["ages"].split(",")]
       volumes = [float(v) for v in row["volumes"].split(",")]
       curve = Curve(
           label=row["name"],
           points=list(zip(ages, volumes))
       )
       model.register_curve(curve)

Parallel Computation
--------------------

For large models, ws3 provides parallel computation capabilities:

.. code-block:: python

   from ws3.forest_helper import PersistentWorkerPool

   # Create a worker pool
   pool = PersistentWorkerPool(n_workers=4)

   # Define work items
   work_items = [(i, model) for i in range(100)]

   # Process in parallel
   results = pool.map(simulate_period, work_items)

   # Shutdown pool
   pool.shutdown()

Limitations and Boundaries
--------------------------

ws3 has known limitations:

1. **Aspatial default**: Spatial allocation is optional and less mature
2. **No stochastic optimization**: Only deterministic optimization
3. **Limited disturbance modeling**: Fire, insects require custom code
4. **Single objective**: Multi-objective optimization requires extension
5. **No real-time updating**: Models are static, not dynamic

When ws3 May Not Be the Right Tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider alternative tools if you need:

- **Real-time decision support**: Use a database-driven system
- **Complex spatial constraints**: Use a GIS-based optimizer
- **Stochastic programming**: Use a dedicated stochastic optimizer
- **Multi-agent simulation**: Use an agent-based modeling framework
- **Climate change integration**: Use a dynamic vegetation model

Extending ws3: Best Practices
-----------------------------

1. **Subclass, don't modify**: Extend base classes rather than changing ws3 code
2. **Keep it modular**: Each extension should have a single responsibility
3. **Write tests**: Test your extensions thoroughly
4. **Document assumptions**: Document any assumptions about growth, prices, etc.
5. **Validate against data**: Compare model output to observed data

Exercises
---------

**Exercise 1 (Easy)**: Create a custom growth curve that uses a
logistic function instead of tabulated values.

**Exercise 2 (Medium)**: Implement a custom area selector that avoids
harvesting within 100 meters of water bodies.

**Exercise 3 (Hard)**: Extend ws3 to support multi-objective optimization
(Pareto front) by modifying the Problem class.

Further Reading
---------------

- :doc:`ch07_financial_analysis` — Financial analysis
- :doc:`ch08_uncertainty_and_risk` — Uncertainty and risk
- :doc:`/guides/extending-ws3` — Detailed extension guide
- :doc:`reference/contracts/index` — Data contracts and module boundaries