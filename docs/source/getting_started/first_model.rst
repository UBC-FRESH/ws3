Your First Model
================

This walkthrough builds a complete wood supply model with optimization.
You'll learn how to define a realistic scenario, set up an optimization
problem, and extract the optimal harvest schedule.

Scenario
--------

You manage a 2,000-hectare forest with:

- 800 hectares of Douglas-fir (Site Index 50), ages 20-80
- 600 hectares of Spruce (Site Index 40), ages 30-70
- 400 hectares of Cedar (Site Index 45), ages 40-90
- 200 hectares of mixed broadleaf, ages 10-60

Your objective: maximize net present value (NPV) over a 100-year horizon
(20 periods of 5 years each), subject to:

- Maximum annual harvest: 200 hectares
- Minimum ending inventory: 400 hectares of each species
- Sustainable yield: harvest cannot exceed 50% of available volume per period

Building the Model
------------------

Step 1: Create the Forest Model

.. code-block:: python

   from ws3.forest import ForestModel
   from ws3.core import Curve

   model = ForestModel()

Step 2: Add Development Types

.. code-block:: python

   # Douglas-fir stands at different ages
   for age in [20, 30, 40, 50, 60, 70, 80]:
       model.add_development_type(
           code=f"DF-SI50-A{age}",
           area=800.0 / 7,  # Distribute evenly
           age=age,
           species="Pseudotsuga menziesii",
           site_index=50
       )

   # Spruce stands at different ages
   for age in [30, 40, 50, 60, 70]:
       model.add_development_type(
           code=f"SP-SI40-A{age}",
           area=600.0 / 5,
           age=age,
           species="Picea sitchensis",
           site_index=40
       )

   # Cedar stands at different ages
   for age in [40, 50, 60, 70, 80, 90]:
       model.add_development_type(
           code=f"CE-SI45-A{age}",
           area=400.0 / 6,
           age=age,
           species="Thuja plicata",
           site_index=45
       )

   # Mixed broadleaf
   for age in [10, 20, 30, 40, 50, 60]:
       model.add_development_type(
           code=f"ML-A{age}",
           area=200.0 / 6,
           age=age,
           species="Mixed",
           site_index=0
       )

Step 3: Define Growth Curves

.. code-block:: python

   # Douglas-fir volume curve
   df_vol = Curve(
       x=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
       y=[0, 5, 25, 65, 120, 200, 300, 400, 470, 500, 510],
       name="DF_volume"
   )
   model.add_curve("volume", df_vol)

   # Spruce volume curve
   sp_vol = Curve(
       x=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
       y=[0, 3, 15, 40, 80, 140, 210, 280, 340, 380, 400],
       name="SP_volume"
   )
   model.add_curve("volume", sp_vol)

   # Cedar volume curve
   ce_vol = Curve(
       x=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
       y=[0, 2, 10, 30, 60, 110, 180, 260, 340, 400, 440],
       name="CE_volume"
   )
   model.add_curve("volume", ce_vol)

Step 4: Define Actions

.. code-block:: python

   # Harvest action
   model.add_action(
       code="HARV",
       descr="Clearcut harvest",
       components=["volume"],
       transitions={
           "DF-SI50-A20": "Bare", "DF-SI50-A30": "Bare",
           "DF-SI50-A40": "Bare", "DF-SI50-A50": "Bare",
           "DF-SI50-A60": "Bare", "DF-SI50-A70": "Bare",
           "DF-SI50-A80": "Bare",
           "SP-SI40-A30": "Bare", "SP-SI40-A40": "Bare",
           "SP-SI40-A50": "Bare", "SP-SI40-A60": "Bare",
           "SP-SI40-A70": "Bare",
           "CE-SI45-A40": "Bare", "CE-SI45-A50": "Bare",
           "CE-SI45-A60": "Bare", "CE-SI45-A70": "Bare",
           "CE-SI45-A80": "Bare", "CE-SI45-A90": "Bare",
           "ML-A10": "Bare", "ML-A20": "Bare",
           "ML-A30": "Bare", "ML-A40": "Bare",
           "ML-A50": "Bare", "ML-A60": "Bare"
       }
   )

   # Bare site (post-harvest)
   model.add_development_type(
       code="Bare",
       area=0.0,
       age=0,
       species="",
       site_index=0
   )

Step 5: Set Up Optimization

.. code-block:: python

   from ws3.opt import Problem

   # Create optimization problem
   prob = Problem()

   # Decision variables: harvest area for each development type in each period
   harvest_vars = {}
   for dt_code in model.development_types:
       for period in range(20):
           var_name = f"harv_{dt_code}_p{period}"
           harvest_vars[(dt_code, period)] = prob.add_variable(
               name=var_name,
               vtype="continuous",
               lb=0,
               ub=model.get_development_type_area(dt_code)
           )

   # Objective: maximize NPV
   # Simplified: volume harvested * price ($50/m³) discounted at 5%
   npv = 0
   discount_rate = 0.05
   for (dt_code, period), var in harvest_vars.items():
       # Get volume at current age
       age = model.get_development_type_age(dt_code)
       volume_per_ha = model.get_curve_value("volume", age)
       npv += var * volume_per_ha * 50 * (1 + discount_rate) ** (-period * 5)

   prob.set_objective(npv, sense="maximize")

   # Constraint 1: Maximum annual harvest area
   for period in range(20):
       period_harvest = sum(
           harvest_vars[(dt_code, period)]
           for dt_code in model.development_types
       )
       prob.add_constraint(
           name=f"max_harvest_p{period}",
           expr=period_harvest <= 200
       )

   # Constraint 2: Sustainable yield (volume harvested <= 50% of available)
   for period in range(20):
       for dt_code in model.development_types:
           age = model.get_development_type_age(dt_code)
           available_volume = (
               model.get_development_type_area(dt_code) *
               model.get_curve_value("volume", age)
           )
           prob.add_constraint(
               name=f"sustain_{dt_code}_p{period}",
               expr=harvest_vars[(dt_code, period)] *
                    model.get_curve_value("volume", age) <=
                    0.5 * available_volume
           )

Step 6: Solve and Inspect Results

.. code-block:: python

   # Solve with HiGHS (default solver)
   prob.solve(solver="highs")

   # Get solution
   solution = prob.get_solution()

   # Print harvest schedule
   print("Optimal Harvest Schedule:")
   print("-" * 60)
   for period in range(20):
       total_harvest = 0
       for dt_code in model.development_types:
           var_name = f"harv_{dt_code}_p{period}"
           area = solution.get(var_name, 0)
           if area > 0:
               total_harvest += area
               print(f"  Period {period}: {dt_code} = {area:.1f} ha")
       print(f"  Total: {total_harvest:.1f} ha")
       print()

   print(f"Total NPV: ${solution.objective_value:,.0f}")

What You've Learned
-------------------

In this walkthrough, you:

1. Created a :py:class:`ws3.forest.ForestModel` with multiple development types
2. Defined growth curves for different species
3. Set up management actions with transitions
4. Built an optimization problem with decision variables and constraints
5. Solved the problem and inspected the optimal harvest schedule

Next Steps
----------

- :doc:`architecture_overview` — Understand how ws3 components fit together
- :doc:`/howto/running-optimization` — Deep dive into optimization setup
- :doc:`/howto/parallel-optimization` — Speed up large models with parallel processing
- :doc:`/textbook/ch02_forest_inventory` — Learn about forest inventory data