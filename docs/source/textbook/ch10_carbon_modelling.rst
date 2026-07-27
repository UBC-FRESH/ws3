Chapter 10: Carbon Modelling
============================

Learning Objectives
-------------------

After reading this chapter, you should be able to:

- Explain the role of carbon accounting in modern forest management
- Calculate standing carbon stocks from forest inventory data
- Model carbon fluxes from harvest and decomposition
- Integrate carbon accounting into wood supply optimization
- Understand the policy context (BC Emissions Trading System, carbon credits)

Why Carbon Matters in Forest Management
---------------------------------------

Carbon accounting has become a central concern in forest management for
several reasons:

1. **Climate policy**: British Columbia's Emissions Trading System (ETS)
   requires forest operators to report and offset emissions
2. **Carbon credits**: Forest carbon sequestration can generate revenue
   through carbon markets
3. **Sustainability reporting**: Stakeholders demand carbon-transparent
   management plans
4. **Regulatory compliance**: The BC Forest and Range Practices Act
   requires carbon considerations in planning

Carbon in the Forest System
---------------------------

Forest carbon exists in multiple pools:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Pool
     - Location
     - Description
   * - Above-ground biomass
     - Living trees (bole, branches, foliage)
     - Largest pool in most stands
   * - Below-ground biomass
     - Roots
     - Typically 15-25% of above-ground
   * - Dead wood
     - Standing dead trees, coarse woody debris
     - Decomposes slowly (decades)
   * - Litter
     - Leaf litter, fine woody debris
     - Decomposes moderately (years)
   * - Soil organic matter
     - Humus layer
     - Largest but slowest-changing pool

.. mermaid::

   graph TD
     ATMOSPHERE["Atmosphere<br/>(CO₂)"] --> SEQUESTER["Photosynthesis<br/>(carbon uptake)"]
     SEQUESTER --> TREES["Living trees"]
     TREES --> ABOVE["Above-ground<br/>biomass"]
     TREES --> BELOW["Below-ground<br/>biomass"]
     TREES --> HARVEST["Harvest"]
     HARVEST --> PRODUCTS["Wood products<br/>(long-term storage)"]
     HARVEST --> DECOMP["Decomposition<br/>(CO₂ release)"]
     DECOMP --> ATMOSPHERE
     TREES --> DEAD["Dead wood"]
     DEAD --> DECOMP

Carbon Calculation Basics
-------------------------

Carbon is approximately 50% of dry biomass. The conversion from biomass
to carbon uses a simple factor:

.. math::

   \\text{Carbon (tonnes)} = \\text{Biomass (tonnes)} \\times 0.5

Biomass can be estimated from volume using species-specific factors:

.. math::

   \\text{Biomass (tonnes/ha)} = \\text{Volume (m³/ha)} \\times \\text{Bulk Density (tonnes/m³)} \\times (1 + \\text{Root-Shoot Ratio})

Using ws3 for Carbon Calculations
----------------------------------

ws3 doesn't have built-in carbon functions, but you can calculate carbon
stocks from the model output:

.. code-block:: python

   from ws3.forest import ForestModel
   from ws3.core import Curve

   # Define biomass curves (tonnes/ha of carbon)
   df_carbon = Curve(
       label="DF-SI50_carbon",
       points=[(0, 0), (10, 1), (20, 5), (30, 12), (40, 22),
               (50, 35), (60, 50), (70, 65), (80, 78), (90, 88), (100, 95)]
   )

   spruce_carbon = Curve(
       label="SP-SI40_carbon",
       points=[(0, 0), (10, 0.5), (20, 3), (30, 8), (40, 15),
               (50, 25), (60, 38), (70, 50), (80, 60), (90, 68), (100, 73)]
   )

   # Calculate carbon stock for a development type
   dt_area = 500.0  # hectares
   dt_age = 60  # years

   carbon_per_ha = df_carbon(dt_age)
   total_carbon = carbon_per_ha * dt_area

   print(f"Carbon stock: {total_carbon:.1f} tonnes C")
   print(f"CO₂ equivalent: {total_carbon * 3.67:.1f} tonnes CO₂")

Carbon Fluxes from Harvest
--------------------------

When trees are harvested, carbon is released through:

1. **Immediate decomposition**: Slash and residues decompose (5-20 year half-life)
2. **Product decay**: Wood products gradually release carbon (decades to centuries)
3. **Soil disturbance**: Harvesting disturbs soil organic matter

.. mermaid::

   graph TD
     HARVEST["Harvest"] --> SLASH["Slash decomposition<br/>(fast, 5-20 yr)"]
     HARVEST --> PRODUCTS["Wood products<br/>(slow, decades)"]
     HARVEST --> SOIL["Soil disturbance<br/>(very slow, centuries)"]
     SLASH --> RELEASE["CO₂ release"]
     PRODUCTS --> RELEASE
     SOIL --> RELEASE

A simple carbon accounting model:

.. code-block:: python

   def calculate_harvest_carbon_flux(volume_harvested_m3, species="DF"):
       """Calculate carbon flux from a harvest event."""

       # Conversion: m³ to tonnes biomass (approximate)
       bulk_density = {"DF": 0.45, "SP": 0.40, "CE": 0.35}
       biomass_factor = bulk_density.get(species, 0.40)

       biomass_tonnes = volume_harvested_m3 * biomass_factor
       carbon_tonnes = biomass_tonnes * 0.5

       # Split into pools
       slash_fraction = 0.3  # 30% goes to slash (fast decomposition)
       product_fraction = 0.6  # 60% goes to products (slow decomposition)
       soil_fraction = 0.1  # 10% from soil disturbance

       slash_carbon = carbon_tonnes * slash_fraction
       product_carbon = carbon_tonnes * product_fraction
       soil_carbon = carbon_tonnes * soil_fraction

       return {
           "slash": slash_carbon,
           "products": product_carbon,
           "soil": soil_carbon,
           "total": carbon_tonnes
       }

   flux = calculate_harvest_carbon_flux(1000, species="DF")
   print(f"Slash carbon: {flux['slash']:.1f} tonnes C")
   print(f"Product carbon: {flux['products']:.1f} tonnes C")
   print(f"Soil carbon: {flux['soil']:.1f} tonnes C")
   print(f"Total flux: {flux['total']:.1f} tonnes C")

Carbon in Optimization
----------------------

Carbon can be incorporated into the optimization objective:

.. code-block:: python

   from ws3.opt import Problem

   prob = Problem("carbon_optimization")

   # Decision variables
   harvest_var_names = {}
   for dt_code in ["DF-SI50", "SP-SI40"]:
       for period in range(20):
           var_name = f"harv_{dt_code}_p{period}"
           prob.add_var(var_name, vtype="continuous", lb=0)
           harvest_var_names[(dt_code, period)] = var_name

   # Objective: maximize NPV + carbon revenue
   # z() takes a dict keyed on variable names
   timber_price = 50  # $/m³
   carbon_price = 50  # $/tonne CO₂ (ETS price)
   discount_rate = 0.05

   npv_coeffs = {}
   for (dt_code, period), var_name in harvest_var_names.items():
       volume_per_ha = 200  # m³/ha
       timber_revenue = volume_per_ha * timber_price
       carbon_revenue = volume_per_ha * 0.45 * 0.5 * 3.67 * carbon_price
       coeff = (timber_revenue + carbon_revenue) * (1 + discount_rate) ** (-period * 5)
       npv_coeffs[var_name] = coeff
   prob.z(coeffs=npv_coeffs)

   # Constraint: carbon budget (max allowable emissions)
   max_carbon_budget = 10000  # tonnes CO₂ over 100 years
   carbon_per_m3 = 0.45 * 0.5 * 3.67  # tonnes CO₂ per m³
   carbon_coeffs = {}
   for var_name in harvest_var_names.values():
       carbon_coeffs[var_name] = 200 * carbon_per_m3
   prob.add_constraint("carbon_budget", coeffs=carbon_coeffs, sense="leq", rhs=max_carbon_budget)

   # Set solver and solve
   prob.solver("highs")
   prob.solve()

Carbon Reporting
----------------

For regulatory compliance, you need to report carbon stocks and fluxes:

.. code-block:: python

   import pandas as pd

   # Calculate carbon stocks by period
   carbon_by_period = []
   for period in range(20):
       age = 20 + period * 5  # starting age + periods elapsed
       carbon_stock = df_carbon(age) * 500  # tonnes C
       carbon_by_period.append({
           "period": period,
           "age": age,
           "carbon_stock_tonnes_C": carbon_stock,
           "carbon_stock_tonnes_CO2": carbon_stock * 3.67
       })

   df_carbon_report = pd.DataFrame(carbon_by_period)
   print(df_carbon_report)

   # Export for reporting
   df_carbon_report.to_csv("carbon_report.csv", index=False)

Policy Context
--------------

British Columbia's carbon pricing framework:

- **Carbon tax**: Applied to fossil fuel combustion (not directly to forestry)
- **Emissions Trading System (ETS)**: Cap-and-trade for large emitters
- **Forest Management Carbon Budget**: Each FMU has an allowable carbon
  budget based on projected stock changes

Key references:

- BC Ministry of Forests: *Carbon Accounting Guidelines for Forest Management*
- BC Emissions Trading Scheme: *Forest Sector Participation*
- IPCC: *Good Practice Guidance for Land Use, Land-Use Change and Forestry*

Limitations
-----------

Carbon modelling in ws3 has limitations:

1. **Simplified pools**: Only above-ground biomass is typically modelled
2. **No soil dynamics**: Soil carbon changes are approximated
3. **No product substitution**: Benefits of wood substitution for concrete/steel
   are not modelled
4. **Static prices**: Carbon prices are assumed constant

For more sophisticated carbon accounting, consider coupling ws3 with
specialized tools like CO2STATS or the BC Forest Carbon Calculator.

Exercises
---------

**Exercise 1 (Easy)**: Calculate the carbon stock for a 500-hectare
Douglas-fir stand at ages 40, 60, 80, and 100.

**Exercise 2 (Medium)**: Extend the carbon calculation to include below-ground
biomass (assume root-shoot ratio of 0.2) and calculate total ecosystem carbon.

**Exercise 3 (Hard)**: Formulate an optimization problem that maximizes
NPV subject to a carbon budget constraint. Compare the optimal harvest
schedule with and without the carbon constraint.

Further Reading
---------------

- :doc:`ch07_financial_analysis` — Financial analysis
- :doc:`ch05_optimization` — Optimization fundamentals
- :doc:`/howto/financial-scenarios` — Financial scenario analysis
- IPCC Good Practice Guidance for Land Use, Land-Use Change and Forestry