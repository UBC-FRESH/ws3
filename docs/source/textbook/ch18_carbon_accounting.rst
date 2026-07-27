.. _textbook_ch18_carbon_accounting:

=============================
Chapter 18: Carbon Accounting in Detail
=============================

Learning Objectives
-------------------

After reading this chapter, you should be able to:

- Understand forest carbon cycling and pool dynamics
- Model carbon stocks and fluxes in forest ecosystems
- Integrate carbon accounting with harvest optimization
- Calculate carbon offsets and credits
- Use FEMIC for detailed carbon modeling

Introduction
------------

Carbon accounting is increasingly important in forest management due to:

- **Climate change mitigation**: Forests as carbon sinks
- **Carbon markets**: Trading carbon offsets and credits
- **Policy requirements**: Government carbon reporting
- **Sustainability certification**: Carbon footprint assessment

Forest ecosystems store carbon in multiple pools:

- **Above-ground biomass**: Trees, shrubs, epiphytes
- **Below-ground biomass**: Roots
- **Deadwood**: Standing and fallen dead wood
- **Litter**: Organic matter on forest floor
- **Soil organic matter**: Decomposed organic material
- **Harvested products**: Wood products in use

This chapter explores detailed carbon accounting methods and their
integration with ws3 optimization.

Forest Carbon Pools
-------------------

**Above-Ground Biomass (AGB)**:

The largest carbon pool in most forests. Includes:

- Tree trunks, branches, and foliage
- Epiphytes and lianas
- Dead standing trees (snags)

Carbon content: ~50% of dry biomass

**Below-Ground Biomass (BGB)**:

Root systems that store carbon:

- Fine roots (<2mm diameter)
- Coarse roots (>2mm diameter)
- Root exudates

Typically 20-30% of AGB

**Deadwood**:

Dead organic matter in various stages of decomposition:

- **Standing dead trees**: Snags
- **Fallen logs**: Coarse woody debris
- **Small debris**: Branches, twigs

Decomposition rate depends on:

- Wood density
- Climate (temperature, moisture)
- Decomposer activity

**Litter**:

Organic matter on forest floor:

- Leaf litter
- Small branches
- Bark

Turnover rate: 1-5 years

**Soil Organic Matter (SOM)**:

Largest carbon pool in most forests:

- **Active pool**: Fast-turning over (decades)
- **Stable pool**: Slow-turning over (centuries)
- **Resistant pool**: Very slow turnover (millennia)

Carbon Stock Estimation
------------------------

**Allometric Equations**:

Estimate biomass from tree measurements:

.. math::

   AGB = a \cdot D^b \cdot H^c

Where:
- :math:`AGB` = above-ground biomass (kg)
- :math:`D` = diameter at breast height (cm)
- :math:`H` = total height (m)
- :math:`a`, :math:`b`, :math:`c` = species-specific parameters

**Example Equation** (Douglas-fir):

.. math::

   AGB = 0.0623 \cdot D^{1.96} \cdot H^{0.35}

**Carbon Content**:

Convert biomass to carbon:

.. math::

   C = AGB \cdot CF

Where :math:`CF` is the carbon fraction (typically 0.47-0.50).

Carbon Fluxes
-------------

**Gross Primary Production (GPP)**:

Total carbon fixed by photosynthesis:

.. math::

   GPP = NPP + R_a

Where:
- :math:`NPP` = net primary production
- :math:`R_a` = autotrophic respiration

**Net Primary Production (NPP)**:

Carbon available for growth after respiration:

.. math::

   NPP = GPP - R_a

**Net Ecosystem Production (NEP)**:

Carbon balance of entire ecosystem:

.. math::

   NEP = NPP - R_h

Where :math:`R_h` is heterotrophic respiration (decomposition).

**Harvest Effects**:

Harvesting affects carbon fluxes:

- **Immediate**: Release carbon from harvested biomass
- **Short-term**: Reduced photosynthesis (fewer trees)
- **Long-term**: Regrowth sequesters carbon
- **Product pools**: Carbon stored in wood products

Carbon Accounting with ws3
---------------------------

**Step 1: Define Carbon Objectives**

.. code-block:: python

   # compile_scenario is a user-defined helper (see examples/util.py),
   # not part of ws3.core. It builds a ws3.opt.Problem from a ForestModel.
   # The typical approach is to build the Problem directly:
   from ws3.opt import Problem
   
   prob = Problem("carbon_max")
   # Add variables, constraints, and objective using prob.add_var(),
   # prob.add_constraint(), and prob.z(coeffs=dict) as shown elsewhere
   # in this chapter and in ch05_optimization.rst.

**Step 2: Calculate Carbon Stocks**

.. code-block:: python

   def calculate_carbon_stocks(fm, schedule):
       """Calculate carbon stocks for a harvest schedule.
       
       :param fm: ForestModel instance
       :param schedule: harvest schedule
       :return: dictionary of carbon stocks by pool
       """
       carbon_stocks = {
           'above_ground': 0.0,
           'below_ground': 0.0,
           'deadwood': 0.0,
           'litter': 0.0,
           'soil': 0.0,
       }
       
       # Calculate stocks for each development type
       for dt_code in schedule['dt_code'].unique():
           dt_data = fm.development_types[
               fm.development_types['code'] == dt_code
           ]
           
           if dt_data.empty:
               continue
           
           # Get carbon stock estimates (simplified)
           age = dt_data['age'].values[0]
           volume = dt_data['volume_m3_ha'].values[0]
           
           # Estimate carbon stocks (tC/ha)
           agb = volume * 0.5 * 0.5  # 50% carbon content
           c_stocks = {
               'above_ground': agb,
               'below_ground': agb * 0.2,  # 20% of AGB
               'deadwood': agb * 0.1,  # 10% of AGB
               'litter': agb * 0.05,  # 5% of AGB
               'soil': agb * 2.0,  # 2x AGB (typical ratio)
           }
           
           # Add to totals
           for pool, stock in c_stocks.items():
               carbon_stocks[pool] += stock * dt_data['area_ha'].values[0]
       
       return carbon_stocks

**Step 3: Calculate Carbon Fluxes**

.. code-block:: python

   def calculate_carbon_fluxes(carbon_stocks_pre, carbon_stocks_post):
       """Calculate carbon fluxes from pre to post harvest.
       
       :param carbon_stocks_pre: carbon stocks before harvest
       :param carbon_stocks_post: carbon stocks after harvest
       :return: dictionary of fluxes by pool
       """
       fluxes = {}
       
       for pool in carbon_stocks_pre.keys():
           flux = carbon_stocks_post[pool] - carbon_stocks_pre[pool]
           fluxes[pool] = flux
       
       # Total flux
       fluxes['total'] = sum(fluxes.values())
       
       return fluxes

**Step 4: Optimize for Carbon**

.. code-block:: python

   # Set solver and solve
   problem.solver("gurobi")
   problem.solve()
   
   # Get solution
   solution = problem.solution()
   for var_name, value in solution.items():
       if value > 0:
           print(f"  {var_name}: {value:.2f}")
   print(f"Objective value: {problem.z():.2f}")
   
   # Carbon stocks and fluxes are calculated from model output:
   # Query dtype.area(period) for each period, then apply
   # allometric equations to estimate carbon stocks.
   
   print("Carbon Fluxes (tC):")
   for pool, flux in fluxes.items():
       print(f"  {pool}: {flux:.2f}")
   print(f"  Total: {fluxes['total']:.2f}")

FEMIC Integration
-----------------

**FEMIC** (Forest Ecosystem Management Integration Component) provides
detailed carbon modeling with:

- **Multiple carbon pools**: Detailed pool structure
- **Decomposition models**: Different rates for each pool
- **Product cascades**: Carbon in harvested products
- **Disturbance effects**: Fire, insects, windthrow

**Using FEMIC with ws3**:

.. code-block:: python

   from ws3.integration import FEMICIntegrator
   
   # Create FEMIC integrator
   femic = FEMICIntegrator()
   
   # Calculate detailed carbon budget
   carbon_budget = femic.calculate_carbon_budget(
       schedule=schedule,
       landscape=fm.development_types
   )
   
   print("FEMIC Carbon Budget:")
   for key, value in carbon_budget.items():
       print(f"  {key}: {value:.2f}")

Carbon Market Applications
---------------------------

**Carbon Offsets**:

Forest carbon offsets represent verified carbon reductions:

- **Avoided deforestation**: Preventing carbon release
- **Afforestation/reforestation**: Adding new carbon stocks
- **Improved forest management**: Enhancing carbon sequestration

**Carbon Credits**:

Carbon credits can be traded in voluntary or compliance markets:

- **Price**: Varies by market ($5-100/tC)
- **Verification**: Third-party validation required
- **Additionality**: Must demonstrate carbon benefit beyond business-as-usual

**Calculating Carbon Revenue**:

.. code-block:: python

   def calculate_carbon_revenue(carbon_flux, carbon_price):
       """Calculate revenue from carbon credits.
       
       :param carbon_flux: net carbon sequestration (tC)
       :param carbon_price: price per tonne of carbon ($/tC)
       :return: revenue ($)
       """
       if carbon_flux < 0:
           return 0  # No revenue for carbon emissions
       
       return carbon_flux * carbon_price
   
   # Example calculation
   carbon_price = 25.0  # $/tC
   carbon_revenue = calculate_carbon_revenue(fluxes['total'], carbon_price)
   print(f"Carbon revenue: ${carbon_revenue:.2f}")

Case Study: Carbon-Neutral Forest Management
---------------------------------------------

**Objective**: Develop a harvest schedule that achieves carbon neutrality
while maintaining timber production.

**Constraints**:

- Minimum timber volume harvest
- Carbon neutrality (net flux = 0)
- Even-flow requirements
- Adjacency constraints

**Solution Approach**:

1. Define carbon neutrality constraint
2. Add timber production requirements
3. Solve multi-objective optimization
4. Analyze trade-offs

.. code-block:: python

   # Define carbon neutrality constraint
   problem.add_constraint(
       name="carbon_neutral",
       coeffs={'carbon_flux': 1.0},
       sense='eq',
       rhs=0.0
   )
   
   # Add timber requirement
   problem.add_constraint(
       name="timber_min",
       coeffs={'volume_harvest': 1.0},
       sense='geq',
       rhs=50000  # 50,000 m3 minimum
   )
   
   # Set solver and solve
   problem.solver("gurobi")
   problem.solve()
   
   # Get solution
   solution = problem.solution()
   carbon_balance = solution['carbon_flux']
   volume_harvest = solution['volume_harvest']
   print(f"Carbon balance: {carbon_balance:.2f} tC")
   print(f"Timber harvested: {volume_harvest:.2f} m3")

Summary
-------

This chapter covered detailed carbon accounting for forest management:

- **Carbon pools**: Above-ground, below-ground, deadwood, litter, soil
- **Carbon fluxes**: GPP, NPP, NEP, harvest effects
- **Stock estimation**: Allometric equations and carbon content
- **ws3 integration**: Carbon objectives and constraints
- **FEMIC**: Detailed carbon modeling
- **Carbon markets**: Offsets, credits, and revenue

These techniques enable forest managers to optimize for carbon while
maintaining other management objectives.

Exercises
---------

1. **Carbon Stocks**: Calculate carbon stocks for a hypothetical forest
   stand using allometric equations. Compare with literature values.

2. **Carbon Fluxes**: Calculate carbon fluxes for a harvest scenario.
   Identify which pools contribute most to emissions.

3. **Carbon Neutrality**: Develop a carbon-neutral harvest schedule for
   a 1000-hectare forest. Compare with business-as-usual scenario.

4. **Carbon Revenue**: Calculate carbon revenue for different carbon
   prices ($10, $25, $50, $100/tC). At what price does carbon become
   economically significant?

5. **FEMIC Integration**: Use FEMIC to model carbon dynamics for a
   rotational harvest system. Compare with simplified approach.

Related Resources
-----------------

* :doc:`carbon-modelling` (how-to guide)
* :doc:`../textbook/ch10_carbon_modelling` (introductory carbon)
* :doc:`../textbook/ch11_femic_models` (FEMIC details)
* IPCC Guidelines: https://www.ipcc.ch/report/2006-ipcc-national-greenhouse-gas-inventory/