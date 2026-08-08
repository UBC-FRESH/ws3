Chapter 8: Uncertainty and Risk
===============================

Learning Objectives
-------------------

After reading this chapter, you should be able to:

- Identify sources of uncertainty in wood supply models
- Use scenario analysis to explore uncertain outcomes
- Understand the limitations of deterministic models
- Apply basic risk assessment techniques to forest management plans

Why Does Uncertainty Matter?
----------------------------

Forest management operates in a world of uncertainty:

- **Growth uncertainty**: Trees don't grow exactly as predicted
- **Market uncertainty**: Timber prices fluctuate
- **Disturbance uncertainty**: Fire, insects, windthrow
- **Policy uncertainty**: Regulations may change
- **Climate uncertainty**: Future climate may differ from historical

Ignoring uncertainty can lead to:

- Over-optimistic harvest plans
- Inadequate buffer stocks
- Financial losses
- Ecological damage

Sources of Uncertainty
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Source
     - Description
   * - Growth
     - Actual growth may differ from predicted curves
   * - Prices
     - Timber prices change over time
   * - Disturbances
     - Fire, insects, windthrow reduce inventory
   * - Policy
     - New regulations may restrict harvest
   * - Climate
     - Future climate may alter growth patterns

Scenario Analysis
-----------------

**Scenario analysis** explores how outcomes change under different
assumptions. Instead of a single "best estimate," you examine multiple
scenarios:

.. code-block:: python

   from ws3.forest import ForestModel
   from ws3.core import Curve

   # Scenario analysis in ws3 is done by comparing outcomes from
   # different model configurations. The typical workflow:
   #
   #   1. Build model with optimistic growth curves:
   #      model = ForestModel("optimistic", "/path/to/optimistic_data",
   #                          2024, horizon=20, period_length=10)
   #      model.import_areas_section()
   #      model.import_yields_section()  # uses optimistic curves
   #      model.import_actions_section()
   #      model.import_transitions_section()
   #      model.reset_actions()
   #      model.grow(start_period=1)
   #
   #   2. Build model with pessimistic growth curves:
   #      model = ForestModel("pessimistic", "/path/to/pessimistic_data",
   #                          2024, horizon=20, period_length=10)
   #      # ... same import steps with pessimistic data ...
   #
   #   3. Compare results by querying area/volume at each period:
   #      for period in model.periods:
   #          for dtype in model.dtypes.values():
   #              area = dtype.area(period)
   #              # query yield curves for volume at current age
   #
   # Curve construction uses points=[(x,y)] format:
   #   optimistic_curve = Curve(label="optimistic_vol",
   #       points=[(0,0),(10,8),(20,35),(30,90),...,(100,640)],
   #       is_volume=True)

   # Example: define curves for scenario comparison
   optimistic_curve = Curve(
       label="optimistic_vol",
       is_volume=True,
       points=[(0, 0), (10, 8), (20, 35), (30, 90), (40, 160),
               (50, 260), (60, 380), (70, 500), (80, 580),
               (90, 620), (100, 640)]
   )

   pessimistic_curve = Curve(
       label="pessimistic_vol",
       is_volume=True,
       points=[(0, 0), (10, 3), (20, 15), (30, 40), (40, 75),
               (50, 130), (60, 200), (70, 280), (80, 350),
               (90, 400), (100, 420)]
   )

   # Compare by running separate models with different curve data
   # and querying dtype.area(period) and yield curve values for each period

Monte Carlo Simulation
----------------------

**Monte Carlo simulation** generates many random scenarios to estimate
the probability distribution of outcomes:

.. code-block:: python

   import numpy as np

   # Define growth curve parameters
   mean_volume = 500  # m³/ha
   std_volume = 100   # m³/ha

   # Generate 1000 random scenarios
   n_scenarios = 1000
   volumes = np.random.normal(mean_volume, std_volume, n_scenarios)

   # Calculate NPV for each scenario
   npvs = []
   for vol in volumes:
       revenue = vol * 50  # $/m³
       npv = revenue / (1.05 ** 40) - 10000  # Discount to present
       npvs.append(npv)

   # Summarize results
   print(f"Mean NPV: ${np.mean(npvs):,.0f}")
   print(f"Std dev: ${np.std(npvs):,.0f}")
   print(f"P(NPV > 0): {np.mean(npvs > 0)*100:.1f}%")
   print(f"95th percentile: ${np.percentile(npvs, 95):,.0f}")

Risk Assessment
---------------

**Risk assessment** evaluates the likelihood and impact of adverse events:

.. code-block:: python

   # Define disturbance probabilities
   fire_prob = 0.02  # 2% chance per year
   insect_prob = 0.05  # 5% chance per year

   # Calculate probability of no disturbance over 100 years
   no_disturb_prob = (1 - fire_prob) ** 100 * (1 - insect_prob) ** 100
   print(f"Probability of no disturbance in 100 years: {no_disturb_prob*100:.1f}%")

   # Calculate expected volume loss
   expected_loss = 1 - no_disturb_prob
   print(f"Expected volume loss: {expected_loss*100:.1f}%")

Adaptive Management
-------------------

**Adaptive management** acknowledges uncertainty and adjusts plans as
new information becomes available:

.. mermaid::

   graph TD
     PLAN["Plan"] --> IMPLEMENT["Implement"]
     IMPLEMENT --> MONITOR["Monitor outcomes"]
     MONITOR --> LEARN["Learn from results"]
     LEARN --> ADJUST["Adjust plan"]
     ADJUST --> IMPLEMENT

Benefits of Adaptive Management:

1. **Reduces regret**: Plans can be adjusted based on actual outcomes
2. **Improves learning**: Monitoring generates new knowledge
3. **Builds resilience**: Flexible plans handle uncertainty better
4. **Increases stakeholder confidence**: Transparent process

Limitations of Deterministic Models
-----------------------------------

Deterministic wood supply models (like basic ws3 models) have limitations:

1. **Single outcome**: Only one "best" plan, no probability distribution
2. **Fixed parameters**: Growth curves, prices, costs are fixed
3. **No feedback**: Cannot learn from monitoring results
4. **Ignores tail risks**: Rare but severe events are not modeled

To address these limitations:

- Use scenario analysis to explore multiple futures
- Apply sensitivity analysis to identify key drivers
- Incorporate adaptive management principles
- Consider stochastic optimization for risk-aware decisions

Exercises
---------

**Exercise 1 (Easy)**: Run a scenario analysis with optimistic and
pessimistic growth curves. Compare the total harvest volumes.

**Exercise 2 (Medium)**: Perform a Monte Carlo simulation with 1000
scenarios to estimate the probability of NPV > 0.

**Exercise 3 (Hard)**: Design an adaptive management plan that includes
monitoring triggers and adjustment rules.

Further Reading
---------------

- :doc:`ch05_optimization` — Optimization fundamentals
- :doc:`../howto/faq` — Frequently asked questions
- :doc:`/guides/troubleshooting` — Common issues and solutions