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

   model = ForestModel()

   # Define development types
   model.add_development_type(
       code="DF-SI50",
       area=1000.0,
       age=40,
       species="Pseudotsuga menziesii",
       site_index=50
   )

   # Define growth curves for different scenarios
   optimistic_curve = Curve(
       x=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
       y=[0, 8, 35, 90, 160, 260, 380, 500, 580, 620, 640],
       name="optimistic"
   )

   pessimistic_curve = Curve(
       x=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
       y=[0, 3, 15, 40, 75, 130, 200, 280, 350, 400, 420],
       name="pessimistic"
   )

   # Run simulation with optimistic growth
   model.add_curve("volume", optimistic_curve)
   results_opt = model.run_simulation(horizon=20)
   print(f"Optimistic total volume: {results_opt.total_volume():.0f} m³")

   # Run simulation with pessimistic growth
   model.add_curve("volume", pessimistic_curve)
   results_pess = model.run_simulation(horizon=20)
   print(f"Pessimistic total volume: {results_pess.total_volume():.0f} m³")

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
- :doc:`/howto/model-validation` — Model validation techniques
- :doc:`/guides/troubleshooting` — Common issues and solutions