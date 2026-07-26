Chapter 15: Modelling Natural Disturbances
==========================================

Learning Objectives
-------------------

After reading this chapter, you should be able to:

- Explain the role of natural disturbances in forest estate modelling
- Distinguish deterministic, stochastic, and scenario-based disturbance approaches
- Implement disturbance transitions and salvage logic in a ws3-style model
- Design resilience-oriented indicators for plans under disturbance uncertainty
- Connect disturbance outputs to downstream spatial and supply-chain analysis

Why Model Disturbances?
-----------------------

Natural disturbances (wildfire, insect outbreaks, windthrow, drought-related
mortality) are first-order drivers of long-term forest dynamics. Strategic plans
that ignore disturbance dynamics often overstate sustainable harvest and
understate system risk.

Ignoring disturbance can lead to:

- **Over-optimistic harvest schedules**: Models assume all inventory
  remains intact, but disturbances regularly remove it
- **Inadequate reserve planning**: No buffer for disturbance-related
  inventory losses
- **Poor risk assessment**: Cannot quantify the probability of meeting
  harvest targets
- **Weak adaptation planning**: Silviculture and salvage decisions are not
  stress-tested against plausible disturbance regimes

.. mermaid::

   graph TD
    FOREST["Forest landscape state"] --> FIRE["Wildfire"]
    FOREST --> INSECT["Insects/pathogens"]
    FOREST --> WIND["Wind/ice events"]
    FOREST --> DROUGHT["Drought stress"]
    FIRE --> IMPACT["Mortality, quality loss, access constraints"]
    INSECT --> IMPACT
    WIND --> IMPACT
    DROUGHT --> IMPACT
    IMPACT --> TRANSITIONS["New transitions + salvage options"]
    TRANSITIONS --> PLAN["Updated harvest, risk, and regeneration plan"]

Disturbance Modelling Paradigms
-------------------------------

Three common paradigms are used in strategic forest modelling.

1. Deterministic deductions
   A fixed annual loss percentage is applied to affected strata. This is simple
   and fast, but masks volatility and spatial clustering.

2. Stochastic event simulation
   Random events are sampled from probability distributions (occurrence, size,
   severity). This captures variability and tail risk but is computationally heavier.

3. Scenario envelopes
   Several plausible disturbance futures are imposed (e.g., low/medium/high
   fire pressure). This is practical for policy communication and sensitivity analysis.

In practice, organizations often combine 2 and 3: stochastic simulation inside
named scenario envelopes.

Disturbance Types
-----------------

**Wildfire**

Wildfire is often modelled as a hazard process with event size and severity
sub-models.

Key dimensions:

- **Probability**: Annual probability of ignition per hectare
- **Spread/size**: Distribution of fire sizes and weather-conditioned spread
- **Severity**: Surface vs crown effects with class-specific mortality
- **Seasonality**: Higher risk in dry, hot periods

.. code-block:: python

   # Example hazard proxy for scenario screening.
   # Use calibrated regional models in production.

   def wildfire_probability(age, moisture_index, slope_aspect):
       """Estimate annual wildfire probability."""
       # Fuel load proxy rises with stand age.
       base_prob = 0.001 * (1 + age / 100)

       # Moisture reduces risk.
       moisture_factor = max(0.1, 1 - moisture_index)

       # South/west aspects are often drier.
       aspect_factor = {"S": 1.2, "SW": 1.1, "W": 1.0,
                        "NW": 0.9, "N": 0.8, "NE": 0.85,
                        "E": 0.9, "SE": 0.95}.get(slope_aspect, 1.0)

       return base_prob * moisture_factor * aspect_factor

**Insect Outbreaks**

Insects and pathogens can generate multi-year mortality pulses and quality
degradation.

Typical features:

- **Host specificity**: Some insects attack specific species
- **Cyclicity**: Outbreak cycles can repeat over decades
- **Threshold effects**: Damage can accelerate beyond density/age thresholds
- **Climate sensitivity**: Warmer winters reduce insect mortality

**Windthrow**

Wind and ice damage produce abrupt structural loss, especially in exposed stands.

Important predictors:

- **Exposure**: Coastal and ridge-top stands are more vulnerable
- **Soil depth**: Shallow soils increase uprooting risk
- **Tree height**: Taller trees are more susceptible
- **Frequency**: Return intervals of 50-200 years for major events

From Disturbance to Transition Logic
------------------------------------

In an aspatial strategic model, disturbance effects are typically represented by
area transfers between development types, with optional salvage pathways.

Conceptually, for development type :math:`d` in period :math:`t`:

.. math::

  A_{d,t+1} = A_{d,t} - H_{d,t} - D_{d,t} + R_{d,t} + S_{d,t}

where:

- :math:`A_{d,t}` is standing area,
- :math:`H_{d,t}` is planned harvest,
- :math:`D_{d,t}` is disturbed area,
- :math:`R_{d,t}` is regeneration inflow,
- :math:`S_{d,t}` is salvage-related transfer.

This bookkeeping is the core of disturbance-aware planning.

Integrating Disturbances with ws3
---------------------------------

ws3 can represent disturbance through actions, transitions, and yield impacts.
The exact API varies by project conventions, but the pattern is stable.

Common implementation pattern:

1. **State classes**: Add post-disturbance development types
2. **Disturbance actions**: Define FIRE/INSECT/WIND actions that transfer area
3. **Salvage actions**: Define optional salvage transitions with reduced yields
4. **Regeneration pathways**: Route disturbed classes into managed recovery

.. code-block:: python

   from ws3.forest import ForestModel

   model = ForestModel()

   # Add post-disturbance state class.
   model.add_development_type(
       code="DF-SI50-disturbed",
       area=0.0,
       age=0,
       species="Pseudotsuga menziesii",
       site_index=50
   )

   # Disturbance transition.
   model.add_action(
       code="FIRE",
       descr="Wildfire disturbance",
       components=["volume"],
       transitions={
           "DF-SI50": "DF-SI50-disturbed"
       }
   )

   # Optional salvage pathway.
   model.add_action(
       code="SALVAGE",
       descr="Salvage harvest after disturbance",
       components=["volume"],
       transitions={
           "DF-SI50-disturbed": "DF-SI50-regen"
       }
   )

   # The simulation controller applies FIRE stochastically and SALVAGE
   # according to policy and operational constraints.

Scenario Design for Disturbance Planning
----------------------------------------

A useful strategic experiment set usually varies four levers:

- **Disturbance intensity**: expected annual disturbed area
- **Disturbance severity**: merchantability loss and regeneration delay
- **Operational response**: salvage capacity, access constraints, replanting lag
- **Climate trend**: non-stationary shift in disturbance frequency/severity

Example scenario matrix:

.. list-table:: Disturbance Scenario Envelope
   :header-rows: 1
   :widths: 20 20 20 40

   * - Scenario
     - Disturbance rate
     - Salvage capacity
     - Interpretation
   * - S1 Baseline
     - Historical mean
     - Current
     - Continuation of recent regime
   * - S2 Elevated fire
     - +30%
     - Current
     - Stress-test under warmer/drier conditions
   * - S3 Elevated + response
     - +30%
     - Expanded
     - Tests operational adaptation investment
   * - S4 Compound risk
     - +30% fire + outbreaks
     - Expanded
     - Multi-disturbance pressure case

Resilience Metrics
------------------

Beyond total harvest, disturbance-aware plans should track resilience metrics:

- **Reliability**: probability of meeting minimum harvest commitments
- **Recovery time**: years to return to baseline harvest capacity after shock
- **Structural diversity**: area balance across age/species cohorts
- **Salvage dependence**: share of realized harvest from salvage rather than planned entries
- **Regeneration debt**: deferred re-establishment area

These metrics often reveal plan fragility that average-volume indicators hide.

Challenges
----------

Disturbance modelling remains difficult because:

1. **Stochasticity**: Disturbances are inherently random
2. **Scale mismatch**: Strategic models are often aspatial while disturbance is spatially clustered
3. **Data sparsity**: Long clean time series are rare for many regions and agents
4. **Non-stationarity**: Climate trend breaks historical frequency assumptions
5. **Compounding effects**: Fire-after-beetle, drought-after-thinning, and access failures interact

Worked Example: Simple Disturbance Stress Test
----------------------------------------------

Assume a planning unit with 100,000 ha operable area and baseline planned
harvest capacity of 2,000 ha/year.

- Baseline disturbance: 1.0%/year (1,000 ha/year)
- Elevated disturbance: 1.5%/year (1,500 ha/year)
- Salvageable fraction: 40% of disturbed area
- Salvage utilization: 70% of salvageable area

Under elevated disturbance:

1. Disturbed area increases by 500 ha/year.
2. Salvage recovered area is :math:`500 * 0.4 * 0.7 = 140` ha/year.
3. Net additional unavailable area is approximately 360 ha/year.

If this persists, the model should test whether regeneration and treatment
programs can offset this gap fast enough to maintain medium-term flow targets.

Link to Spatial and Supply-Chain Modules
----------------------------------------

This chapter connects directly to:

- :doc:`ch14_spades_integration` for spatially explicit event simulation
- :doc:`ch16_supply_chain` for downstream value/consumption effects

A key practical insight is that disturbance is not only an ecological risk; it is
also a fibre quality, timing, and logistics risk for the supply chain.

Future Directions
-----------------

Potential extensions to ws3 for disturbance modelling:

- **Risk-constrained optimization**: add reliability constraints on key outputs
- **Adaptive re-planning**: periodic parameter updates from monitoring streams
- **Coupled models**: integrate with SpaDES for event realism and feedback
- **Policy levers**: explicit salvage, reserve, and regeneration investment controls

Exercises
---------

1. Easy: Add a disturbed state class and define FIRE and SALVAGE transitions
   for one species/site class in a toy model.
2. Medium: Build three disturbance scenarios (baseline, elevated, elevated with
   increased salvage) and compare reliability and recovery time.
3. Hard: Formulate a planning objective that maximizes discounted harvest value
   subject to a minimum probability of meeting flow commitments under
   stochastic disturbance.

Further Reading
---------------

- :doc:`ch14_spades_integration` — Integrating ws3 with SpaDES
- :doc:`ch08_uncertainty_and_risk` — Uncertainty and risk analysis
- :doc:`ch16_supply_chain` — Value-creation and supply-chain response to shocks
- Government and agency disturbance atlases for your planning region (fire,
  insects, and wind events)