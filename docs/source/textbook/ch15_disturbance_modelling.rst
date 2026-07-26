Chapter 15: Modelling Natural Disturbances
==========================================

.. note::

   This chapter is a work-in-progress. The structure and some content are
   in place, but detailed examples and case studies will be added as they
   become available.

Learning Objectives
-------------------

After reading this chapter, you should be able to:

- Explain the role of natural disturbances in forest estate modelling
- Describe the main disturbance types modelled in forest planning
- Understand how disturbances affect development types and transitions
- Identify opportunities to extend ws3 with disturbance sub-models
- Recognize the limitations of current disturbance modelling in ws3

Why Model Disturbances?
-----------------------

Natural disturbances — wildfire, insect outbreaks, windthrow — are
fundamental drivers of forest dynamics. Ignoring them in planning models
leads to:

- **Over-optimistic harvest schedules**: Models assume all inventory
  remains intact, but disturbances regularly remove it
- **Inadequate reserve planning**: No buffer for disturbance-related
  inventory losses
- **Poor risk assessment**: Cannot quantify the probability of meeting
  harvest targets

.. mermaid::

   graph TD
     FOREST["Forest landscape"] --> FIRE["Wildfire"]
     FOREST --> INSECT["Insect outbreak"]
     FOREST --> WIND["Windthrow"]
     FOREST --> OTHER["Other disturbances"]
     FIRE --> IMPACT["Inventory loss"]
     INSECT --> IMPACT
     WIND --> IMPACT
     OTHER --> IMPACT
     IMPACT --> PLANNING["Planning implications"]

Disturbance Types
-----------------

**Wildfire**

The dominant natural disturbance in western Canadian forests. Characterized
by:

- **Probability**: Annual probability of ignition per hectare
- **Size**: Distribution of fire sizes (typically log-normal)
- **Severity**: Crown fire vs. surface fire (different mortality rates)
- **Seasonality**: Higher risk in dry, hot periods

.. code-block:: python

   # Example: Wildfire probability model
   # (Placeholder — actual implementation requires calibration)

   def wildfire_probability(age, moisture_index, slope_aspect):
       """Estimate annual wildfire probability."""
       # Base probability increases with age (fuel accumulation)
       base_prob = 0.001 * (1 + age / 100)

       # Moisture reduces probability
       moisture_factor = max(0.1, 1 - moisture_index)

       # Aspect affects dryness (south-facing is drier)
       aspect_factor = {"S": 1.2, "SW": 1.1, "W": 1.0,
                        "NW": 0.9, "N": 0.8, "NE": 0.85,
                        "E": 0.9, "SE": 0.95}.get(slope_aspect, 1.0)

       return base_prob * moisture_factor * aspect_factor

**Insect Outbreaks**

Bark beetle and defoliator outbreaks can cause widespread mortality:

- **Host specificity**: Some insects attack specific species
- **Cyclicity**: Outbreaks tend to be cyclical (every 30-80 years)
- **Threshold effects**: Damage accelerates past certain density thresholds
- **Climate sensitivity**: Warmer winters reduce insect mortality

**Windthrow**

Wind events can cause widespread treefall:

- **Exposure**: Coastal and ridge-top stands are more vulnerable
- **Soil depth**: Shallow soils increase uprooting risk
- **Tree height**: Taller trees are more susceptible
- **Frequency**: Return intervals of 50-200 years for major events

Integrating Disturbances with ws3
---------------------------------

Currently, ws3 does not have built-in disturbance modelling. However,
disturbances can be approximated by:

1. **Reducing development type areas**: Simulate disturbance by
   transferring area from affected DTs to a "disturbed" DT
2. **Modifying growth curves**: Reduce growth rates for damaged stands
3. **Adding transition rules**: Define post-disturbance transitions

.. code-block:: python

   from ws3.forest import ForestModel

   model = ForestModel()

   # Add a "disturbed" development type
   model.add_development_type(
       code="DF-SI50-disturbed",
       area=0.0,
       age=0,
       species="Pseudotsuga menziesii",
       site_index=50
   )

   # Define disturbance transition
   model.add_action(
       code="FIRE",
       descr="Wildfire disturbance",
       components=["volume"],
       transitions={
           "DF-SI50": "DF-SI50-disturbed"
       }
   )

   # In simulation, apply disturbance each period with given probability
   # (Implementation would go in the simulation loop)

Challenges
----------

Modelling natural disturbances is challenging because:

1. **Stochasticity**: Disturbances are inherently random
2. **Scale**: Disturbances operate at landscape scales
3. **Data scarcity**: Long-term disturbance records are limited
4. **Climate change**: Future disturbance regimes are uncertain
5. **Interaction**: Multiple disturbances can interact (e.g., fire after
   beetle kill)

Future Directions
-----------------

Potential extensions to ws3 for disturbance modelling:

- **Stochastic optimization**: Incorporate disturbance probability into
  the optimization objective
- **Scenario analysis**: Run multiple disturbance scenarios and compare
- **Adaptive management**: Update disturbance parameters based on
  monitoring data
- **Integration with SpaDES**: Use the `spades_ws3` module for
  spatially-explicit disturbance simulation

Exercises
---------

**Exercise 1 (Easy)**: Add a "disturbed" development type to a ws3
model and define a transition from a healthy type.

**Exercise 2 (Medium)**: Write a function that simulates wildfire
probability based on stand age and moisture conditions.

**Exercise 3 (Hard)**: Extend the ws3 optimization to include a
probabilistic disturbance constraint (e.g., ensure at least 80%
probability of meeting harvest targets despite disturbances).

Further Reading
---------------

- :doc:`ch14_spades_integration` — Integrating ws3 with SpaDES
- :doc:`ch08_uncertainty_and_risk` — Uncertainty and risk analysis
- BC Ministry of Forests: *Wildfire Risk Management*
- Natural Resources Canada: *Insect Disturbance in Canadian Forests*