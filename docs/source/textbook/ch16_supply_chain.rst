Chapter 16: Linking ws3 with Forest Sector Supply Chain Models
==============================================================

.. note::

   This chapter is a work-in-progress. The structure is in place, and
   the conceptual framework is described, but detailed examples and
   case studies will be added as material becomes available.

Learning Objectives
-------------------

After reading this chapter, you should be able to:

- Explain the relationship between wood supply models and forest sector
  supply chain models
- Identify the data interfaces between ws3 and downstream models
- Understand how ws3 schedules feed into logistics and processing models
- Recognize opportunities for integrated modelling across the forest
  sector
- Plan extensions to ws3 for supply chain integration

The Forest Sector Supply Chain
------------------------------

The forest products supply chain connects forest management to end
consumers:

.. mermaid::

   graph LR
     FMU["Forest Management<br/>Unit (ws3)"] --> LOGISTICS["Logistics<br/>(haulage, landing)"]
     LOGISTICS --> MILL["Milling<br/>(sawmill, pulp mill)"]
     MILL --> PRODUCT["Products<br/>(lumber, panels)"]
     PRODUCT --> MARKET["Markets<br/>(domestic, export)"]

ws3 operates at the **Forest Management Unit (FMU)** level, producing
harvest schedules. These schedules feed into:

1. **Logistics models**: Optimize haulage routes, landing locations,
   and transportation modes
2. **Milling models**: Schedule sawmill and pulp mill operations
3. **Market models**: Match supply to demand across regions

Data Interfaces
---------------

The key data flowing from ws3 to downstream models:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Data
     - Format
     - Description
   * - Harvest schedule
     - CSV/Excel
     - Period-by-period harvest by development type
   * - Volume by species
     - CSV
     - Merchantable volume by species and period
   * - Area by age class
     - CSV
     - Standing inventory by age class and period
   * - Net revenue
     - CSV
     - Financial performance by period

.. code-block:: python

   # Export ws3 results for supply chain integration
   results = model.run_simulation(horizon=20)

   # Export harvest schedule
   schedule = results.to_dataframe()
   schedule.to_csv("output/harvest_schedule.csv", index=False)

   # Export volume by species
   volume_by_species = results.volume_by_species()
   volume_by_species.to_csv("output/volume_by_species.csv", index=False)

Integration Patterns
--------------------

**Pattern 1: ws3 → Logistics optimization**

ws3 produces the harvest schedule. A logistics model takes this schedule
and determines:

- Which landing sites to use
- What haulage routes to employ
- When to build/maintain roads

.. code-block:: python

   # ws3 output → logistics input
   # (Conceptual — actual integration depends on the logistics model)

   harvest_schedule = ws3_results.harvest_by_period()

   # Logistics model would:
   # 1. Read harvest_schedule
   # 2. Optimize landing locations
   # 3. Assign harvest blocks to landings
   # 4. Calculate haulage costs

**Pattern 2: ws3 → Mill scheduling**

ws3 produces volume forecasts. A mill scheduling model takes these
forecasts and determines:

- What product mixes to produce
- When to process each species
- How to allocate inventory across mills

**Pattern 3: Integrated optimization**

In principle, ws3 and downstream models could be optimized jointly:

.. mermaid::

   graph TD
     WS3["ws3<br/>(harvest optimization)"] <--> LOGISTICS["Logistics<br/>(haulage optimization)"]
     WS3 --> MILL["Mill<br/>(production optimization)"]
     LOGISTICS --> MILL
     MILL --> MARKET["Market<br/>(demand matching)"]

This is an active area of research. Challenges include:

- **Computational complexity**: Joint optimization of many variables
- **Data requirements**: Detailed logistics and mill data
- **Model compatibility**: Different time scales and spatial resolutions

Current Limitations
-------------------

ws3 is not currently designed for direct supply chain integration:

1. **Aspatial output**: ws3 produces area-based schedules, not spatial
   harvest blocks
2. **No logistics costs**: Harvest costs in ws3 don't include haulage
3. **No mill constraints**: ws3 doesn't know about mill capacity
4. **No market signals**: ws3 optimizes based on timber prices, not
   end-product prices

Potential Extensions
--------------------

To enable supply chain integration, ws3 could be extended to:

1. **Export spatial data**: Include coordinates for harvest blocks
2. **Add logistics parameters**: Distance to landing, road access
3. **Support multiple objectives**: Include logistics and mill
   objectives in the optimization
4. **Provide APIs**: REST APIs for downstream model integration

Exercises
---------

**Exercise 1 (Easy)**: Export a ws3 harvest schedule to CSV and inspect
the output format.

**Exercise 2 (Medium)**: Design a data schema for passing ws3 results
to a hypothetical logistics optimization model.

**Exercise 3 (Hard)**: Formulate a joint optimization problem that
simultaneously optimizes harvest scheduling (ws3) and logistics
(haulage routes).

Further Reading
---------------

- :doc:`ch14_spades_integration` — Integrating ws3 with SpaDES
- :doc:`ch12_fhops_integration` — Using fhops for harvest cost curves
- Forest products supply chain literature
- Integrated forest planning research