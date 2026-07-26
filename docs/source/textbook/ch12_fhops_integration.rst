Chapter 12: Harvest Cost Curves with FHOPS
===========================================

Learning Objectives
-------------------

After reading this chapter, you should be able to:

- Explain what FHOPS is and its role in the ws3 ecosystem
- Generate harvest cost yield curves using fhops
- Understand the relationship between productivity, costing, and yield
- Inject fhops-generated curves into ws3 models
- Use the fhops CLI for common tasks

What Is FHOPS?
--------------

**FHOPS** (Forest Harvest Operations) is a tool for generating harvest
cost yield curves. While ws3 models *what* gets harvested and *when*,
FHOPS models *how much it costs* to harvest.

FHOPS fills a critical gap: traditional wood supply models often use
simplified, static harvest costs. FHOPS generates **dynamic harvest cost
curves** that vary by:

- **Productivity**: Site quality, stand density, tree size
- **Distance**: Distance to landing, road access
- **Terrain**: Slope, soil conditions
- **Species**: Different species require different harvesting techniques

.. mermaid::

   graph TD
     INPUTS["Input Data<br/>(inventory, terrain, roads)"] --> FHOPS["FHOPS<br/>Cost modeling"]
     FHOPS --> CURVES["Harvest Cost<br/>Yield Curves"]
     CURVES --> WS3["ws3 ForestModel<br/>(integration)"]
     WS3 --> OPT["Optimization<br/>(with accurate costs)"]

The Productivity Concept
------------------------

FHOPS organizes cost modeling around **productivity**. A productivity
class represents a combination of factors that affect harvesting efficiency:

- **Stand density**: More trees per hectare = more time per hectare
- **Tree size**: Larger trees = more time per tree
- **Terrain**: Steeper slopes = slower machine movement
- **Soil conditions**: Wet soils = reduced machine mobility

.. code-block:: python

   from fhops.productivity import ProductivityRegistry

   # Register productivity classes
   registry = ProductivityRegistry()

   registry.add_productivity_class(
       name="high_productivity",
       description="Flat terrain, good access, moderate density",
       base_cost_per_m3=25.0,
       density_factor=1.0,
       slope_factor=1.0,
       distance_factor=1.0
   )

   registry.add_productivity_class(
       name="low_productivity",
       description="Steep terrain, poor access, high density",
       base_cost_per_m3=45.0,
       density_factor=1.5,
       slope_factor=1.8,
       distance_factor=2.0
   )

Generating Cost Curves
----------------------

FHOPS generates harvest cost curves that relate harvest volume to cost:

.. code-block:: python

   from fhops.costing import CostCurveGenerator

   # Generate cost curves for different productivity classes
   generator = CostCurveGenerator(registry)

   cost_curves = generator.generate(
       productivity_classes=["high_productivity", "low_productivity"],
       volume_range=[0, 1000],  # m³/ha
       num_points=50
   )

   # Each curve maps volume to cost
   for pc_name, curve in cost_curves.items():
       print(f"{pc_name}: cost at 500 m³/ha = ${curve(500):.2f}")

Cost Curve Structure
--------------------

A harvest cost curve typically has this structure:

.. mermaid::

   graph TD
     VOLUME["Harvest Volume<br/>(m³/ha)"] --> FIXED["Fixed Costs<br/>(setup, mobilization)"]
     VOLUME --> VARIABLE["Variable Costs<br/>(per m³)"]
     VARIABLE --> DENSITY["Density adjustment"]
     VARIABLE --> DISTANCE["Distance adjustment"]
     VARIABLE --> SLOPE["Slope adjustment"]
     FIXED --> TOTAL["Total Cost<br/>($/ha)"]
     VARIABLE --> TOTAL

The total cost is:

.. math::

   \\text{Total Cost} = \\text{Fixed Costs} + \\text{Variable Cost per m³} \\times \\text{Volume} \\times \\text{Adjustment Factors}

Integrating with ws3
--------------------

The key integration point: fhops cost curves can be injected into ws3
as part of the financial analysis:

.. code-block:: python

   from ws3.forest import ForestModel
   from ws3.core import Curve
   from fhops.costing import CostCurveGenerator

   # Generate fhops cost curves
   registry = ProductivityRegistry()
   # ... add productivity classes ...
   generator = CostCurveGenerator(registry)
   cost_curves = generator.generate(
       productivity_classes=["high_productivity", "low_productivity"],
       volume_range=[0, 1000],
       num_points=50
   )

   # Create ws3 model
   model = ForestModel()

   # Add development types
   model.add_development_type(
       code="DF-SI50-HighProd",
       area=500.0,
       age=40,
       species="Pseudotsuga menziesii",
       site_index=50
   )

   # Add growth curve (volume)
   volume_curve = Curve(
       x=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
       y=[0, 5, 25, 65, 120, 200, 300, 400, 470, 500, 510],
       name="DF-SI50_volume"
   )
   model.add_curve("volume", volume_curve)

   # Add cost curve from fhops
   high_prod_cost = cost_curves["high_productivity"]
   cost_curve = Curve(
       x=high_prod_cost.x,
       y=high_prod_cost.y,
       name="DF-SI50_HighProd_cost"
   )
   model.add_curve("harvest_cost", cost_curve)

   # Now the model has both volume and cost curves
   # Use them in optimization to maximize net revenue
   volume_at_60 = volume_curve(60)  # m³/ha
   cost_at_60 = cost_curve(60)  # $/ha
   net_revenue_per_ha = volume_at_60 * 50 - cost_at_60  # $/ha

The fhops CLI
-------------

FHOPS provides a command-line interface for common tasks:

.. code-block:: bash

   # Generate cost curves from configuration
   fhops generate-cost-curves \
       --config config/costing.yaml \
       --output output/cost_curves.csv

   # List available productivity classes
   fhops list-productivity-classes

   # Validate a costing configuration
   fhops validate-config config/costing.yaml

   # Export curves for ws3 integration
   fhops export-ws3 config/costing.yaml output/ws3_curves/

CLI Configuration
-----------------

The fhops CLI uses YAML configuration files:

.. code-block:: yaml

   # config/costing.yaml
   productivity_classes:
     - name: high_productivity
       base_cost_per_m3: 25.0
       density_factor: 1.0
       slope_factor: 1.0
       distance_factor: 1.0

     - name: low_productivity
       base_cost_per_m3: 45.0
       density_factor: 1.5
       slope_factor: 1.8
       distance_factor: 2.0

   volume_range:
     min: 0
     max: 1000
     num_points: 50

   output:
     format: csv
     path: output/cost_curves.csv

Best Practices
--------------

1. **Calibrate costs to local conditions**: Use actual harvesting data
   to calibrate fhops parameters
2. **Validate against benchmarks**: Compare fhops output to known cost
   estimates
3. **Use productivity classes consistently**: Define clear criteria for
   each productivity class
4. **Document assumptions**: Record the data sources and assumptions
   behind cost parameters
5. **Version configurations**: Track changes to costing parameters over time

Exercises
---------

**Exercise 1 (Easy)**: Generate harvest cost curves for two productivity
classes and plot them.

**Exercise 2 (Medium)**: Create a ws3 model with both volume and cost
curves, and calculate net revenue at different ages.

**Exercise 3 (Hard)**: Build an optimization problem that uses fhops
cost curves to find the rotation age that maximizes net present value.

Further Reading
---------------

- :doc:`ch11_femic_models` — Building models with FEMIC
- :doc:`ch13_freshforge` — Automating workflows with FreshForge
- :doc:`ch07_financial_analysis` — Financial analysis fundamentals
- FHOPS documentation: https://fhops.readthedocs.io