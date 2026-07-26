Chapter 11: Building Models with FEMIC
======================================

Learning Objectives
-------------------

After reading this chapter, you should be able to:

- Explain what FEMIC is and how it relates to ws3
- Create a FEMIC instance (a complete, runnable forest estate model)
- Use FEMIC's pipeline and workflow system to automate model building
- Bridge FEMIC instances to ws3 for simulation and optimization
- Understand the FEMIC configuration and parameter system

What Is FEMIC?
--------------

**FEMIC** (Forest Estate Modeling Integrated Components) is a framework
for building, configuring, and running forest estate models. While ws3
provides the low-level simulation and optimization engine, FEMIC provides
the higher-level infrastructure to:

1. **Define instances**: A FEMIC "instance" is a complete, self-contained
   forest estate model — inventory, growth curves, actions, constraints,
   and parameters — all configured and ready to run.
2. **Automate model building**: Pipelines and workflows handle the tedious
   parts: loading inventory data, generating development types, fitting
   growth curves, defining actions and transitions.
3. **Ensure reproducibility**: Instances are defined by configuration files,
   not interactive sessions. Run the same instance twice and get the same
   results.
4. **Bridge to ws3**: FEMIC instances can be materialized into ws3
   :py:class:`ws3.forest.ForestModel` objects for simulation and optimization.

.. mermaid::

   graph TD
     CONFIG["FEMIC Instance<br/>Configuration"] --> PIPELINE["Pipeline<br/>(automated build)"]
     PIPELINE --> INSTANCE["Instance<br/>(complete model)"]
     INSTANCE --> WS3["ws3 ForestModel<br/>(simulation/optimization)"]
     WS3 --> RESULTS["Results<br/>(schedule, NPV, etc.)"]

The Instance Concept
--------------------

A **FEMIC instance** is the central unit of work. It represents a complete
forest estate model for a specific area of interest, with all parameters
defined. Think of it as a "model recipe" — you can instantiate the same
recipe for different areas or scenarios.

.. code-block:: python

   from femic.instance_bootstrap import bootstrap_instance
   from femic.instance_context import InstanceContext

   # Bootstrap an instance from configuration
   instance = bootstrap_instance(
       instance_name="my_fmu",
       area_of_interest="data/aoi.shp",
       inventory="data/inventory.geojson"
   )

   # The instance now contains:
   # - Development types (from inventory)
   # - Growth curves (from vdyp parameters)
   # - Actions and transitions
   # - Model parameters (horizon, period length, etc.)

   # Inspect the instance
   print(f"Development types: {len(instance.development_types)}")
   print(f"Total area: {instance.total_area()} ha")
   print(f"Horizon: {instance.horizon} periods")

Pipelines
---------

**Pipelines** automate the process of building an instance from raw data.
A pipeline is a sequence of steps that transform inventory data into a
complete model.

Common pipeline steps:

1. **Data loading**: Read inventory from GeoJSON, shapefile, or CSV
2. **Aggregation**: Group inventory records into development types
3. **Curve fitting**: Generate growth curves from inventory data or
   provincial yield tables
4. **Action definition**: Define management actions and transitions
5. **Validation**: Check the instance for consistency

.. code-block:: python

   from femic.pipeline import Pipeline

   # Define a pipeline
   pipeline = Pipeline(
       steps=[
           "load_inventory",
           "aggregate_development_types",
           "fit_growth_curves",
           "define_actions",
           "validate_instance"
       ]
   )

   # Run the pipeline
   instance = pipeline.run(
       inventory="data/inventory.geojson",
       output_dir="output/my_instance"
   )

Workflows
---------

**Workflows** orchestrate multiple pipelines and instances. A workflow
defines the overall modeling process: build the base model, run scenarios,
compare results.

.. code-block:: python

   from femic.workflows import Workflow

   # Define a workflow with multiple scenarios
   workflow = Workflow(
       name="harvest_scenarios",
       scenarios=[
           {"name": "baseline", "params": {"max_harvest": 200}},
           {"name": "conservation", "params": {"max_harvest": 100}},
           {"name": "intensive", "params": {"max_harvest": 400}}
       ]
   )

   # Run all scenarios
   results = workflow.run()

   # Compare results
   for scenario_name, result in results.items():
       print(f"{scenario_name}: NPV = ${result.npv:,.0f}")

The FEMIC-to-ws3 Bridge
-----------------------

FEMIC provides a bridge to convert instances into ws3 models:

.. code-block:: python

   from femic.ws3_bridge import instance_to_ws3_model

   # Convert a FEMIC instance to a ws3 ForestModel
   ws3_model = instance_to_ws3_model(instance)

   # Now use ws3 for simulation
   results = ws3_model.run_simulation(horizon=instance.horizon)

   # Or for optimization
   from ws3.opt import Problem
   prob = Problem()
   # ... build optimization problem using ws3_model ...
   prob.solve(solver="highs")

This bridge ensures that the complex configuration defined in FEMIC
translates correctly into ws3's data structures.

FreshForge Integration
----------------------

FEMIC integrates with **FreshForge**, a tool for materializing and managing
model configurations. FreshForge handles:

- Parameter versioning and tracking
- Configuration templating
- Reproducible environment setup

.. code-block:: python

   from femic.freshforge import FreshForgeMaterializer

   # Materialize a configuration from FreshForge
   materializer = FreshForgeMaterializer(
       config_repo="freshforge_configs",
       config_version="v1.2.0"
   )

   instance = materializer.materialize(
       template="bc_fmu_template",
       parameters={"fmu_name": "my_fmu", "horizon": 20}
   )

Configuration Files
-------------------

FEMIC instances are typically defined by configuration files:

.. code-block:: yaml

   # instance_config.yaml
   instance:
     name: my_fmu
     area_of_interest: data/aoi.shp
     horizon: 20
     period_length: 5

   inventory:
     source: data/inventory.geojson
     aggregation:
       keys: [species, site_index]
       min_area: 10.0

   curves:
     volume:
       source: provincial_yield_tables
       species_mapping:
         Douglas-fir: Pseudotsuga menziesii
         Spruce: Picea sitchensis

   actions:
     - code: HARV
       descr: Clearcut harvest
       transitions:
         DF-SI50: Bare
         SP-SI40: Bare

     - code: PLNT
       descr: Plant after harvest
       transitions:
         Bare: DF-SI50

Best Practices
--------------

1. **Version your configurations**: Use FreshForge to track configuration
   changes over time
2. **Test instances before running**: Use FEMIC's validation to catch
   errors early
3. **Use pipelines for reproducibility**: Don't build instances interactively
4. **Separate data from configuration**: Keep inventory data separate from
   model parameters
5. **Document your instances**: Include metadata about the area, data
   sources, and assumptions

Exercises
---------

**Exercise 1 (Easy)**: Create a FEMIC instance from a sample inventory
dataset and print the development type summary.

**Exercise 2 (Medium)**: Build a pipeline that loads inventory data,
aggregates into development types, and fits growth curves.

**Exercise 3 (Hard)**: Create a workflow that runs three harvest scenarios
(baseline, conservation, intensive) and compares their NPV outcomes.

Further Reading
---------------

- :doc:`ch12_fhops_integration` — Using fhops for harvest cost curves
- :doc:`ch13_freshforge` — Automating workflows with FreshForge
- :doc:`/reference/modules/forest` — ws3 ForestModel API reference
- FEMIC documentation: https://femic.readthedocs.io