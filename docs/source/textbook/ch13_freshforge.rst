Chapter 13: Workflow Automation with FreshForge
================================================

Learning Objectives
-------------------

After reading this chapter, you should be able to:

- Explain what FreshForge is and its role in the UBC-FRESH ecosystem
- Use FreshForge to automate and orchestrate complex modeling workflows
- Create reproducible modeling pipelines that are transparent and auditable
- Integrate FreshForge with FEMIC and ws3 for end-to-end automation
- Understand the principles of reproducible forest modeling

What Is FreshForge?
-------------------

**FreshForge** is a framework for automating and orchestrating complex
forest modeling workflows. It addresses a fundamental problem: forest
models are often built interactively, making them difficult to reproduce,
audit, or scale.

FreshForge provides:

1. **Workflow orchestration**: Define complex multi-step workflows as
   code, not as a series of manual commands
2. **Reproducibility**: Every run produces the same results given the
   same inputs and configuration
3. **Transparency**: The workflow definition is human-readable and
   auditable
4. **Modularity**: Break complex workflows into reusable components
5. **Integration**: Works with FEMIC, ws3, fhops, and other UBC-FRESH
   tools

.. mermaid::

   graph TD
     WF["FreshForge<br/>Workflow Definition"] --> STEP1["Step 1:<br/>Data Preparation"]
     STEP1 --> STEP2["Step 2:<br/>FEMIC Instance"]
     STEP2 --> STEP3["Step 3:<br/>fhops Cost Curves"]
     STEP3 --> STEP4["Step 4:<br/>ws3 Simulation"]
     STEP4 --> STEP5["Step 5:<br/>Optimization"]
     STEP5 --> OUTPUT["Output:<br/>Schedule + Reports"]

The Problem FreshForge Solves
-----------------------------

Without FreshForge, a typical forest modeling workflow looks like this:

1. Open QGIS, load inventory data
2. Run a Python script to aggregate into development types
3. Open Excel, manually define growth curves
4. Write another Python script to build the ws3 model
5. Run the model, check results
6. If something is wrong, go back to step 2 and try again
7. Repeat until results look reasonable

This process is:

- **Not reproducible**: Hard to run the same analysis twice
- **Not transparent**: Hard to understand what was done
- **Not auditable**: Hard to verify results
- **Not scalable**: Hard to run for multiple FMUs

With FreshForge, the same workflow is defined as code:

.. code-block:: python

   from freshforge_workflows import Workflow, Step

   # Define the workflow
   workflow = Workflow(
       name="bc_fmu_analysis",
       description="Standard BC FMU wood supply analysis",
       steps=[
           Step(
               name="prepare_data",
               command="python scripts/prepare_data.py",
               inputs=["data/inventory.geojson"],
               outputs=["output/prepared_inventory.csv"]
           ),
           Step(
               name="build_instance",
               command="python scripts/build_instance.py",
               inputs=["output/prepared_inventory.csv"],
               outputs=["output/instance.pkl"]
           ),
           Step(
               name="generate_costs",
               command="fhops generate-cost-curves --config config/costing.yaml",
               inputs=["config/costing.yaml"],
               outputs=["output/cost_curves.csv"]
           ),
           Step(
               name="simulate",
               command="python scripts/simulate.py",
               inputs=["output/instance.pkl", "output/cost_curves.csv"],
               outputs=["output/simulation_results.csv"]
           ),
           Step(
               name="optimize",
               command="python scripts/optimize.py",
               inputs=["output/simulation_results.csv"],
               outputs=["output/optimal_schedule.csv"]
           )
       ]
   )

   # Run the workflow
   workflow.run()

Workflow Components
-------------------

FreshForge workflows consist of:

**Steps**: Individual tasks in the workflow (data preparation, simulation, etc.)

**Dependencies**: Define the order in which steps run

**Inputs/Outputs**: Track data flow between steps

**Parameters**: Configuration values that can be varied between runs

.. mermaid::

   graph TD
     S1["Step 1:<br/>Data Prep"] --> S2["Step 2:<br/>Instance Build"]
     S1 --> S3["Step 3:<br/>Cost Curves"]
     S2 --> S4["Step 4:<br/>Simulation"]
     S3 --> S4
     S4 --> S5["Step 5:<br/>Optimization"]
     S5 --> REPORT["Report Generation"]

Creating Reusable Steps
-----------------------

Steps can be packaged as reusable components:

.. code-block:: python

   from freshforge_workflows import Step, StepLibrary

   # Define a reusable step
   prepare_inventory_step = Step(
       name="prepare_inventory",
       command="python scripts/prepare_inventory.py {input} {output}",
       inputs=["{input}"],
       outputs=["{output}"],
       description="Prepare inventory data for modeling"
   )

   # Register it in a step library
   library = StepLibrary()
   library.register(prepare_inventory_step)

   # Use it in a workflow
   workflow = Workflow(
       name="my_analysis",
       steps=[
           library.get("prepare_inventory"),
           # ... other steps ...
       ]
   )

Materialization
---------------

FreshForge's **materialization** system ensures that workflows produce
consistent results by:

1. **Locking dependencies**: Pinning package versions
2. **Caching intermediates**: Avoiding re-computation
3. **Validating inputs**: Checking that required files exist
4. **Recording metadata**: Logging what was run and when

.. code-block:: python

   from freshforge_materialization import Materializer

   # Materialize a workflow with locked dependencies
   materializer = Materializer(
       workflow="workflows/bc_fmu_analysis.yaml",
       lockfile="workflows/lockfile.lock",
       cache_dir="output/cache"
   )

   result = materializer.run()

   print(f"Workflow completed: {result.success}")
   print(f"Steps executed: {len(result.executed_steps)}")
   print(f"Duration: {result.duration}")

Transparency and Auditability
-----------------------------

FreshForge workflows are designed to be transparent:

- **Human-readable**: Workflow definitions are YAML or Python
- **Version-controlled**: Store workflows in Git
- **Executable**: Run workflows directly, not just view them
- **Documented**: Each step has a description and purpose

.. code-block:: yaml

   # workflows/bc_fmu_analysis.yaml
   name: bc_fmu_analysis
   description: >
     Standard BC FMU wood supply analysis.
     Produces optimal harvest schedule for a management unit.

   steps:
     - name: prepare_data
       description: "Load and clean inventory data"
       command: "python scripts/prepare_data.py"
       inputs: ["data/inventory.geojson"]
       outputs: ["output/prepared_inventory.csv"]

     - name: build_instance
       description: "Build FEMIC instance from prepared data"
       command: "python scripts/build_instance.py"
       inputs: ["output/prepared_inventory.csv"]
       outputs: ["output/instance.pkl"]

   parameters:
     horizon: 20
     period_length: 5
     discount_rate: 0.05

End-to-End Example
------------------

A complete FreshForge workflow for a BC FMU:

.. code-block:: python

   from freshforge_workflows import Workflow, Step
   from freshforge_materialization import Materializer

   # Define the workflow
   workflow = Workflow(
       name="fmu_analysis",
       steps=[
           Step(
               name="load_inventory",
               command="python -m femic.cli load-inventory "
                       "--input {inventory} --output {output}",
               inputs=["{inventory}"],
               outputs=["{output}"]
           ),
           Step(
               name="build_instance",
               command="python -m femic.cli build-instance "
                       "--config {config} --output {output}",
               inputs=["{config}"],
               outputs=["{output}"]
           ),
           Step(
               name="generate_costs",
               command="fhops generate-cost-curves "
                       "--config {cost_config} --output {output}",
               inputs=["{cost_config}"],
               outputs=["{output}"]
           ),
           Step(
               name="simulate_and_optimize",
               command="python -m ws3.cli run "
                       "--instance {instance} "
                       "--costs {costs} "
                       "--horizon {horizon} "
                       "--output {output}",
               inputs=["{instance}", "{costs}"],
               outputs=["{output}"],
               params={"horizon": 20}
           ),
           Step(
               name="generate_report",
               command="python scripts/generate_report.py "
                       "--input {input} --output {output}",
               inputs=["{input}"],
               outputs=["{output}"]
           )
       ]
   )

   # Run with materialization for reproducibility
   materializer = Materializer(
       workflow=workflow,
       lockfile="workflows/lockfile.lock"
   )

   result = materializer.run(
       inventory="data/fmu_inventory.geojson",
       config="config/instance.yaml",
       cost_config="config/costing.yaml"
   )

   # Check results
   if result.success:
       print("Workflow completed successfully")
       print(f"Report: {result.outputs['generate_report']}")
   else:
       print(f"Workflow failed: {result.error}")

Best Practices
--------------

1. **Define workflows as code**: Don't rely on interactive sessions
2. **Version control everything**: Workflows, configurations, data
3. **Use materialization**: Lock dependencies and cache intermediates
4. **Test workflows on small data**: Validate before running on full data
5. **Document assumptions**: Record why each step exists
6. **Monitor and log**: Track workflow execution for debugging
7. **Reuse steps**: Package common steps as libraries

Exercises
---------

**Exercise 1 (Easy)**: Define a simple FreshForge workflow with three
steps: data preparation, simulation, and report generation.

**Exercise 2 (Medium)**: Extend the workflow to include error handling
and logging. Add a step that validates the simulation output.

**Exercise 3 (Hard)**: Create a reusable step library with common
operations (data loading, instance building, cost curve generation)
and use it to build a complex multi-FMU analysis workflow.

Further Reading
---------------

- :doc:`ch11_femic_models` — Building models with FEMIC
- :doc:`ch12_fhops_integration` — Using fhops for harvest cost curves
- FreshForge documentation: https://freshforge.readthedocs.io
- UBC-FRESH ecosystem overview