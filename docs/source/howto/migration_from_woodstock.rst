.. _howto-migration-from-woodstock:

========================================
Migration Guide: Woodstock to ws3
========================================

This guide helps users migrate models and workflows from Woodstock (R) to
ws3 (Python). While ws3 is designed to be compatible with Woodstock data
formats, there are important differences in API, workflow, and capabilities.

.. contents:: Table of Contents
   :depth: 2
   :local:

Overview
========

Woodstock vs ws3 at a Glance
------------------------------

+------------------+----------------------------------+----------------------------------+
| Aspect           | Woodstock (R)                    | ws3 (Python)                     |
+==================+==================================+==================================+
| Language         | R                                | Python 3.9+                      |
+------------------+----------------------------------+----------------------------------+
| Optimization     | R optimization packages          | PuLP, Gurobi, CBC, GLPK          |
+------------------+----------------------------------+----------------------------------+
| Data Format      | Woodstock-compatible             | Woodstock-compatible (same)      |
+------------------+----------------------------------+----------------------------------+
| Spatial Support  | Limited                          | Full GeoPandas integration       |
+------------------+----------------------------------+----------------------------------+
| Parallelism      | Limited                          | Multi-core, distributed          |
+------------------+----------------------------------+----------------------------------+
| Interactivity    | RMarkdown                        | Jupyter Notebooks                |
+------------------+----------------------------------+----------------------------------+
| Extensibility    | R packages                       | Python packages, SciPy ecosystem |
+------------------+----------------------------------+----------------------------------+

Key Differences
===============

1. **API Design**

   Woodstock uses a functional, pipe-based API:

   .. code-block:: R

      model <- woodstock_model() %>%
        import_landscape("data/landscape.shp") %>%
        import_yields("data/yields.csv") %>%
        optimize(objective = "maximize_npv")

   ws3 uses an object-oriented API:

   .. code-block:: python

      fm = ForestModel(
          model_name="my_model",
          model_path="data/woodstock_model_files"
      )
      fm.import_landscape_section()
      fm.import_yields_section()
      problem = compile_scenario(fm, objective="maximize_npv")
      solution = problem.solve(solver="gurobi")

2. **Development Types**

   Woodstock:

   .. code-block:: R

      dt <- development_type(
          code = "CWHvm1_DWG_1",
          species = "DWG",
          site_index = 1
      )

   ws3:

   .. code-block:: python

      dt_key = ('TSA24', 'CWHvm1', 1, 'DWG', 'curve_001')
      # 5-element tuple: (TSA, THLB, AU, species, yield_curve)

3. **Yield Curves**

   Woodstock:

   .. code-block:: R

      yield_curve <- yield_curve(
          species = "DWG",
          site_index = 1,
          curve_data = data.frame(age = c(10, 20, 30), volume = c(5, 15, 30))
      )

   ws3:

   .. code-block:: python

      # Yield curves are loaded from CSV files
      # Format: species, site_index, age, volume, ...
      fm.import_yields_section(convert_periods_to_years=10)

4. **Actions**

   Woodstock:

   .. code-block:: R

      action <- harvest_action(
          code = "CLEARCUT",
          cost = 45.0,
          revenue = list(sawlog = 120.0, pulpwood = 35.0)
      )

   ws3:

   .. code-block:: python

      # Actions are defined in CSV files
      # Loaded via fm.import_actions_section()
      fm.actions["harvest"].is_harvest = True

Migration Steps
===============

Step 1: Prepare Woodstock Data
-------------------------------

Export your Woodstock model to CSV format:

.. code-block:: R

   # Export inventory
   write.csv(landscape, "data/inventory.csv")

   # Export yield curves
   write.csv(yield_curves, "data/yield_curves.csv")

   # Export actions
   write.csv(actions, "data/actions.csv")

   # Export areas
   write.csv(areas, "data/areas.csv")

Step 2: Convert Data Formats
------------------------------

ws3 expects Woodstock-compatible file formats. Most Woodstock data can be
used directly. If you exported to CSV, ensure:

- **Inventory**: Has columns for development type codes, area, age, volume
- **Yield curves**: Has columns for species, site_index, age, volume
- **Actions**: Has columns for action code, cost, revenue
- **Areas**: Has columns for area ID, boundary geometry

Step 3: Create ForestModel in Python
--------------------------------------

.. code-block:: python

   from ws3.forest import ForestModel

   # Initialize model
   fm = ForestModel(
       model_name="my_model",
       model_path="data/woodstock_model_files",  # Woodstock output directory
       base_year=2020,
       horizon=10,
       period_length=10,
       max_age=1000
   )

   # Import sections
   fm.import_landscape_section()
   fm.import_areas_section(convert_periods_to_years=10)
   fm.import_yields_section(convert_periods_to_years=10)
   fm.import_actions_section(convert_periods_to_years=10)
   fm.import_transitions_section(convert_periods_to_years=10)

   # Initialize
   fm.initialize_areas()
   fm.add_null_action()
   fm.reset_actions()
   fm.actions["harvest"].is_harvest = True

Step 4: Define Objectives
---------------------------

Woodstock:

.. code-block:: R

   objective <- objective_function(
       type = "npv",
       discount_rate = 0.05,
       weights = c(sawlog = 1.0, pulpwood = 0.5)
   )

ws3:

.. code-block:: python

   from ws3.core import compile_scenario

   problem = compile_scenario(
       fm,
       scenario_name="base",
       objective="maximize_npv",
       weights={"npv": 1.0},
       discount_rate=0.05
   )

Step 5: Add Constraints
-------------------------

Woodstock:

.. code-block:: R

   constraint <- even_flow_constraint(
       tolerance = 0.1,
       period_length = 10
   )

ws3:

.. code-block:: python

   # Even-flow constraints are built into compile_scenario
   # For spatial constraints, see spatial-schedule-allocation how-to

   problem = compile_scenario(
       fm,
       scenario_name="base",
       objective="maximize_npv",
       even_flow_tolerance=0.1
   )

Step 6: Solve and Analyze
---------------------------

Woodstock:

.. code-block:: R

   result <- optimize(model, objective, constraints)
   schedule <- result$schedule
   plot(schedule)

ws3:

.. code-block:: python

   solution = problem.solve(solver="gurobi")
   schedule = solution.get_schedule()

   # Visualize
   import matplotlib.pyplot as plt
   plt.bar(schedule['period'], schedule['volume'])
   plt.show()

Common Conversions
===================

Conversion 1: Simple Optimization
-----------------------------------

Woodstock:

.. code-block:: R

   model <- woodstock_model() %>%
     import_landscape("landscape.shp") %>%
     import_yields("yields.csv") %>%
     optimize(objective = "maximize_npv", solver = "gurobi")

ws3:

.. code-block:: python

   fm = ForestModel(
       model_name="model",
       model_path="woodstock_model_files"
   )
   fm.import_landscape_section()
   fm.import_yields_section()

   problem = compile_scenario(fm, objective="maximize_npv")
   solution = problem.solve(solver="gurobi")

Conversion 2: Multi-Objective Optimization
--------------------------------------------

Woodstock:

.. code-block:: R

   model <- woodstock_model() %>%
     optimize(
       objectives = list(
         npv = list(weight = 0.5),
         even_flow = list(weight = 0.3),
         carbon = list(weight = 0.2)
       )
     )

ws3:

.. code-block:: python

   from ws3.opt import multi_objective_optimize

   objectives = [
       {"name": "npv", "weight": 0.5, "direction": "maximize"},
       {"name": "even_flow", "weight": 0.3, "direction": "minimize_deviation"},
       {"name": "carbon", "weight": 0.2, "direction": "maximize"}
   ]

   pareto_front = multi_objective_optimize(
       fm,
       objectives=objectives,
       solver="gurobi"
   )

Conversion 3: Spatial Constraints
-----------------------------------

Woodstock:

.. code-block:: R

   model <- woodstock_model() %>%
     add_adjacency_constraint(min_contiguous = 50) %>%
     optimize()

ws3:

.. code-block:: python

   from ws3.spatial import add_adjacency_constraints

   add_adjacency_constraints(
       problem,
       fm,
       min_contiguous_area=50,
       max_adjacency_violations=0.1
   )

   solution = problem.solve(solver="gurobi")

Conversion 4: Scenario Analysis
---------------------------------

Woodstock:

.. code-block:: R

   scenarios <- list(
     base = list(objective = "maximize_npv"),
     conservation = list(objective = "maximize_carbon"),
     timber = list(objective = "maximize_volume")
   )

   results <- lapply(scenarios, function(s) optimize(model, s$objective))

ws3:

.. code-block:: python

   from ws3.core import compile_scenario

   scenarios = {
       "base": {"objective": "maximize_npv"},
       "conservation": {"objective": "maximize_carbon"},
       "timber": {"objective": "maximize_volume"}
   }

   results = {}
   for name, params in scenarios.items():
       problem = compile_scenario(fm, scenario_name=name, **params)
       results[name] = problem.solve(solver="gurobi")

Troubleshooting
===============

Issue 1: "Development type not found"
---------------------------------------

**Woodstock**: Uses character codes like "CWHvm1_DWG_1"

**ws3**: Uses 5-element tuples like ('TSA24', 'CWHvm1', 1, 'DWG', 'curve_001')

**Solution**: Ensure yield curve keys match the development type structure.

Issue 2: "Yield curve not found"
----------------------------------

**Woodstock**: Automatically matches yield curves by species and site index

**ws3**: Requires explicit key matching with optional wildcards

**Solution**: Use '?' wildcards in yield curve keys:

.. code-block:: python

   # Match any TSA
   mask_key = ('?', 'CWHvm1', 1, 'DWG', 'curve_001')

Issue 3: "Solver not found"
------------------------------

**Woodstock**: Uses R optimization packages

**ws3**: Requires Python solver packages

**Solution**:

.. code-block:: bash

   # For Gurobi
   pip install gurobipy

   # For CBC (includes with PuLP)
   pip install pulp

Issue 4: "Infeasible problem"
-------------------------------

Both Woodstock and ws3 can encounter infeasible problems. Common causes:

- Constraints too restrictive
- Data errors (missing yields, invalid ages)
- Solver numerical issues

**Solution**: Relax constraints, check data, try different solver.

Performance Comparison
======================

ws3 typically outperforms Woodstock in:

1. **Large-scale problems**: Better parallelism and solver integration
2. **Spatial analysis**: Native GeoPandas support
3. **Interactivity**: Jupyter notebooks for exploratory analysis
4. **Extensibility**: Python ecosystem (SciPy, scikit-learn, etc.)

Woodstock may be preferable for:

1. **Legacy workflows**: Existing R-based pipelines
2. **Specific R packages**: If you rely on R-specific tools
3. **Team familiarity**: If your team is more comfortable with R

Conclusion
==========

ws3 provides a modern, Python-based alternative to Woodstock with enhanced
capabilities for spatial analysis, parallel optimization, and interactivity.
The data formats are compatible, making migration straightforward.

For detailed examples, see the Phase 5 notebooks in `examples/`:

- `070_ws3_quickstart_complete_workflow.ipynb`
- `071_ws3_scenario_analysis_and_comparison.ipynb`
- `073_ws3_spatial_constraints.ipynb`
- `074_ws3_multi_objective_optimization.ipynb`
- `075_ws3_parallel_optimization.ipynb`

If you encounter issues during migration, check the :doc:`faq` or file an
issue on GitHub.