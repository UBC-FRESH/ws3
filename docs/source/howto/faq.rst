.. _howto-faq:

==================================
Frequently Asked Questions (FAQ)
==================================

This document answers the most common questions about using ws3. If your
question isn't answered here, check the :doc:`migration_from_woodstock` guide
or file an issue on GitHub.

.. contents:: Table of Contents
   :depth: 2
   :local:

Common Setup Questions
======================

Q1: How do I install ws3?
--------------------------

.. code-block:: bash

   # From source (recommended for development)
   git clone https://github.com/UBC-FRESH/ws3.git
   cd ws3
   pip install -e .

   # With development dependencies
   pip install -e ".[dev]"

Q2: What Python version does ws3 support?
------------------------------------------

ws3 requires Python 3.9 or higher. We recommend Python 3.10+ for best
compatibility with optimization solvers.

Q3: Which solvers are supported?
---------------------------------

ws3 supports multiple optimization solvers:

- **Gurobi** (recommended, commercial)
- **CBC** (open-source, included with PuLP)
- **GLPK** (open-source)
- **XPRESS** (commercial)

Install solver-specific packages as needed:

.. code-block:: bash

   # PuLP includes CBC
   pip install pulp

   # Gurobi requires a license
   pip install gurobipy

Q4: How do I verify my installation?
--------------------------------------

.. code-block:: python

   import ws3
   print(ws3.__version__)

   # Run a quick test
   from ws3.forest import ForestModel
   fm = ForestModel(model_name="test", model_path="data/woodstock_model_files_tsa24_clipped")
   fm.import_landscape_section()
   print("Installation OK!")

Common Data Questions
=====================

Q5: What data format does ws3 expect?
---------------------------------------

ws3 expects Woodstock-compatible data:

- **Inventory**: Shapefile or GeoJSON with development type codes
- **Yield curves**: CSV with columns: species, site_index, age, volume, ...
- **Actions**: CSV defining harvest treatments
- **Areas**: CSV defining analysis unit boundaries

See :doc:`data-preparation` for detailed format requirements.

Q6: How do I convert Woodstock data to ws3 format?
----------------------------------------------------

ws3 is designed to be compatible with Woodstock data formats. In most cases,
you can use Woodstock data directly. See :doc:`migration_from_woodstock` for
detailed conversion guidance.

Q7: What is a development type (DT) key?
------------------------------------------

A development type key is a 5-element tuple:

.. code-block:: python

   dt_key = (TSA_code, THLB_code, AU_code, species_code, yield_curve_id)

Example:

.. code-block:: python

   dt_key = ('TSA24', 'CWHvm1', 1, 'DWG', 'curve_001')

The DT key uniquely identifies a forest compartment in the model.

Q8: How do DT key masks work?
-------------------------------

Yield curve keys can use ``'?'`` wildcards to apply to multiple values:

.. code-block:: python

   # Matches any TSA
   mask_key = ('?', 'CWHvm1', 1, 'DWG', 'curve_001')

   # Matches any species
   mask_key = ('TSA24', 'CWHvm1', 1, '?', 'curve_001')

When loading yields, masks are matched against concrete keys to build the
yield table.

Common Modeling Questions
=========================

Q9: How do I create a ForestModel?
------------------------------------

.. code-block:: python

   from ws3.forest import ForestModel

   fm = ForestModel(
       model_name="my_model",
       model_path="data/woodstock_model_files_tsa24_clipped",
       base_year=2020,
       horizon=10,
       period_length=10,
       max_age=1000
   )

   # Import all sections
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

Q10: How do I run an optimization?
------------------------------------

.. code-block:: python

   from ws3.core import compile_scenario

   # Compile scenario
   problem = compile_scenario(
       fm,
       scenario_name="base",
       objective="maximize_npv",
       weights={"npv": 1.0}
   )

   # Solve
   solution = problem.solve(solver="gurobi")

   # Get results
   schedule = solution.get_schedule()
   print(schedule.head())

Q11: How do I add spatial constraints?
----------------------------------------

.. code-block:: python

   from ws3.spatial import add_adjacency_constraints

   # Add adjacency constraints
   add_adjacency_constraints(
       problem,
       fm,
       min_contiguous_area=50,  # hectares
       max_adjacency_violations=0.1  # 10% tolerance
   )

   # Solve with spatial constraints
   solution = problem.solve(solver="gurobi")

Q12: How do I run multi-objective optimization?
-------------------------------------------------

.. code-block:: python

   from ws3.opt import multi_objective_optimize

   # Define objectives
   objectives = [
       {"name": "npv", "weight": 0.5, "direction": "maximize"},
       {"name": "even_flow", "weight": 0.3, "direction": "minimize_deviation"},
       {"name": "carbon", "weight": 0.2, "direction": "maximize"}
   ]

   # Run multi-objective optimization
   pareto_front = multi_objective_optimize(
       fm,
       objectives=objectives,
       horizon=10,
       solver="gurobi"
   )

   # Visualize trade-offs
   pareto_front.plot_trade_offs()

Common Error Messages
=====================

Q13: "ModuleNotFoundError: No module named 'ws3'"
--------------------------------------------------

**Cause**: ws3 is not installed or not in your Python path.

**Solution**:

.. code-block:: bash

   # Check if installed
   pip show ws3

   # If not installed, install it
   pip install -e /path/to/ws3

   # If installed but not found, check Python path
   python -c "import sys; print(sys.path)"

Q14: "KeyError: 'development_type' not in development types"
--------------------------------------------------------------

**Cause**: Yield curve references a development type that doesn't exist in
the inventory.

**Solution**:

.. code-block:: python

   # Check available development types
   print(fm.development_types.columns)

   # Check yield curve keys
   print(fm.yields.keys())

   # Ensure yield curve keys match inventory
   # Use '?' wildcards for flexible matching

Q15: "Solver not found: gurobi"
---------------------------------

**Cause**: Gurobi solver is not installed or licensed.

**Solution**:

.. code-block:: bash

   # Install Gurobi (requires license)
   pip install gurobipy

   # Or use open-source solver
   pip install pulp  # includes CBC

   # Specify solver in solve()
   solution = problem.solve(solver="cbc")

Q16: "Infeasible problem"
--------------------------

**Cause**: Constraints are too restrictive or conflicting.

**Solution**:

1. **Relax constraints**: Reduce even-flow tolerance, increase adjacency
   violation tolerance
2. **Check data**: Ensure inventory data is valid and complete
3. **Simplify**: Remove constraints one at a time to identify the culprit
4. **Increase horizon**: Longer planning horizons provide more flexibility

Q17: "Optimization took too long"
-----------------------------------

**Cause**: Problem is too large or solver is slow.

**Solution**:

1. **Use parallel solving**: See :doc:`parallel-optimization`
2. **Reduce problem size**: Use a subset of the landscape
3. **Tune solver parameters**: Adjust MIP gap, time limits
4. **Use warm start**: Provide initial solution

Q18: "Results don't match Woodstock output"
---------------------------------------------

**Cause**: Differences in solver, formulation, or data processing.

**Solution**:

1. **Use same solver**: Run Woodstock and ws3 with the same solver
2. **Check data**: Ensure data is processed identically
3. **Verify objectives**: Check objective function coefficients
4. **Review constraints**: Compare constraint formulations

See :doc:`migration_from_woodstock` for detailed comparison guidance.

Advanced Questions
==================

Q19: How do I add custom growth functions?
--------------------------------------------

See :doc:`custom-growth-function` for detailed instructions.

Q19: How do I define custom actions?
--------------------------------------

See :doc:`action-definition` for detailed instructions.

Q20: How do I create custom area selectors?
---------------------------------------------

See :doc:`custom-area-selector` for detailed instructions.

Troubleshooting Checklist
==========================

If you encounter an error, follow this checklist:

1. **Check installation**

   .. code-block:: bash

      pip show ws3
      python -c "import ws3; print(ws3.__version__)"

2. **Check data files**

   .. code-block:: python

      # Verify files exist
      import os
      print(os.path.exists("data/woodstock_model_files_tsa24_clipped"))

3. **Check development types**

   .. code-block:: python

      print(fm.development_types.shape)
      print(fm.development_types.head())

4. **Check yield curves**

   .. code-block:: python

      print(len(fm.yields))
      print(list(fm.yields.keys())[:5])

5. **Check actions**

   .. code-block:: python

      print(fm.actions.keys())
      print(fm.actions["harvest"].is_harvest)

6. **Check solver**

   .. code-block:: bash

      python -c "import gurobipy; print('Gurobi OK')"
      python -c "from pulp import PULP_CBC_CMD; print('CBC OK')"

Getting More Help
==================

- **Documentation**: https://ws3.readthedocs.io
- **GitHub Issues**: https://github.com/UBC-FRESH/ws3/issues
- **Examples**: `examples/` directory in the repository
- **Notebooks**: `examples/070_ws3_quickstart_complete_workflow.ipynb`

When asking for help, please include:

1. ws3 version: ``ws3.__version__``
2. Python version: ``python --version``
3. Solver: Which solver you're using
4. Error message: Full traceback
5. Minimal reproducible example: Smallest code that triggers the error