#!/usr/bin/env python3
"""
Create a properly structured ws3 quickstart notebook with multiple cells.
"""

import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Cell 1: Title and description (markdown)
nb.cells.append(nbf.v4.new_markdown_cell("""# Quickstart: Complete ws3 Workflow from Data to Results

This notebook demonstrates a complete end-to-end workflow using `ws3`, from loading data to solving an optimization problem and visualizing results. This is designed for users who want a single, self-contained example that covers the entire modelling process.

> **Prerequisites**: Python 3.9+, `ws3` package installed, Jupyter environment with matplotlib available

## What You'll Learn

- How to set up a complete ws3 model from scratch
- How to import and prepare forest inventory data
- How to define yield curves and actions
- How to formulate and solve an optimization problem
- How to visualize and interpret results"""))

# Cell 2: Environment setup (python)
nb.cells.append(nbf.v4.new_code_cell("""%load_ext autoreload
%autoreload 2

# Optionally install ws3 from local source for development
import sys
import os

if '--dev' in sys.argv:
    !pip uninstall -y ws3
    !pip install -e ..

# Install required packages
!pip install pandas geopandas matplotlib numpy"""))

# Cell 3: Import libraries (python)
nb.cells.append(nbf.v4.new_code_cell("""import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import ws3.forest
import ws3.core
import numpy as np"""))

# Cell 4: Define model parameters (python)
nb.cells.append(nbf.v4.new_code_cell("""# Basic model parameters
base_year = 2020
horizon = 10  # number of periods
period_length = 10  # years per period
max_age = 1000  # maximum age for stands
tvy_name = "totvol"  # total volume measure

print(f"Model Parameters:")
print(f"  Base Year: {base_year}")
print(f"  Horizon: {horizon} periods")
print(f"  Period Length: {period_length} years")
print(f"  Max Age: {max_age} years")
print(f"  TVY Measure: {tvy_name}")"""))

# Cell 5: Load forest inventory data (python)
nb.cells.append(nbf.v4.new_code_cell("""# Load stand inventory data
stands_path = "data/shp/tsa24_clipped.shp/stands.shp"
stands = gpd.read_file(stands_path)

print(f"Loaded {len(stands)} stands")
print(f"\\nColumns: {list(stands.columns)}")
print(f"\\nFirst few rows:")
stands.head()"""))

# Cell 6: Load yield curve data (python)
nb.cells.append(nbf.v4.new_code_cell("""# Load AU (Analysis Unit) table
au_table = pd.read_csv("data/au_table.csv").set_index("au_id")

# Load curve table
curve_table = pd.read_csv("data/curve_table.csv")

# Load curve points table
curve_points_table = pd.read_csv("data/curve_points_table.csv").set_index("curve_id")

print(f"Number of Analysis Units: {len(au_table)}")
print(f"Number of Yield Curves: {len(curve_table)}")
print(f"\\nAU Table columns: {list(au_table.columns)}")
print(f"\\nCurve Table columns: {list(curve_table.columns)}")"""))

# Cell 7: Prepare data for ws3 (python)
nb.cells.append(nbf.v4.new_code_cell("""# Add THLB (Timber Harvesting Land Base) attribute to stands
# THLB = 1 if managed, 0 if unmanaged
au_table["thlb"] = au_table.apply(
    lambda row: 0 if row.unmanaged_curve_id == row.managed_curve_id else 1, 
    axis=1
)

# Map THLB to stands
stands["theme1"] = stands.apply(
    lambda row: au_table.loc[row.theme2].thlb, 
    axis=1
)

# Add yield curve ID to theme4 for tracking
stands["theme4"] = stands.curve1

print(f"Stands with THLB=1 (managed): {(stands.theme1 == 1).sum()}")
print(f"Stands with THLB=0 (unmanaged): {(stands.theme1 == 0).sum()}")"""))

# Cell 8: Create ForestModel object (python)
nb.cells.append(nbf.v4.new_code_cell("""# Create ForestModel instance
fm = ws3.forest.ForestModel(
    model_name="quickstart_example",
    model_path="data/woodstock_model_files_tsa24_clipped",
    base_year=base_year,
    horizon=horizon,
    period_length=period_length,
    max_age=max_age
)

# Import model sections
fm.import_landscape_section()
fm.import_areas_section(convert_periods_to_years=period_length)
fm.import_yields_section(convert_periods_to_years=period_length)
fm.import_actions_section(convert_periods_to_years=period_length)
fm.import_transitions_section(convert_periods_to_years=period_length)

# Initialize areas and add actions
fm.initialize_areas()
fm.add_null_action()
fm.reset_actions()

# Mark harvest action
fm.actions["harvest"].is_harvest = True

print(f"Model created: {fm.model_name}")
print(f"Number of areas: {len(fm.areas)}")
print(f"Available actions: {list(fm.actions.keys())}")"""))

# Cell 9: Formulate optimization problem (python)
nb.cells.append(nbf.v4.new_code_cell("""# Import optimization utilities
from util import run_scenario

# Run a basic scenario
problem = run_scenario(
    fm, 
    scenario_name="base-cgen_gs", 
    print_df=True, 
    workers=1
)

print(f"Problem formulated successfully")
print(f"Objective: {problem.objective}")"""))

# Cell 10: Solve the problem (python)
nb.cells.append(nbf.v4.new_code_cell("""# Solve the problem
solution = problem.solve(solver="gurobi")

print(f"Solving complete!")
print(f"Optimal value: {solution.objective_value:.2f}")
print(f"Solver status: {solution.status}")"""))

# Cell 11: Extract and visualize results (python)
nb.cells.append(nbf.v4.new_code_cell("""# Extract harvest volumes by period
harvest_volumes = solution.get_variable_values("x")

# Aggregate by period
period_volumes = {}
for au_id, action, period, volume in harvest_volumes:
    if action == "harvest":
        period_volumes[period] = period_volumes.get(period, 0) + volume

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Harvest volume by period
periods = sorted(period_volumes.keys())
volumes = [period_volumes[p] for p in periods]

axes[0].bar(range(len(periods)), volumes)
axes[0].set_xlabel('Period')
axes[0].set_ylabel('Harvest Volume')
axes[0].set_title('Harvest Volume by Period')
axes[0].set_xticks(range(len(periods)))
axes[0].set_xticklabels([f'P{p+1}' for p in periods])

# Plot 2: Even-flow deviation
even_flow_target = min(volumes)
deviations = [v - even_flow_target for v in volumes]

axes[1].bar(range(len(periods)), deviations)
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_xlabel('Period')
axes[1].set_ylabel('Deviation from Minimum')
axes[1].set_title('Even-Flow Deviation')
axes[1].set_xticks(range(len(periods)))
axes[1].set_xticklabels([f'P{p+1}' for p in periods])

plt.tight_layout()
plt.show()"""))

# Cell 12: Analyze results (python)
nb.cells.append(nbf.v4.new_code_cell("""# Get solution statistics
print("Solution Statistics:")
print(f"  Objective Value: {solution.objective_value:.2f}")
print(f"  Solver Status: {solution.status}")
print(f"  Number of Variables: {problem.n_vars}")
print(f"  Number of Constraints: {problem.n_constraints}")

# Check if solution is feasible
if solution.status == "OPTIMAL":
    print("\\n✓ Solution is optimal!")
else:
    print(f"\\n⚠ Solution status: {solution.status}")"""))

# Cell 13: Export results (python)
nb.cells.append(nbf.v4.new_code_cell("""# Export harvest schedule to CSV
harvest_schedule = []
for au_id, action, period, volume in harvest_volumes:
    if action == "harvest":
        harvest_schedule.append({
            'Area': au_id,
            'Action': action,
            'Period': period,
            'Volume': volume
        })

schedule_df = pd.DataFrame(harvest_schedule)
schedule_df.to_csv("harvest_schedule.csv", index=False)

print(f"Exported {len(schedule_df)} harvest actions to harvest_schedule.csv")
schedule_df.head()"""))

# Cell 14: Summary (markdown)
nb.cells.append(nbf.v4.new_markdown_cell("""## Summary

In this notebook, you learned how to:

1. ✓ Set up a ws3 modelling environment
2. ✓ Load and prepare forest inventory data
3. ✓ Import yield curves and define actions
4. ✓ Create a ForestModel object
5. ✓ Formulate an optimization problem
6. ✓ Solve the problem with Gurobi
7. ✓ Visualize and interpret results
8. ✓ Export results for further analysis

## Next Steps

Now that you've completed a basic workflow, you can explore:

- **Advanced optimization**: Try different objective functions (maximize revenue, minimize carbon emissions, etc.)
- **Spatial constraints**: Add adjacency and contiguous area constraints
- **Multiple objectives**: Use multi-objective optimization to balance competing goals
- **Scenario analysis**: Compare different management scenarios
- **Carbon accounting**: Integrate libCBM for detailed carbon pool modeling"""))

# Cell 15: Troubleshooting (markdown)
nb.cells.append(nbf.v4.new_markdown_cell("""## Troubleshooting

**Common Issues:**

1. **Solver not found**: Make sure Gurobi or GLPK is installed and accessible
   ```bash
   # Check Gurobi
   gurobi_cl --version
   
   # Check GLPK
   glpsol --version
   ```

2. **Memory errors**: For large models, try reducing the horizon or using parallel solving
   ```python
   # Use parallel solving
   solution = problem.solve(solver="gurobi", workers=4)
   ```

3. **Infeasible solution**: Check your constraints and data
   - Verify all areas have valid yield curves
   - Check that harvest actions are properly defined
   - Ensure age constraints are realistic"""))

# Cell 16: References (markdown)
nb.cells.append(nbf.v4.new_markdown_cell("""## References

- [ws3 Documentation](https://ws3.readthedocs.io)
- [Woodstock Documentation](https://woodstock.sourceforge.net)
- [libCBM Documentation](https://libcbm.readthedocs.io)
- [Gurobi Documentation](https://www.gurobi.com/documentation/)

## Acknowledgments

This notebook uses sample data from Timber Supply Area 24 in British Columbia, Canada. The data is derived from publicly-available BC Vegetation Resource Inventory (VRI) datasets."""))

# Save the notebook
with open('examples/070_ws3_quickstart_complete_workflow.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully: examples/070_ws3_quickstart_complete_workflow.ipynb")
print(f"Total cells: {len(nb.cells)}")
print(f"  Markdown cells: {sum(1 for c in nb.cells if c.cell_type == 'markdown')}")
print(f"  Code cells: {sum(1 for c in nb.cells if c.cell_type == 'code')}")