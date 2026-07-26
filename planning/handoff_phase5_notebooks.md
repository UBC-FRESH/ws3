# Phase 5 Task 5.2 Handoff: Interactive Jupyter Notebooks

**Status**: Mostly Complete  
**Branch**: `feature/ws3-phase5`  
**Last Updated**: 2026-07-26  

## What Was Accomplished

### Completed
1. ✅ Created `examples/070_ws3_quickstart_complete_workflow.ipynb`
   - End-to-end ws3 workflow from data loading to optimization results
   - 16 properly structured cells (4 markdown, 12 code)
   - Covers: environment setup, data preparation, model creation, optimization, visualization, export

2. ✅ Created `examples/071_ws3_scenario_analysis_and_comparison.ipynb`
   - Scenario analysis with multiple objectives (even-flow, maximize, minimize)
   - Sensitivity analysis on planning horizon
   - Trade-off visualization between competing objectives
   - Composite scoring for scenario ranking
   - 11 major sections with multiple cells each

3. ✅ Created `scripts/create_notebook.py`
   - Helper script for creating properly structured notebooks using nbformat
   - Demonstrates best practices for notebook creation

4. ✅ Created `examples/072_ws3_carbon_accounting_with_libcbm.ipynb`
   - Integrate libCBM for detailed carbon pool modeling
   - Show carbon sequestration vs. harvest trade-offs
   - Demonstrate carbon budget calculations
   - 25+ cells covering full carbon accounting workflow

5. ✅ Created `examples/073_ws3_spatial_constraints.ipynb`
   - Add adjacency constraints
   - Contiguous area requirements
   - Spatial connectivity analysis
   - 20+ cells with rasterization and connectivity metrics

6. ✅ Created `examples/074_ws3_multi_objective_optimization.ipynb`
   - Pareto-optimal solutions
   - Weighted objective functions
   - Goal programming approaches
   - 20+ cells with multi-objective comparison

7. ✅ Created `examples/075_ws3_parallel_optimization.ipynb`
   - Multi-core solver utilization
   - Parameter sweeps
   - Performance benchmarking
   - 20+ cells with speedup/efficiency analysis

### Issues Encountered
- Initial notebook creation attempts resulted in single-cell notebooks (all content in one cell)
- Solution: Used Python nbformat library to create properly structured multi-cell notebooks
- Scripts moved to `scripts/` directory to keep repo root clean

## Remaining Work

### Task 5.2 Subtasks

1. **Additional Interactive Notebooks** (Target: 5+ total) ✅ COMPLETE
   - [x] `072_ws3_carbon_accounting_with_libcbm.ipynb`
   - [x] `073_ws3_spatial_constraints.ipynb`
   - [x] `074_ws3_multi_objective_optimization.ipynb`
   - [x] `075_ws3_parallel_optimization.ipynb`

2. **FAQ Section** 
   - [ ] Create `docs/source/howto/faq.md`
   - [ ] Document top 20 common user questions
   - [ ] Include troubleshooting steps
   - [ ] Add code examples for common errors

3. **Migration Guide**
   - [ ] Create `docs/source/howto/migration_from_woodstock.md`
   - [ ] Document differences between Woodstock and ws3
   - [ ] Provide conversion scripts
   - [ ] Show side-by-side comparisons

## Technical Context

### Notebook Structure Requirements
- Each section should be a separate markdown cell with explanatory text
- Code cells should be focused and runnable independently
- Use `%load_ext autoreload` and `%autoreload 2` for development
- Include error handling and troubleshooting sections
- Export results to CSV for further analysis

### Data Files Used
- `data/shp/tsa24_clipped.shp/stands.shp` - Forest inventory (500 stands)
- `data/au_table.csv` - Analysis unit definitions
- `data/curve_table.csv` - Yield curve metadata
- `data/curve_points_table.csv` - Yield curve data points
- `data/woodstock_model_files_tsa24_clipped/` - Woodstock model files

### Key Libraries
- `ws3.forest.ForestModel` - Main model class
- `ws3.core` - Core optimization functions
- `pandas` - Data manipulation
- `geopandas` - Spatial data handling
- `matplotlib` - Visualization
- `numpy` - Numerical computations

### Common Patterns
```python
# Standard imports
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import ws3.forest
from util import run_scenario

# Model creation pattern
fm = ws3.forest.ForestModel(
    model_name="example",
    model_path="data/woodstock_model_files_tsa24_clipped",
    base_year=2020,
    horizon=10,
    period_length=10,
    max_age=1000
)

fm.import_landscape_section()
fm.import_areas_section(convert_periods_to_years=period_length)
fm.import_yields_section(convert_periods_to_years=period_length)
fm.import_actions_section(convert_periods_to_years=period_length)
fm.import_transitions_section(convert_periods_to_years=period_length)
fm.initialize_areas()
fm.add_null_action()
fm.reset_actions()
fm.actions["harvest"].is_harvest = True

# Scenario running
problem = run_scenario(fm, scenario_name="base", workers=1)
solution = problem.solve(solver="gurobi")
```

## Next Steps for New Session

1. **Test Existing Notebooks**
   - Verify all 7 notebooks (070-075) run correctly
   - Fix any import or data path issues
   - Ensure all visualizations render properly

2. **Create FAQ Section**
   - Review existing troubleshooting guides
   - Identify common pain points from GitHub issues
   - Write clear, concise Q&A with code examples

3. **Create Migration Guide**
   - Document Woodstock vs. ws3 differences
   - Provide conversion utilities
   - Show practical examples

4. **Update Documentation**
   - Add notebook links to README
   - Update getting started guide
   - Create notebook index page

5. **Commit and Push**
   - Commit each notebook separately with clear messages
   - Push to `feature/ws3-phase5` branch
   - Update GitHub issue #60 with progress

## Success Criteria for Task 5.2

- [x] 5+ interactive notebooks available in `examples/` (7 total: 070-075)
- [ ] Each notebook is self-contained and runnable (needs testing)
- [ ] FAQ section addresses top 20 user questions
- [ ] Migration guide helps users convert Woodstock models
- [ ] All notebooks tested and verified to work
- [ ] Documentation updated with notebook links

## References

- Phase 5 Issue: https://github.com/UBC-FRESH/ws3/issues/60
- Phase 5 Roadmap: `planning/phase5_roadmap.md`
- Phase 4 Closeout: `planning/phase4_closeout_summary.md`
- Existing Notebooks: `examples/010_ws3_model_example-fromscratch.ipynb`
- Documentation: `docs/source/getting_started/`

## Notes for Continuation

- The notebooks use sample data from TSA 24 in British Columbia
- All notebooks should work with the existing test data
- Use `scripts/create_notebook.py` as a template for creating new notebooks
- Follow the cell structure pattern: markdown explanation → code → output
- Include error handling and common troubleshooting in each notebook
- Export results to CSV for reproducibility
- **Next steps**: Test all notebooks, create FAQ section, create migration guide

- The notebooks use sample data from TSA 24 in British Columbia
- All notebooks should work with the existing test data
- Use `scripts/create_notebook.py` as a template for creating new notebooks
- Follow the cell structure pattern: markdown explanation → code → output
- Include error handling and common troubleshooting in each notebook
- Export results to CSV for reproducibility