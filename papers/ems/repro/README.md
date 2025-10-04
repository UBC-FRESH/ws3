Reproducibility package for EMS manuscript (WS3 + libCBM case study)

Overview
- This package reproduces the EMS manuscript case study based on the existing example notebook 031_ws3_libcbm_sequential-builtin.ipynb, without modifying the scientific workflow.
- It creates the figures and tables referenced in the paper from a deterministic script.

Contents
- requirements.txt: minimal pinned dependencies to run the case study
- make_repro.sh: end-to-end script to create a virtual environment, install deps, and generate outputs
- generate_case_study.py: Python script that executes the case study workflow
- generate_spatial_allocation.py: spatial allocation reproduction matching the manuscript example
- style.py: plotting style helper (uses FRESH palette if available)

Optional scalability benchmarks
- A larger public dataset (five non-overlapping TSAs in NE BC) can be fetched with DataLad to run deterministic scaling tests.
- Enable via `RUN_SCALING=1 bash papers/ems/repro/make_repro.sh`. The first run performs `datalad install -r -g -s https://github.com/UBC-FRESH/cccandies_demo_input papers/ems/repro/data/cccandies_demo_input`.
- `papers/ems/repro/run_scaling_benchmarks.py` now bootstraps WS3 directly from the dataset (no Woodstock sections required), sorts TSAs by complexity, and evaluates single TSA and cumulative “mash-up” combinations.
- Each combination executes the heuristic scheduler in deterministic single-core mode and records sequential spatial allocation timings. Setting `RUN_LP=1` adds Model~I LP benchmarks that build the problem with 1 and 16 workers (parallel coefficient compilation) and solve with matching HiGHS thread counts.
- Spatial allocation uses `ForestRaster` to generate (and immediately clean up) disturbance rasters; results and diagnostics are written to `papers/ems/tables/perf_scaling.csv`, including optional LP build/solve metrics and memory footprints.
- `generate_scaling_figures.py` rebuilds the manuscript scaling plots directly from `perf_scaling.csv` whenever that file is present.

Outputs (created)
- papers/ems/figs/
  - f3_spatial_allocation.png
  - f4a_harvest_and_stock.png
  - f4b_carbon_stocks.png
- papers/ems/tables/
  - scenario_flows.csv (columns: period, harvest_area_ha, harvest_volume_m3, growing_stock_m3)
  - annual_carbon_stocks.csv

How to run (Linux, bash)
1) From repository root, run:

   bash papers/ems/repro/make_repro.sh

2) Results will be written under papers/ems/figs and papers/ems/tables.

Notes
- The workflow uses the exact data and logic from examples/031_ws3_libcbm_sequential-builtin.ipynb.
- If you have issues with libcbm installation on your platform, consider a clean virtual environment and ensure system-level build tools are available.
