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

Outputs (created)
- papers/ems/figs/
  - f3_spatial_allocation.png
  - f4a_harvest_and_stock.png
  - f4b_carbon_stocks.png
- papers/ems/tables/
  - scenario_flows.csv
  - annual_carbon_stocks.csv

How to run (Linux, bash)
1) From repository root, run:

   bash papers/ems/repro/make_repro.sh

2) Results will be written under papers/ems/figs and papers/ems/tables.

Notes
- The workflow uses the exact data and logic from examples/031_ws3_libcbm_sequential-builtin.ipynb.
- If you have issues with libcbm installation on your platform, consider a clean virtual environment and ensure system-level build tools are available.
