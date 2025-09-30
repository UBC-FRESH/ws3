#!/bin/bash
set -euo pipefail

# Repro script for EMS manuscript figures/tables
# - Creates a venv under ./.ems-venv
# - Installs pinned deps
# - Generates diagrams (architecture, workflow, graphical abstract)
# - Generates the spatial allocation map used in the manuscript
# - Runs the case study script to generate results figures and tables

PYBIN=${PYBIN:-python3}
VENV_DIR=.ems-venv

echo "[1/6] Creating virtual environment at ${VENV_DIR}"
${PYBIN} -m venv ${VENV_DIR}
source ${VENV_DIR}/bin/activate

pip install --upgrade pip setuptools wheel

echo "[2/6] Installing requirements"
pip install -r papers/ems/repro/requirements.txt

# If using a local checkout, ensure it is importable over pip ws3
# pip install -e .

echo "[3/6] Generating diagrams"
${PYBIN} papers/ems/repro/generate_diagrams.py

echo "[4/6] Generating spatial allocation map"
${PYBIN} papers/ems/repro/generate_spatial_allocation.py

echo "[5/6] Generating Example 040 figure"
${PYBIN} papers/ems/repro/generate_example040_assets.py

echo "[6/6] Running case study"
${PYBIN} papers/ems/repro/generate_case_study.py

echo "[Done] Outputs written under papers/ems/figs and papers/ems/tables"
