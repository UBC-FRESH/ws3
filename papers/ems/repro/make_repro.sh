#!/bin/bash
set -euo pipefail

# Repro script for EMS manuscript figures/tables
# - Creates a venv under ./.ems-venv
# - Installs pinned deps
# - Generates diagrams (architecture, workflow, graphical abstract)
# - Generates a spatial schematic stub (will be replaced by final map later)
# - Runs the case study script to generate results figures and tables

PYBIN=${PYBIN:-python3}
VENV_DIR=.ems-venv

echo "[1/5] Creating virtual environment at ${VENV_DIR}"
${PYBIN} -m venv ${VENV_DIR}
source ${VENV_DIR}/bin/activate

pip install --upgrade pip setuptools wheel

echo "[2/5] Installing requirements"
pip install -r papers/ems/repro/requirements.txt

# If using a local checkout, ensure it is importable over pip ws3
# pip install -e .

echo "[3/5] Generating diagrams"
${PYBIN} papers/ems/repro/generate_diagrams.py

echo "[4/5] Generating spatial stub"
${PYBIN} papers/ems/repro/generate_spatial_stub.py

echo "[4.5/5] Generating Example 040 figure"
${PYBIN} papers/ems/repro/generate_example040_assets.py

echo "[5/5] Running case study"
${PYBIN} papers/ems/repro/generate_case_study.py

echo "[Done] Outputs written under papers/ems/figs and papers/ems/tables"
