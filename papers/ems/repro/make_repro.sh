#!/bin/bash
set -euo pipefail

# Repro script for EMS manuscript figures/tables
# - Creates a venv under ./.ems-venv
# - Installs pinned deps
# - Runs the case study script to generate figures/tables

PYBIN=${PYBIN:-python3}
VENV_DIR=.ems-venv

echo "[1/4] Creating virtual environment at ${VENV_DIR}"
${PYBIN} -m venv ${VENV_DIR}
source ${VENV_DIR}/bin/activate

pip install --upgrade pip setuptools wheel

echo "[2/4] Installing requirements"
pip install -r papers/ems/repro/requirements.txt

# If using a local checkout, ensure it is importable over pip ws3
# pip install -e .

echo "[3/4] Running case study"
${PYBIN} papers/ems/repro/generate_case_study.py

echo "[4/4] Outputs written under papers/ems/figs and papers/ems/tables"
