#!/bin/bash
# Validation script for ws3 code quality
# This script runs all quality checks and exits with appropriate error codes

set -e

echo "==================================="
echo "ws3 Code Quality Validation"
echo "==================================="
echo ""

# Check if in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Not in virtual environment"
    echo "Consider activating .venv-mcp or your preferred venv"
    echo ""
fi

# Run mypy
echo "Running mypy..."
python3 -m mypy ws3/ --ignore-missing-imports --strict
if [ $? -eq 0 ]; then
    echo "✓ mypy passed"
else
    echo "✗ mypy failed"
    exit 1
fi
echo ""

# Run ruff
echo "Running ruff..."
python3 -m ruff check ws3/
if [ $? -eq 0 ]; then
    echo "✓ ruff check passed"
else
    echo "✗ ruff check failed"
    exit 1
fi

python3 -m ruff format --check ws3/
if [ $? -eq 0 ]; then
    echo "✓ ruff format passed"
else
    echo "✗ ruff format failed"
    exit 1
fi
echo ""

# Run tests
echo "Running tests..."
python3 -m pytest tests/ -v
if [ $? -eq 0 ]; then
    echo "✓ tests passed"
else
    echo "✗ tests failed"
    exit 1
fi
echo ""

echo "==================================="
echo "All validation checks passed!"
echo "==================================="