# v1.1.0a1 Smoke Test Plan

**Purpose**: Validate v1.1.0a1 alpha release before wider student/researcher distribution.  
**Date**: 2026-07-26  
**Status**: Ready for execution  

---

## Overview

v1.1.0a1 introduces three new modules (`advanced_modeling`, `perf`, `integration`), eight new notebooks (070-078), four new how-to guides, and two new textbook chapters. This plan defines the smoke test protocol for student and research collaborators to validate the alpha before it becomes the stable release.

---

## Test Phases

### Phase 1: Import Validation (5 minutes)

Verify the package installs and all new modules are importable.

```bash
# Install from source
cd /path/to/ws3
pip install -e ".[docs]"

# Verify version and imports
python -c "
import ws3
print(f'ws3 version: {ws3.__version__}')
assert ws3.__version__ == '1.1.0a1', f'Expected 1.1.0a1, got {ws3.__version__}'

# Verify new modules are importable
from ws3 import advanced_modeling, perf, integration
print('All new modules importable: OK')

# Verify key classes
from ws3.advanced_modeling import StochasticOptimizer, MultiObjectiveOptimizer, DynamicPlanner, ClimateScenarioManager
from ws3.perf import SolverTuner, MemoryProfiler, PerformanceBenchmark, ResultCache, IncrementalSolver
from ws3.integration import FHOPSIntegrator, FEMICIntegrator, FreshForgeIntegrator, SpaDESIntegrator, RESTAPIServer
print('All key classes importable: OK')
"
```

**Expected**: All assertions pass, no import errors.  
**Pass Criteria**: Version is `1.1.0a1`, all new modules and classes import successfully.

---

### Phase 2: Module Instantiation (15 minutes)

Create instances of each new class and call a basic method to verify they don't crash on initialization.

```python
# advanced_modeling.py
from ws3.opt import Problem
from ws3.advanced_modeling import *

# Create a simple problem
problem = Problem()
for i in range(10):
    problem.add_variable(f"x{i}", "continuous", 0, 100)
problem.set_objective({f"x{i}": 1.0 for i in range(10)})

# Test each class
stoch = StochasticOptimizer(problem)
print(f"StochasticOptimizer created: {stoch}")

multi = MultiObjectiveOptimizer(problem)
print(f"MultiObjectiveOptimizer created: {multi}")

dynamic = DynamicPlanner(problem, n_periods=3)
print(f"DynamicPlanner created: {dynamic}")

climate = ClimateScenarioManager()
print(f"ClimateScenarioManager created: {climate}")

# perf.py
tuner = SolverTuner(problem, solver='highs')
print(f"SolverTuner created: {tuner}")

profiler = MemoryProfiler()
print(f"MemoryProfiler created: {profiler}")

bench = PerformanceBenchmark()
print(f"PerformanceBenchmark created: {bench}")

cache = ResultCache()
print(f"ResultCache created: {cache}")

inc_solver = IncrementalSolver(problem)
print(f"IncrementalSolver created: {inc_solver}")

# integration.py
fhops = FHOPSIntegrator()
print(f"FHOPSIntegrator created: {fhops}")

femic = FEMICIntegrator()
print(f"FEMICIntegrator created: {femic}")

freshforge = FreshForgeIntegrator()
print(f"FreshForgeIntegrator created: {freshforge}")

spades = SpaDESIntegrator()
print(f"SpaDESIntegrator created: {spades}")

# RESTAPIServer - don't start, just instantiate
rest = RESTAPIServer(host='127.0.0.1', port=8765)
print(f"RESTAPIServer created: {rest}")

print("\nAll classes instantiated successfully: OK")
```

**Expected**: All classes instantiate without errors.  
**Pass Criteria**: No exceptions during instantiation.

---

### Phase 3: Notebook Execution (30 minutes)

Run each Phase 5 notebook in a clean environment. Use `nbconvert` to execute and capture output.

```bash
# Execute all Phase 5 notebooks
for nb in examples/070_ws3_quickstart_complete_workflow.ipynb \
          examples/071_ws3_scenario_analysis_and_comparison.ipynb \
          examples/073_ws3_spatial_constraints.ipynb \
          examples/074_ws3_multi_objective_optimization.ipynb \
          examples/075_ws3_parallel_optimization.ipynb \
          examples/076_ws3_performance_optimization.ipynb \
          examples/077_ws3_integration_examples.ipynb \
          examples/078_ws3_advanced_modeling.ipynb; do
    echo "Testing: $nb"
    jupyter nbconvert --to notebook --execute --inplace "$nb" 2>&1 | tail -5
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "  PASS"
    else
        echo "  FAIL"
    fi
done
```

**Expected**: All notebooks execute without kernel errors.  
**Pass Criteria**: Exit code 0 for all notebooks. Note: some cells may fail due to missing data files (expected — document which ones).

---

### Phase 4: Package Distribution (10 minutes)

Build the package and verify it can be installed from the built artifact.

```bash
# Build
python -m build

# Check
twine check dist/*

# Install from wheel
pip install dist/ws3-1.1.0a1-py3-none-any.whl

# Verify
python -c "import ws3; print(ws3.__version__)"
```

**Expected**: Package builds, passes `twine check`, installs, and imports correctly.  
**Pass Criteria**: Version prints `1.1.0a1`, no build errors.

---

### Phase 5: Integration with Existing Tests (10 minutes)

Run the existing test suite to ensure new modules don't break existing functionality.

```bash
python -m pytest tests/ -v --tb=short
```

**Expected**: All existing tests pass.  
**Pass Criteria**: No test failures. New import checks in `test_documentation.py` should pass.

---

## Reporting Results

Collaborators should report results using this template:

```markdown
## Smoke Test Results: v1.1.0a1

**Date**: YYYY-MM-DD  
**Environment**: Python X.Y.Z, OS, key dependencies  
**Tester**: Name  

### Phase 1: Import Validation
- [ ] PASS / FAIL
- Notes: ...

### Phase 2: Module Instantiation
- [ ] PASS / FAIL
- Notes: ...

### Phase 3: Notebook Execution
- [ ] PASS / FAIL
- Notes: ...

### Phase 4: Package Distribution
- [ ] PASS / FAIL
- Notes: ...

### Phase 5: Existing Tests
- [ ] PASS / FAIL
- Notes: ...

### Issues Found
1. ...
2. ...
```

Submit results as a GitHub issue comment or PR comment.

---

## Known Limitations

- Notebooks 070-078 require data files that may not be available in all environments. Some cells may fail — this is expected and should be documented.
- Integration modules (`FHOPSIntegrator`, `FEMICIntegrator`, etc.) won't actually connect to external services without those services running. Instantiation only is being tested.
- `RESTAPIServer` instantiation is tested, but starting the server is not part of this smoke test.

---

## Next Steps

After smoke testing completes:

1. Collect all test reports
2. Fix any critical issues (import errors, instantiation failures, test failures)
3. If all phases pass, promote to v1.1.0 stable
4. Update CHANGELOG.md to remove "(alpha)" tag
5. Update `ws3/__init__.py` version to `1.1.0`
6. Close GitHub parent issue #60