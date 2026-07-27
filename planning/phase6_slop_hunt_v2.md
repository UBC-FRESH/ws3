# Phase 6.1 — Second AI Slop Hunt (Critical Verification)

**Date:** 2026-07-26  
**Method:** Every claim in docs must be traceable to code in `ws3/*.py`. If no code backing exists, flag as suspected AI slop.  
**Baseline:** v1.0.5 docs are trusted as slop-free reference.

---

## Methodology

For each documentation file:
1. Extract every claim about ws3 functionality
2. Search `ws3/*.py` for the claimed method/class/function
3. If found: verify the signature matches the docs
4. If not found: flag as **SUSPECTED AI SLOP**

---

## File-by-File Audit

### `howto/data-preparation.rst`

**Claim:** `model.add_development_type(code=..., species=..., site_index=..., age=..., area=...)`  
**Code check:** No such method exists in `ws3/forest.py`. `ForestModel.__init__()` takes `model_name, model_path, base_year, horizon, period_length, max_age, area_epsilon, curve_epsilon`. Data is loaded via `import_landscape_section()`, `import_areas_section()`, etc.  
**Verdict:** ❌ SUSPECTED AI SLOP — fake API

**Claim:** `model.add_curve("volume", df_volume)`  
**Code check:** No such method. Curves are loaded via `import_yields_section()`. The `Curve` class is in `ws3/core.py` with constructor `Curve(points: List[Tuple[int, float]])`.  
**Verdict:** ❌ SUSPECTED AI SLOP — fake API

**Claim:** `model.add_action(code=..., descr=..., components=[...], transitions={...})`  
**Code check:** No such method. Actions are loaded via `import_actions_section()`.  
**Verdict:** ❌ SUSPECTED AI SLOP — fake API

**Claim:** `model.run_simulation(horizon=20)`  
**Code check:** No such method. Simulation is run via `ForestModel.run()` or through optimization.  
**Verdict:** ❌ SUSPECTED AI SLOP — fake API

---

### `howto/curve-definition.rst`

**Claim:** `GrowthCurve(species=..., site_index=..., ages=..., volumes=..., components=[...])`  
**Code check:** No `GrowthCurve` class exists. The actual class is `Curve` in `ws3/core.py` with constructor `Curve(points: List[Tuple[int, float]])`.  
**Verdict:** ❌ SUSPECTED AI SLOP — fake class, fake constructor

---

### `howto/action-definition.rst`

**Claim:** `model.add_action(code=..., descr=..., components=[...], transitions={...})`  
**Code check:** No such method. Actions loaded via `import_actions_section()`.  
**Verdict:** ❌ SUSPECTED AI SLOP — fake API

---

### `howto/running-optimization.rst`

**Claim:** `solve_optimization(model=..., horizon=..., objective=..., flow_constraints=..., area_constraints=...)`  
**Code check:** No such function exists anywhere in `ws3/`.  
**Verdict:** ❌ SUSPECTED AI SLOP — completely fabricated function

**Claim:** `compile_scenario(fm, scenario_name=..., **params)`  
**Code check:** No such function in `ws3/`.  
**Verdict:** ❌ SUSPECTED AI SLOP — fabricated function

---

### `howto/parallel-optimization.rst`

**Claim:** `solve_optimization(model=..., horizon=..., objective=..., flow_constraints=...)`  
**Code check:** Same fabricated function as above.  
**Verdict:** ❌ SUSPECTED AI SLOP — fabricated function

---

### `howto/spatial-schedule-allocation.rst`

**Claim:** Uses `gpd.read_file()`, `spatial_df['dt_code']` manipulation  
**Code check:** `geopandas` is not in `ws3` dependencies. Spatial operations use `ws3.spatial` module with `SpatialConstraint`, `SpatialOptimizer`. The pseudocode doesn't match actual API.  
**Verdict:** ❌ SUSPECTED AI SLOP — pseudocode, not real API

---

### `howto/libcbm-callbacks.rst`

**Claim:** `State()`, `state.get_carbon()`, `state.remove_carbon()`, `state.add_carbon()`, `state.advance()`  
**Code check:** No `State` class exists in `ws3/`. libCBM integration uses `ws3.forest` callbacks with different signatures.  
**Verdict:** ❌ SUSPECTED AI SLOP — fabricated API

---

### `howto/financial-scenarios.rst`

**Claim:** `calculate_revenue()`, `calculate_npv()`  
**Code check:** No such functions in `ws3/`. Financial calculations use `ws3.common.sylv_cred()`, `harv_cost()`, etc.  
**Verdict:** ❌ SUSPECTED AI SLOP — fabricated functions

---

### `howto/scenario-analysis.rst`

**Claim:** `compile_scenario(fm, scenario_name=..., **params)`  
**Code check:** No such function.  
**Verdict:** ❌ SUSPECTED AI SLOP — fabricated function

---

### `howto/reproducibility.rst`

**Claim:** YAML-based config approach with `ForestModel.import_*_section()`  
**Code check:** `ForestModel.import_*_section()` methods exist, but the YAML config approach described doesn't match how ws3 actually works.  
**Verdict:** ⚠️ PARTIALLY SLOP — real methods but fabricated workflow

---

### `howto/model-validation.rst`

**Claim:** `model.get_development_types()`, `curve.get_value(age, 'volume')`  
**Code check:** No `get_development_types()` method. Development types accessed via `fm.development_types` attribute. No `get_value()` method on `Curve` — uses `curve.lookup(y)`.  
**Verdict:** ❌ SUSPECTED AI SLOP — fake methods

---

### `howto/faq.rst`

**Claim:** `ForestModel(model_name=..., model_path=..., base_year=..., horizon=..., period_length=..., max_age=...)`  
**Code check:** Constructor signature is `ForestModel(model_name, model_path, base_year, horizon=common.HORIZON_DEFAULT, period_length=common.PERIOD_LENGTH_DEFAULT, max_age=common.MAX_AGE_DEFAULT, area_epsilon=..., curve_epsilon=...)`. The docs show keyword args but the real constructor uses positional-first. Partially matches but misleading.  
**Verdict:** ⚠️ PARTIALLY SLOP — constructor exists but signature misrepresented

**Claim:** `compile_scenario()`, `add_adjacency_constraints()`  
**Code check:** Neither exists.  
**Verdict:** ❌ SUSPECTED AI SLOP — fabricated functions

---

### `howto/migration_from_woodstock.rst`

**Claim:** R code with fictional `woodstock_model()` pipe-based API  
**Code check:** No such R function exists in Woodstock.  
**Verdict:** ❌ SUSPECTED AI SLOP — fabricated R API

**Claim:** `ForestModel(model_name=..., model_path=..., base_year=..., horizon=..., period_length=..., max_age=...)`  
**Code check:** Same as faq.rst — partially matches but keyword args are wrong.  
**Verdict:** ⚠️ PARTIALLY SLOP

---

### `howto/custom-solvers.rst`

**Claim:** `CustomSolver` class, `register_solver()`, `compile_scenario()`  
**Code check:** No `CustomSolver` class. Solvers are set via `Problem(name, solver="highs")`. No `register_solver()` or `compile_scenario()`.  
**Verdict:** ❌ SUSPECTED AI SLOP — fabricated class and functions

---

### `howto/advanced-optimization.rst`

**Claim:** `compile_scenario(fm, scenario_name=..., objectives=...)`, `find_pareto_frontier()`  
**Code check:** `find_pareto_frontier()` exists in `ws3/advanced_modeling.py` on `MultiObjectiveOptimizer`. But `compile_scenario()` doesn't exist. The docs claim it's a top-level function.  
**Verdict:** ⚠️ PARTIALLY SLOP — `find_pareto_frontier` exists but only on optimizer class, not top-level

---

### `howto/custom-area-selector.rst`

**Claim:** `AreaSelector` base class, `model.set_area_selector()`  
**Code check:** No `AreaSelector` class. No `set_area_selector()` method. Area selection is built into `ForestModel` via `OperateSelector` classes in `ws3/forest.py`.  
**Verdict:** ❌ SUSPECTED AI SLOP — fabricated class and method

---

### `howto/custom-growth-function.rst`

**Claim:** `GrowthCurve(species=..., site_index=..., ages=..., volumes=...)`  
**Code check:** Same as curve-definition.rst. No `GrowthCurve` class.  
**Verdict:** ❌ SUSPECTED AI SLOP — fabricated class

---

### `howto/data-validation.rst`

**Claim:** `fm.development_types.shape`, `fm.development_types.duplicated()`, `fm.yields.keys()`  
**Code check:** `fm.development_types` is a dict (not a DataFrame, no `.shape`). `fm.yields` is a dict. The pandas-style API is fabricated.  
**Verdict:** ❌ SUSPECTED AI SLOP — pandas-style API that doesn't exist

---

### `howto/index.rst`

**Claim:** "Step-by-step guides for common modelling tasks in ws3"  
**Code check:** N/A (metadata claim)  
**Verdict:** ⚠️ FILLER — repeats title, adds no value

**Claim:** Files have "Goal", "Prerequisites", "Expected Output", "Troubleshooting" sections  
**Code check:** None of the howto files actually have these sections. The index describes a format that doesn't exist.  
**Verdict:** ❌ SUSPECTED AI SLOP — describes non-existent structure

---

### `getting_started/index.rst`

**Claim:** "This section helps you get up and running with ws3 quickly."  
**Code check:** Filler text.  
**Verdict:** ⚠️ FILLER

**Claim:** "in order for the smoothest onboarding experience"  
**Code check:** Filler phrase.  
**Verdict:** ⚠️ FILLER

---

### `getting_started/quickstart.rst`

**Claim:** `ForestModel()` with no arguments  
**Code check:** Constructor requires `model_name, model_path, base_year`.  
**Verdict:** ❌ SUSPECTED AI SLOP — fake constructor usage

**Claim:** `model.add_development_type(code=..., area=..., age=..., species=..., site_index=...)`  
**Code check:** No such method.  
**Verdict:** ❌ SUSPECTED AI SLOP — fake API

**Claim:** `model.add_curve("volume", df_volume)`  
**Code check:** No such method.  
**Verdict:** ❌ SUSPECTED AI SLOP — fake API

**Claim:** `model.add_action(code=..., descr=..., components=[...], transitions={...})`  
**Code check:** No such method.  
**Verdict:** ❌ SUSPECTED AI SLOP — fake API

**Claim:** `model.run_simulation(horizon=20)`  
**Code check:** No such method.  
**Verdict:** ❌ SUSPECTED AI SLOP — fake API

**Claim:** `results.summary()`, `results.total_volume()`, `results.total_harvest()`  
**Code check:** No `results` object with these methods exists.  
**Verdict:** ❌ SUSPECTED AI SLOP — fabricated results API

---

### `getting_started/first_model.rst`

**Claim:** `Problem()`, `prob.add_variable()`, `prob.set_objective()`, `prob.add_constraint()`  
**Code check:** `Problem` exists but requires `name` param. Methods are `add_var()`, `z()`, `add_constraint()` — not `add_variable`, `set_objective`.  
**Verdict:** ❌ SUSPECTED AI SLOP — wrong method names

---

### `getting_started/architecture_overview.rst`

**Claim:** Various class hierarchies and data flows  
**Code check:** Need to verify each claim against actual class definitions.  
**Verdict:** ⏳ NEEDS VERIFICATION

---

## Summary

| Category | Count |
|----------|-------|
| ❌ SUSPECTED AI SLOP (fake API) | 18 |
| ⚠️ PARTIALLY SLOP (mixed real/fake) | 4 |
| ⚠️ FILLER (no technical value) | 3 |
| ⏳ NEEDS VERIFICATION | 1 |

**Total files audited:** 23  
**Files with confirmed fake APIs:** 18 out of 23 (78%)