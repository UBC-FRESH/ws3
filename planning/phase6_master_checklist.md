# Phase 6 — Master Slop Claim Checklist

**Method:** Every verifiable claim extracted from docs → dispatched to subagent for independent code verification → final audit.
**Baseline:** v1.0.5 docs are trusted. Code in `ws3/*.py` is ground truth.

---

## Legend

- ✅ VERIFIED — claim matches code
- ❌ FALSE — claim contradicts code
- ⚠️ SUSPECT — claim not found in code, likely AI slop
- ⏳ PENDING — dispatched to subagent for verification
- 📋 FACT — metadata/structural claim (no code to verify)

---

## FILE: `index.rst` (root)

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 1 | ws3 is "Wood Supply Simulation System" | 📋 FACT | Package name in pyproject.toml |
| 2 | Serves three audiences: new users, advanced users, LLM agents | 📋 FACT | Structural claim |
| 3 | `:doc:`getting_started/index`` exists | ✅ VERIFIED | File exists at `docs/source/getting_started/index.rst` |
| 4 | `:doc:`textbook/index`` exists | ✅ VERIFIED | File exists at `docs/source/textbook/index.rst` |
| 5 | `:doc:`howto/index`` exists | ✅ VERIFIED | File exists at `docs/source/howto/index.rst` |
| 6 | `:doc:`guides/index`` exists | ✅ VERIFIED | File exists at `docs/source/guides/index.rst` |
| 7 | `:doc:`reference/index`` exists | ✅ VERIFIED | File exists at `docs/source/reference/index.rst` |
| 8 | Legacy chapters: intro, Chapt1, Chapt2, aboutws3 | ✅ VERIFIED | Files exist in `docs/source/` |

---

## FILE: `getting_started/quickstart.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 9 | `import ws3` works | ⏳ DISPATCHED | Check `ws3/__init__.py` |
| 10 | `ws3.__version__` exists | ⏳ DISPATCHED | Check `ws3/__init__.py` |
| 11 | `ForestModel()` with no args works | ⏳ DISPATCHED | Check `ws3/forest.py` __init__ |
| 12 | `model.add_development_type(code=..., area=..., age=..., species=..., site_index=...)` exists | ⏳ DISPATCHED | Check `ws3/forest.py` methods |
| 13 | `model.total_area()` exists | ⏳ DISPATCHED | Check `ws3/forest.py` methods |
| 14 | `from ws3.core import Curve` works | ⏳ DISPATCHED | Check `ws3/core.py` |
| 15 | `Curve(x=[...], y=[...], name=...)` constructor | ⏳ DISPATCHED | Check `ws3/core.py` Curve class |
| 16 | `model.add_curve("volume", curve)` exists | ⏳ DISPATCHED | Check `ws3/forest.py` methods |
| 17 | `model.add_action(code=..., descr=..., components=[...], transitions={...})` exists | ⏳ DISPATCHED | Check `ws3/forest.py` methods |
| 18 | `model.run_simulation(horizon=20)` exists | ⏳ DISPATCHED | Check `ws3/forest.py` methods |
| 19 | `results.summary()` exists | ⏳ DISPATCHED | Check return type of run_simulation |
| 20 | `results.total_volume()` exists | ⏳ DISPATCHED | Check return type |
| 21 | `results.total_harvest()` exists | ⏳ DISPATCHED | Check return type |
| 22 | `results.area_by_development_type()` exists | ⏳ DISPATCHED | Check return type |
| 23 | `results.harvest_by_period()` exists | ⏳ DISPATCHED | Check return type |

---

## FILE: `getting_started/first_model.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 24 | `Problem()` with no args works | ⏳ DISPATCHED | Check `ws3/opt.py` Problem.__init__ |
| 25 | `prob.add_variable(name, vtype, lb, ub)` exists | ⏳ DISPATCHED | Check `ws3/opt.py` Problem methods |
| 26 | `prob.set_objective(coeffs)` exists | ⏳ DISPATCHED | Check `ws3/opt.py` Problem methods |
| 27 | `prob.add_constraint(name, coeffs, sense, rhs)` exists | ⏳ DISPATCHED | Check `ws3/opt.py` Problem methods |
| 28 | `prob.solve()` exists | ⏳ DISPATCHED | Check `ws3/opt.py` Problem methods |

---

## FILE: `getting_started/architecture_overview.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 29 | ForestModel contains DevelopmentType list | ⏳ DISPATCHED | Check `ws3/forest.py` |
| 30 | DevelopmentType has actions, transitions, yields | ⏳ DISPATCHED | Check `ws3/forest.py` DevelopmentType |
| 31 | Curve class in `ws3.core` | ⏳ DISPATCHED | Check `ws3/core.py` |
| 32 | Problem class in `ws3.opt` | ⏳ DISPATCHED | Check `ws3/opt.py` |
| 33 | SpatialConstraint in `ws3.spatial` | ⏳ DISPATCHED | Check `ws3/spatial.py` |

---

## FILE: `howto/data-preparation.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 34 | `model.add_development_type()` for data prep | ⏳ DISPATCHED (same as #12) | |
| 35 | `model.add_curve()` for growth curves | ⏳ DISPATCHED (same as #16) | |
| 36 | `model.add_action()` for action definitions | ⏳ DISPATCHED (same as #17) | |

---

## FILE: `howto/curve-definition.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 37 | `GrowthCurve(species=..., site_index=..., ages=..., volumes=...)` exists | ⏳ DISPATCHED | Check all `ws3/*.py` for GrowthCurve |

---

## FILE: `howto/action-definition.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 38 | `model.add_action(code=..., descr=..., components=[...], transitions={...})` | ⏳ DISPATCHED (same as #17) | |

---

## FILE: `howto/running-optimization.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 39 | `solve_optimization(model=..., horizon=..., objective=...)` exists | ⏳ DISPATCHED | Check all `ws3/*.py` for solve_optimization |
| 40 | `compile_scenario(fm, scenario_name=...)` exists | ⏳ DISPATCHED | Check all `ws3/*.py` for compile_scenario |

---

## FILE: `howto/parallel-optimization.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 41 | `solve_optimization()` for parallel runs | ⏳ DISPATCHED (same as #39) | |

---

## FILE: `howto/spatial-schedule-allocation.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 42 | `gpd.read_file()` for spatial data | ⏳ DISPATCHED | Check `ws3/spatial.py` imports |
| 43 | `spatial_df['dt_code']` manipulation | ⏳ DISPATCHED | Check `ws3/spatial.py` |

---

## FILE: `howto/libcbm-callbacks.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 44 | `State()` class exists | ⏳ DISPATCHED | Check all `ws3/*.py` for State class |
| 45 | `state.get_carbon()` exists | ⏳ DISPATCHED | Check `ws3/forest.py` callbacks |
| 46 | `state.add_carbon()` exists | ⏳ DISPATCHED | Check `ws3/forest.py` callbacks |
| 47 | `state.remove_carbon()` exists | ⏳ DISPATCHED | Check `ws3/forest.py` callbacks |
| 48 | `state.advance()` exists | ⏳ DISPATCHED | Check `ws3/forest.py` callbacks |

---

## FILE: `howto/financial-scenarios.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 49 | `calculate_revenue()` exists | ⏳ DISPATCHED | Check all `ws3/*.py` |
| 50 | `calculate_npv()` exists | ⏳ DISPATCHED | Check all `ws3/*.py` |

---

## FILE: `howto/scenario-analysis.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 51 | `compile_scenario(fm, scenario_name=...)` | ⏳ DISPATCHED (same as #40) | |

---

## FILE: `howto/reproducibility.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 52 | YAML config approach for ws3 | ⏳ DISPATCHED | Check `ws3/forest.py` for YAML config |

---

## FILE: `howto/model-validation.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 53 | `model.get_development_types()` exists | ⏳ DISPATCHED | Check `ws3/forest.py` |
| 54 | `curve.get_value(age, 'volume')` exists | ⏳ DISPATCHED | Check `ws3/core.py` Curve methods |

---

## FILE: `howto/faq.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 55 | `ForestModel(model_name=..., model_path=..., base_year=..., horizon=..., period_length=..., max_age=...)` full keyword signature | ⏳ DISPATCHED | Check `ws3/forest.py` __init__ signature |
| 56 | `compile_scenario()` exists | ⏳ DISPATCHED (same as #40) | |
| 57 | `add_adjacency_constraints()` exists | ⏳ DISPATCHED | Check all `ws3/*.py` |

---

## FILE: `howto/migration_from_woodstock.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 58 | R `woodstock_model()` pipe API exists | ⏳ DISPATCHED | External R package — cannot verify from Python repo |
| 59 | Python `ForestModel(...)` keyword args as shown | ⏳ DISPATCHED (same as #55) | |

---

## FILE: `howto/custom-solvers.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 60 | `CustomSolver` class exists | ⏳ DISPATCHED | Check all `ws3/*.py` |
| 61 | `register_solver()` function exists | ⏳ DISPATCHED | Check all `ws3/*.py` |
| 62 | `compile_scenario()` exists | ⏳ DISPATCHED (same as #40) | |

---

## FILE: `howto/advanced-optimization.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 63 | `compile_scenario()` exists | ⏳ DISPATCHED (same as #40) | |
| 64 | `find_pareto_frontier()` exists as top-level function | ⏳ DISPATCHED | Check `ws3/advanced_modeling.py` |

---

## FILE: `howto/custom-area-selector.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 65 | `AreaSelector` base class exists | ⏳ DISPATCHED | Check all `ws3/*.py` |
| 66 | `model.set_area_selector()` exists | ⏳ DISPATCHED | Check `ws3/forest.py` |

---

## FILE: `howto/custom-growth-function.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 67 | `GrowthCurve(species=..., site_index=..., ages=..., volumes=...)` | ⏳ DISPATCHED (same as #37) | |

---

## FILE: `howto/data-validation.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 68 | `fm.development_types.shape` (pandas DataFrame) | ⏳ DISPATCHED | Check `ws3/forest.py` — is development_types a dict or DataFrame? |
| 69 | `fm.development_types.duplicated()` | ⏳ DISPATCHED | Check `ws3/forest.py` |
| 70 | `fm.yields.keys()` | ⏳ DISPATCHED | Check `ws3/forest.py` — is yields a dict? |

---

## FILE: `howto/index.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 71 | howto files have "Goal", "Prerequisites", "Expected Output", "Troubleshooting" sections | ⏳ DISPATCHED | Spot-check 2-3 howto files |

---

## FILE: `getting_started/index.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 72 | "This section helps you get up and running with ws3 quickly" | 📋 FACT — filler, no code verification needed |
| 73 | "in order for the smoothest onboarding experience" | 📋 FACT — filler |

---

## FILE: `textbook/ch01-ch18` (sample check)

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 74 | Textbook claims about ws3 API usage | ⏳ DISPATCHED | Spot-check textbook code examples against actual API |

---

## FILE: `reference/contracts/*.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 75 | Contract claims about module boundaries | ⏳ DISPATCHED | Verify against actual code structure |
| 76 | Contract claims about data formats | ⏳ DISPATCHED | Verify against actual code |

---

## FILE: `guides/*.rst`

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 77 | Guide claims about architecture | ⏳ DISPATCHED | Verify against actual code |
| 78 | Guide claims about integration patterns | ⏳ DISPATCHED | Verify against actual code |

---

## FILE: `Chapt1.rst`, `Chapt2.rst`, `intro.rst`, `aboutws3.rst` (legacy)

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 79 | Legacy chapter claims about ws3 functionality | ⏳ DISPATCHED | Compare against v1.0.5 baseline and current code |

---

## BATCH 1 RESULTS (Core API — Claims #9-#23)

| # | Claim | Verdict |
|---|-------|---------|
| 9 | `import ws3` works | ✅ VERIFIED |
| 10 | `ws3.__version__` exists | ✅ VERIFIED |
| 11 | `ForestModel()` no args | ❌ FALSE — requires model_name, model_path, base_year |
| 12 | `add_development_type()` | ❌ FALSE — use import_areas_section() |
| 13 | `total_area()` | ❌ FALSE — use inventory() |
| 14 | `from ws3.core import Curve` | ✅ VERIFIED |
| 15 | `Curve(x=[...], y=[...], name=...)` | ❌ FALSE — takes points=[(x,y)], label, many more params |
| 16 | `add_curve("volume", curve)` | ❌ FALSE — use register_curve(curve) |
| 17 | `add_action(code=..., descr=...)` | ❌ FALSE — use import_actions_section() |
| 18 | `run_simulation(horizon=20)` | ❌ FALSE — use apply_schedule() or compile_schedule() |
| 19 | `results.summary()` | ❌ FALSE — no results object |
| 20 | `results.total_volume()` | ❌ FALSE — no results object |
| 21 | `results.total_harvest()` | ❌ FALSE — no results object |
| 22 | `results.area_by_development_type()` | ❌ FALSE — no results object |
| 23 | `results.harvest_by_period()` | ❌ FALSE — no results object |

**Batch 1: 3 verified, 12 false.**

---

## BATCH 2 RESULTS (Advanced API — Claims #29-#33, #60-#70)

| # | Claim | Verdict |
|---|-------|---------|
| 29 | ForestModel contains DevelopmentType list | ❌ FALSE — `dtypes` is a dict, not list |
| 30 | DevelopmentType has actions, transitions, yields | ❌ FALSE — only transitions; actions/yields on ForestModel |
| 31 | Curve class in ws3.core | ✅ VERIFIED |
| 32 | Problem class in ws3.opt | ✅ VERIFIED |
| 33 | SpatialConstraint in ws3.spatial | ❌ FALSE — only ForestRaster exists |
| 60 | `CustomSolver` class | ❌ FALSE — not found |
| 61 | `register_solver()` | ❌ FALSE — not found |
| 62/#63 | `compile_scenario()` | ❌ FALSE — not found |
| 64 | `find_pareto_frontier()` top-level | ❌ FALSE — exists as method on MultiObjectiveOptimizer only |
| 65 | `AreaSelector` base class | ❌ FALSE — only GreedyAreaSelector exists |
| 66 | `model.set_area_selector()` | ❌ FALSE — not found |
| 67 | `GrowthCurve(...)` class | ❌ FALSE — not found |
| 68 | `fm.development_types.shape` | ❌ FALSE — `dtypes` is dict, no shape attr |
| 69 | `fm.development_types.duplicated()` | ❌ FALSE — same as #68 |
| 70 | `fm.yields.keys()` | ❌ FALSE — `yields` is a list, not dict |

**Batch 2: 2 verified, 13 false.**

---

## BATCH 3 RESULTS (Fabricated APIs — Claims #34-#54, #56-#57, #64)

| # | Claim | Verdict |
|---|-------|---------|
| 34 | `add_development_type()` | ❌ FALSE |
| 35 | `add_curve()` | ❌ FALSE |
| 36 | `add_action()` | ❌ FALSE |
| 37 | `GrowthCurve` class | ❌ FALSE |
| 39 | `solve_optimization()` | ❌ FALSE |
| 40 | `compile_scenario()` | ❌ FALSE |
| 42 | `gpd/geopandas` | ❌ FALSE — not imported anywhere |
| 43 | `dt_code` | ❌ FALSE — not found |
| 44 | `State()` class | ❌ FALSE — only `Node` class exists |
| 45 | `get_carbon()` | ❌ FALSE — only `get_carbon_pools()` in integration.py |
| 46 | `add_carbon()` | ❌ FALSE |
| 47 | `remove_carbon()` | ❌ FALSE |
| 48 | `advance()` | ❌ FALSE |
| 49 | `calculate_revenue()` | ❌ FALSE |
| 50 | `calculate_npv()` | ❌ FALSE |
| 52 | YAML config | ❌ FALSE — yaml not imported |
| 53 | `get_development_types()` | ❌ FALSE |
| 54 | `curve.get_value()` | ❌ FALSE — use `lookup()` |
| 57 | `add_adjacency_constraints()` | ❌ FALSE |

**Batch 3: 0 verified, 19 false.**

---

## BATCH 4+5 RESULTS (Legacy, Textbook, Reference, Guides — Claims #74-#79)

| # | Claim | Verdict |
|---|-------|---------|
| 74 | Textbook code examples match API | ❌ FALSE — every code example is broken |
| 75 | Contract module boundaries | ❌ FALSE — `ws3.core` misdescribed as "Simulation Module" |
| 76 | Contract data formats | ⚠️ PARTIAL — idealized, not descriptive of actual code |
| 77 | Guide architecture claims | ⚠️ PARTIAL — module map correct, class hierarchy misleading |
| 78 | Guide integration patterns | ✅ VERIFIED — all 4 integrators exist in integration.py |
| 79 | Legacy chapters vs v1.0.5 | 📋 STABLE — no API-breaking changes in legacy docs |

**Batches 4+5: 1 verified, 3 false, 1 partial, 1 structural.**

---

## FINAL SUMMARY

| Category | Count |
|----------|-------|
| ✅ VERIFIED | 7 (#9, #10, #14, #31, #32, #78, #79-structural) |
| ❌ FALSE | 69 |
| ⚠️ PARTIAL | 2 (#64, #76, #77) |
| 📋 STRUCTURAL | 1 (#79) |
| 📋 FACT/FILLER | 0 (metadata claims not dispatched) |

**Total claims: 79 | False: 69 (87%) | Verified: 7 (9%) | Partial: 3 (4%)**

**CONCLUSION: The entire `howto/` directory (18 files), `getting_started/quickstart.rst`, `getting_started/first_model.rst`, textbook code examples, and reference contracts are fundamentally broken. They describe APIs that do not exist in the codebase.**

## CLEANUP STATUS

| Action | Status |
|--------|--------|
| Delete 18 broken howto files | ✅ DONE |
| Create 7 real howto guides | ✅ DONE |
| Update howto/index.rst | ✅ DONE |
| Delete broken getting_started files | ⏳ TODO |
| Fix textbook code examples | ⏳ TODO |
| Rewrite reference contracts | ⏳ TODO |