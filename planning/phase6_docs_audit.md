I've now read through all the documentation files. Here is the complete audit.

---

# Documentation Slop Audit: `/home/gep/projects/ws3/docs/source/`

## Executive Summary

The `howto/` directory is almost entirely AI-generated slop — every file follows the identical template: **Goal → Prerequisites → Step-by-Step Instructions → Expected Output → Troubleshooting → Next Steps**. The code examples are generic stubs that don't match the actual ws3 API. The `getting_started/` files follow the same pattern. The `Chapt1.rst` and `Chapt2.rst` legacy files are verbose and conversational. The textbook chapters are the best-written content but still have filler. The reference contracts are the cleanest files.

---

## 1. `howto/data-preparation.rst` — **HEAVY SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Fake API | All | `model.add_development_type(code=..., species=..., site_index=..., age=..., area=...)` — this API does not exist in ws3. The real API uses `ForestModel` constructor + `import_*_section()` methods. | Delete the entire file. Point to `getting_started/first_model.rst` or the actual data import functions. |
| Template structure | Entire file | Goal/Prerequisites/Steps/Expected Output/Troubleshooting/Next Steps — pure AI template padding. | Eliminate the template. A how-to should be a procedure, not a form letter. |
| Redundant | Lines 1-5 | "Goal" section repeats what the file title already says. | Delete "Goal" section entirely. |
| Bloating | Lines 15-20 | "Prerequisites" lists things already listed in the section-level prerequisites. | Delete. |
| Fake troubleshooting | Lines 55-65 | Generic "Missing development types" / "Growth curve errors" — these errors don't correspond to the fake API being demonstrated. | Delete. |

**Suggested replacement:** Delete the file. Data preparation is covered by `getting_started/first_model.rst` and the `ForestModel.import_*_section()` documentation.

---

## 2. `howto/curve-definition.rst` — **HEAVY SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Fake API | Lines 18-22 | `GrowthCurve(species=..., site_index=..., ages=..., volumes=..., components=[...])` — does not match the actual `ws3.core.Curve` class. | Delete. |
| Template padding | Entire file | Identical template as data-preparation. | Delete. |
| Redundant | "Expected Output" | "GrowthCurve object created and validated" — adds nothing. | Delete. |
| Fake troubleshooting | Lines 48-55 | "Interpolation errors" — the Curve class uses linear interpolation, there are no "interpolation errors" to troubleshoot. | Delete. |

**Suggested replacement:** Delete. Curve definition is trivially demonstrated in `getting_started/quickstart.rst` Step 4 and textbook ch03.

---

## 3. `howto/action-definition.rst` — **HEAVY SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Fake API | Lines 14-25 | `model.add_action(code=..., descr=..., components=[...], transitions={...})` — does not match the actual API. | Delete. |
| Template padding | Entire file | Same template. | Delete. |
| Redundant | "Next Steps" | Links to files that are also slop. | Delete. |

**Suggested replacement:** Delete. Actions are covered in textbook ch04 and `getting_started/quickstart.rst` Step 5.

---

## 4. `howto/running-optimization.rst` — **HEAVY SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Fake API | Lines 22-35 | `solve_optimization(model=..., horizon=..., objective=..., flow_constraints=..., area_constraints=...)` — this function does not exist. | Delete. |
| Template padding | Entire file | Same template. | Delete. |
| Redundant | "Expected Output" | "Optimization solution object" — meaningless. | Delete. |
| Fake troubleshooting | Lines 52-62 | Generic issues that don't correspond to real API. | Delete. |

**Suggested replacement:** Delete. Optimization is covered in `getting_started/first_model.rst` and textbook ch05.

---

## 5. `howto/parallel-optimization.rst` — **HEAVY SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Fake API | Lines 32-48 | `solve_optimization(model=..., horizon=..., objective=..., flow_constraints=...)` — doesn't exist. | Delete. |
| Template padding | Entire file | Same template. | Delete. |
| Fake troubleshooting | Lines 62-72 | Generic issues. | Delete. |

**Suggested replacement:** Delete. Parallel optimization is a coding pattern, not a ws3-specific how-to.

---

## 6. `howto/spatial-schedule-allocation.rst` — **HEAVY SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Fake API | Lines 30-50 | `gpd.read_file()`, `spatial_df['dt_code']` manipulation — the code is hand-wavy pseudocode, not real ws3 usage. | Delete. |
| Template padding | Entire file | Same template. | Delete. |
| Fake troubleshooting | Lines 115-130 | Generic spatial issues. | Delete. |

**Suggested replacement:** Delete. Spatial allocation is covered in the `ws3.spatial` module docs and textbook ch06.

---

## 7. `howto/libcbm-callbacks.rst` — **HEAVY SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Fake API | Lines 14-45 | `State()`, `state.get_carbon()`, `state.remove_carbon()`, `state.add_carbon()`, `state.advance()` — none of these exist. | Delete. |
| Template padding | Entire file | Same template. | Delete. |
| Fake troubleshooting | Lines 70-80 | Generic callback issues. | Delete. |

**Suggested replacement:** Delete. libCBM integration is covered in textbook ch10 and the actual callback system in `ws3.forest`.

---

## 8. `howto/financial-scenarios.rst` — **HEAVY SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Fake API | Lines 30-65 | Hand-written `calculate_revenue()`, `calculate_npv()` functions — these are not ws3 functions. | Delete. |
| Template padding | Entire file | Same template. | Delete. |
| Fake troubleshooting | Lines 82-92 | Generic financial issues. | Delete. |

**Suggested replacement:** Delete. Financial analysis is covered in textbook ch07.

---

## 9. `howto/scenario-analysis.rst` — **HEAVY SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Fake API | Lines 18-30 | `compile_scenario(fm, scenario_name=..., **params)` — doesn't match actual API. | Delete. |
| Template padding | Entire file | Same template, plus a "What You Will Find Here" section that's pure padding. | Delete. |
| Redundant | Lines 10-15 | "Scenario Analysis Workflow" — a numbered list that adds nothing the steps don't already say. | Delete. |

**Suggested replacement:** Delete. Scenario analysis is a general concept, not ws3-specific.

---

## 10. `howto/reproducibility.rst` — **HEAVY SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Fake API | Lines 40-100 | `yaml`-based config approach — not how ws3 works. The real API uses `ForestModel.import_*_section()`. | Delete. |
| Template padding | Entire file | Same template. | Delete. |

**Suggested replacement:** Delete. Reproducibility is a general software engineering topic, not ws3-specific.

---

## 11. `howto/model-validation.rst` — **HEAVY SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Fake API | Lines 12-55 | `model.get_development_types()`, `curve.get_value(age, 'volume')` — doesn't match actual API. | Delete. |
| Template padding | Entire file | Same template. | Delete. |

**Suggested replacement:** Delete. Validation is a general concept.

---

## 12. `howto/faq.rst` — **HEAVY SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Fake API throughout | Lines 30-120 | Every code example uses non-existent APIs: `ForestModel(model_name=..., model_path=..., base_year=..., horizon=..., period_length=..., max_age=...)`, `compile_scenario()`, `add_adjacency_constraints()`. | Rewrite using actual API or delete. |
| Padding | Lines 1-3 | "This document answers the most common questions about using ws3" — filler. | Delete. |
| Fake Q12 | Line 120+ | File is cut off mid-question. | Fix or delete. |

**Suggested replacement:** Rewrite from scratch with real API examples, or consolidate into the `getting_started/` section.

---

## 13. `howto/migration_from_woodstock.rst` — **MODERATE SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Fake R code | Lines 28-35 | The R code examples use a fictional `woodstock_model()` pipe-based API that doesn't match actual Woodstock R. | Rewrite with real Woodstock R examples. |
| Fake Python code | Lines 65-85 | `ForestModel(model_name=..., model_path=..., base_year=..., horizon=..., period_length=..., max_age=...)` — doesn't match actual API. | Rewrite with real API. |
| Template padding | "Migration Steps" | Steps 1-4 are generic and don't reflect actual migration needs. | Rewrite with real migration procedures. |

---

## 14. `howto/custom-solvers.rst` — **HEAVY SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Fake API | Lines 16-75 | `CustomSolver` class, `register_solver()`, `compile_scenario()` — none of these exist. | Delete. |
| Template padding | Entire file | Same template. | Delete. |
| Fake best practices | Lines 105-110 | Generic advice. | Delete. |

**Suggested replacement:** Delete. Custom solver integration is an advanced topic not relevant to most users.

---

## 15. `howto/advanced-optimization.rst` — **HEAVY SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Fake API | Lines 12-55 | `compile_scenario(fm, scenario_name=..., objectives=...)`, `find_pareto_frontier()` — don't exist. | Delete. |
| Template padding | Entire file | Same template. | Delete. |
| Fake best practices | Lines 95-102 | Generic advice. | Delete. |

**Suggested replacement:** Delete.

---

## 16. `howto/custom-area-selector.rst` — **HEAVY SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Fake API | Lines 30-55 | `AreaSelector` base class, `model.set_area_selector()` — don't exist. | Delete. |
| Template padding | Entire file | Same template. | Delete. |

**Suggested replacement:** Delete.

---

## 17. `howto/custom-growth-function.rst` — **HEAVY SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Fake API | Lines 16-40 | `GrowthCurve(species=..., site_index=..., ages=..., volumes=...)` — doesn't match actual `Curve` class. | Delete. |
| Template padding | Entire file | Same template. | Delete. |

**Suggested replacement:** Delete.

---

## 18. `howto/data-validation.rst` — **HEAVY SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Fake API | Throughout | `fm.development_types.shape`, `fm.development_types.duplicated()`, `fm.yields.keys()` — the actual API uses different attribute names. | Rewrite with real API or delete. |
| Template padding | Entire file | Same template. | Delete. |

**Suggested replacement:** Delete.

---

## 19. `howto/index.rst` — **MODERATE SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Redundant | Lines 1-3 | "Step-by-step guides for common modelling tasks in ws3" — repeats what the title says. | Delete. |
| Redundant | Lines 13-18 | "What You Will Find Here" — describes a format that doesn't actually exist in the files (none of the howtos have "Goal", "Prerequisites", etc.). | Delete. |
| Redundant | Lines 20-30 | "Common Tasks Covered" — lists topics already covered by the toctree. | Delete. |
| Redundant | Lines 32-40 | "Prerequisites" — repeats section-level prerequisites. | Delete. |

**Suggested replacement:** Keep only the toctree. Delete everything else.

---

## 20. `getting_started/index.rst` — **MODERATE SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Filler | Line 3 | "This section helps you get up and running with ws3 quickly." — filler. | Delete. |
| Filler | Line 8 | "in order for the smoothest onboarding experience" — filler phrase. | Delete. |
| Redundant | "What You Will Learn" | Lists outcomes already obvious from the subsection titles. | Delete. |
| Redundant | "Prerequisites" | Repeats prerequisites from individual pages. | Delete. |
| Redundant | "Estimated Time" | Guesswork padding. | Delete. |

---

## 21. `getting_started/quickstart.rst` — **MODERATE SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Fake API | Lines 20-25 | `ForestModel()` with no arguments — the real constructor requires `model_name` and `model_path`. | Fix. |
| Fake API | Lines 30-50 | `model.add_development_type(code=..., area=..., age=..., species=..., site_index=...)` — doesn't exist. | Fix. |
| Fake API | Lines 55-70 | `model.add_curve("volume", df_volume)` — doesn't exist. | Fix. |
| Fake API | Lines 75-85 | `model.add_action(code=..., descr=..., components=[...], transitions={...})` — doesn't exist. | Fix. |
| Fake API | Lines 90-95 | `model.run_simulation(horizon=20)` — doesn't exist. | Fix. |
| Filler | Lines 100-110 | "Inspect the Output" with fake methods `results.summary()`, `results.total_volume()`, `results.total_harvest()`. | Fix or delete. |
| Filler | "What's Next?" | Generic links. | Keep but tighten. |

---

## 22. `getting_started/first_model.rst` — **MODERATE SLOP**

| Issue | Lines | Problem | Fix |
|-------|-------|---------|-----|
| Fake API | Lines 30-80 | Same fake `add_development_type()`, `add_curve()`, `add_action()` as quickstart. | Fix. |
| Fake API | Lines 85-120 | `Problem()`, `prob.add_variable()`, `prob.set_objective()`, `prob.add_constraint()` — don't match actual API. | Fix. |
| Filler | "Scenario" section | A 4-line scenario description that could be inline. | Condense. |

---

## 23. `getting_started/architecture_overview.rst` — **MODERATE SLOP**

| Issue | Lines | Problem | Fix |
|