# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Common Changelog](https://common-changelog.org/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 1.1.0a3 - 2026-07-29 (alpha)

**Status**: Patch release. Clears code-quality debt that was blocking meaningful CI signal, and fixes runtime defects shipped in 1.1.0a2.

### Fixed
- **`_cp` period conditions were broken for `>=` and `<=`.** Two separate defects in `DevelopmentType._compile_oper_expr` meant only `_cp =` ever worked:
  - the `<=` branch referenced an undefined `rel_opertors` (missing `a`), raising `NameError`
  - the bound was then folded in with an unguarded `max(_plo, plo), min(_phi, phi)`; a one-sided comparison leaves the opposite bound `None`, raising `TypeError`. The parallel `_age` branch already guarded for this.
- `resolve_tmask` called `resolve_treplace` and `resolve_tappend` without `self.`, so any model using `_REPLACE` or `_APPEND` theme expressions crashed.
- `_evaluate_basic` referenced a bare `parent` instead of `self.parent`.
- `resolve_replace`'s exception handler printed seven out-of-scope names, raising `NameError` inside the handler and masking the original error.
- A debug print in the target-age resolver referenced `age` where the variable is `sage`.
- Removed `_expand_action`, unreachable dead code that could not execute under any input.

### Changed
- **CI no longer collapses on a lint failure.** `test`, `docs`, and `build` were transitively gated on `lint`; with the lint job permanently red, none of them ever ran. They are now independent. mypy is advisory pending the Phase 2 typing debt.
- Added `.flake8` codifying an explicit policy: correctness enforced (all pyflakes codes, `E7xx` logic errors, syntax, deprecations), style codes that conflict with long-standing project conventions documented and ignored.
- flake8 across `ws3/` and `tests/` now exits clean, down from 1808 findings.

### Added
- Parametrized regression tests covering all three `_cp` relational operators and the invalid-operator path.

### Notes
Three tests were passing without testing anything, and are now explicitly skipped with the reason stated rather than silently vacuous:
- `test_operate` was decorated `@pytest.fixture`, so pytest collected it as a fixture and never ran it; it also requests an `area_selector` fixture that does not exist.
- `test_yield_curve_interpolation` calls `ws3.core.interpolate_curves`, which does not exist; absent test data triggered an early skip before the undefined name was evaluated.
- `test_documentation` computed a section-header flag and never asserted it. The implied assertion is now present and passing.

## 1.1.0a2 - 2026-07-29 (alpha)

**Status**: Second alpha release. Documentation cleanup complete (Phase 6). Ready for user testing.

### Changed
- **Phase 6 complete**: All documentation fabricated APIs purged, legacy stubs removed, troubleshooting guide fixed
- Documentation now uses real ws3 APIs throughout (verified against source)
- `sphinx-build -b html docs/source _build/html -W` passes with zero errors

### Removed
- Deprecated legacy chapters (`Chapt1.rst`, `Chapt2.rst`, `intro.rst`, `aboutws3.rst`)
- 2-line stubs (`SpaDES.rst`, `libCBM.rst`) — content covered by textbook

## 1.1.0a1 - 2026-07-26 (alpha)

**Status**: Alpha release for smoke testing. Not yet stable.

## Phase 7 — Release and Community Building (2026-07-27)

**Status**: Complete (2026-07-29)

### Added
- Issue templates (bug_report, feature_request, question) for GitHub
- Community channels: GitHub Discussions enabled, CONTRIBUTING.md updated with support info
- Phase 7 planning docs and GitHub issues (#87-#91)
- Feature branch `feature/ws3-phase7-release`

### Added
- Phase 5 interactive notebooks (070-078) for advanced workflows
- FAQ section in documentation (`docs/source/howto/faq.rst`)
- Migration guide from Woodstock to ws3 (`docs/source/howto/migration_from_woodstock.rst`)
- Multi-objective optimization examples
- Spatial constraints and adjacency modeling
- Parallel optimization and performance benchmarking
- Scenario analysis and comparison workflows
- New modules: `ws3.advanced_modeling`, `ws3.perf`, `ws3.integration`
- Textbook chapters 17-18 (advanced spatial, carbon accounting)
- How-to guides: advanced-optimization, custom-solvers, data-validation, scenario-analysis
- CI/CD pipeline with lint, test, docs, and publish jobs

### Changed
- Updated ROADMAP.md to reflect Phase 5 progress
- Enhanced documentation with 18 how-to guides (was 14)
- Exported new modules in `ws3.__all__`

### Fixed
- Removed broken duplicate notebook 072 (carbon accounting)
- Resolved DT key/mask matching issues in yield curve loading

### Known Issues
- Textbook chapters 19-20 not yet created (planned but not implemented)
- No dedicated test files for new modules (only import checks)
- CI/CD publish job not configured with PyPI secrets
- Version was previously misreported as 2.0.0 in some documents (now corrected to 1.1.0a1)

## 1.0.0 - 2024-11-24

### Added
- Docstring coverage
- Multiple-format automatic technical documentation system implemented via Sphinx and readthedocs
- Unit tests
- Examples of different purposes
- Add an integrated PuLP optimization module
- A feature to get left handside values of constraints in the optimization module

### Fix
- Fix ws3/cbm connection
- Fix error on populating themes

## 0.0.1 - 2021-10-19
- Initial release
