# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Common Changelog](https://common-changelog.org/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 1.1.0a1 - 2026-07-26 (alpha)

**Status**: Alpha release for smoke testing. Not yet stable.

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
