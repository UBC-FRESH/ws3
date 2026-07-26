# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Common Changelog](https://common-changelog.org/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 2.0.0 - 2026-07-26

### Added
- Phase 5 interactive notebooks (070-075) for advanced workflows
- FAQ section in documentation (`docs/source/howto/faq.rst`)
- Migration guide from Woodstock to ws3 (`docs/source/howto/migration_from_woodstock.rst`)
- Multi-objective optimization examples
- Spatial constraints and adjacency modeling
- Parallel optimization and performance benchmarking
- Scenario analysis and comparison workflows

### Changed
- Updated ROADMAP.md to reflect Phase 5 progress
- Enhanced documentation with 14 how-to guides (was 12)

### Fixed
- Removed broken duplicate notebook 072 (carbon accounting)
- Resolved DT key/mask matching issues in yield curve loading

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
