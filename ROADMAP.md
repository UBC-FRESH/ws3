# Roadmap

This roadmap tracks the current UBC-FRESH-style development workflow for ws3.

## Phase 1 — Establish repository agent and contribution contract

- Parent issue: to be created
- Status: complete
- Branch: `feature/ws3-agent-contract`

### Task 1.1 — Add repository-level agent guidance
- Status: complete
- Scope: create `AGENTS.md` with repository-specific operating guidance for AI coding agents.

### Task 1.2 — Add contributor workflow contract
- Status: complete
- Scope: update `CONTRIBUTING.md` to formalize roadmap-based development, issue hygiene, and verification expectations.

### Task 1.3 — Document roadmap and changelog maintenance
- Status: complete
- Scope: add roadmap tracking notes and keep `CHANGELOG.md` aligned with repository progress.

## Phase 2 — Refactor ws3 toward a fully typed Python codebase

- Parent issue: #53
- Status: complete
- Branch: `feature/ws3-typed-python-refactor`

### Task 2.1 — Add typing infrastructure and package conventions
- Status: complete
- Scope: introduce typing-oriented tooling, package conventions, and a lightweight validation workflow for the repository.
- Child issue: #54

### Task 2.2 — Migrate core modeling modules to typed interfaces and explicit data contracts
- Status: complete
- Scope: incrementally refactor core modules such as `common.py`, `core.py`, `forest.py`, `opt.py`, and `spatial.py` to use explicit type hints and clearer contracts.
- Child issue: #55

### Task 2.3 — Add validation and quality gates for refactor progress
- Status: complete
- Scope: add reproducible checks that keep the refactor measurable and regression-resistant as the migration progresses.
- Child issue: #56

## Phase 3 — Performance, Validation, and Documentation Infrastructure

- Parent issue: #57
- Status: complete
- Branch: `feature/ws3-phase3`

### Task 3.1 — Performance optimizations
- Status: complete
- Scope: optimize Curve arithmetic, forest simulation loops, and optimization solver integration for improved runtime performance.

### Task 3.2 — Enhanced validation and error handling
- Status: complete
- Scope: add comprehensive input validation, error messages, and test coverage (62 tests passing).

### Task 3.3 — Documentation and examples
- Status: complete
- Scope: expand docstrings, add inline documentation, and create example notebooks (1,900+ lines of docs/examples).

### Task 3.5 — LP matrix generation optimization
- Status: complete
- Scope: optimize linear programming matrix generation for faster optimization problem construction.

### Task 3.6 — Notebook verification and critical bug fix
- Status: complete
- Scope: verify all 12 Jupyter notebooks execute successfully and fix critical bug in Curve arithmetic operators.

### Task 3.7 — Sphinx documentation and GitHub Pages deployment
- Status: complete
- Scope: set up Sphinx documentation with automated GitHub Pages deployment, matching femic/freshforge/fhops pattern. Deployed at https://ubc-fresh.github.io/ws3/.

## Phase 4 — Documentation Expansion and Agent-Friendly Docs

- Parent issue: #58
- Status: complete
- Branch: `feature/ws3-phase4-docs`
- Completion date: 2026-07-26

### Task 4.1 — Restructure documentation with audience-based navigation
- Status: complete
- Scope: rewrite `index.rst` landing page with audience navigation (new users, advanced users, LLM agents). Create `getting_started/`, `textbook/`, `howto/`, `reference/`, `guides/` sections.
- Verification: Docs build successfully, landing page has audience navigation, all section indexes created.

### Task 4.2 — Create Getting Started section
- Status: not_started
- Scope: installation guide, quickstart tutorial, first model walkthrough, architecture overview.

### Task 4.3 — Create textbook for forest estate modelling
- Status: complete
- Scope: 16-chapter textbook covering forest inventory, growth/yield, actions/transitions, optimization, spatial allocation, financial analysis, uncertainty, advanced topics, carbon modelling, FEMIC integration, fhops integration, FreshForge workflow automation, SpaDES integration, disturbance modelling, and supply chain integration. Each chapter includes learning objectives, worked examples, and exercises.
- Progress: All 16 chapters complete (ch01-ch16) with substantive chapter content across the full sequence.

### Task 4.4 — Create how-to guides
- Status: complete
- Scope: operational guides for data preparation, curve definition, action definition, optimization, parallel optimization, spatial allocation, libcbm callbacks, financial scenarios, custom selectors, custom growth functions, model validation, and reproducibility.
- Progress: All 12 how-to guides created with runnable examples and troubleshooting sections.

### Task 4.5 — Create agent-friendly contract pages
- Status: complete
- Scope: compact technical contracts for LLM coding agents: repo/runtime invariants, module responsibilities, class hierarchy, solver options. Follow femic/fhops pattern.
- Progress: All 4 contract pages created (data_contracts, runtime_invariants, module_boundaries, output_format_spec).

### Task 4.6 — Create coding agent onboarding guide
- Status: complete
- Scope: guide for LLM coding agents to understand ws3 architecture, data flows, and conventions. Include purpose, use cases, quick contract table, and platform-specific notes.
- Verification: Guide includes module map, class hierarchy (mermaid), data flow diagram, common patterns, and platform notes.

### Task 4.7 — Create troubleshooting and limitations guides
- Status: complete
- Scope: known issues, recovery procedures, honest documentation of boundaries and external dependencies.
- Progress: Created troubleshooting.rst and limitations-and-boundaries.rst with comprehensive coverage.

### Task 4.8 — Update conf.py and verify docs build
- Status: complete
- Scope: enhance Sphinx configuration with new extensions, verify all docs build successfully, ensure GitHub Pages deployment works.
- Verification: Docs build with zero errors and zero warnings.

## Phase 5 — Advanced Features and Production Deployment

- Parent issue: #60
- Status: in_progress
- Branch: `feature/ws3-phase5`
- Start date: 2026-07-26

### Task 5.1 — Advanced Modeling Features
- Status: not_started
- Scope: stochastic optimization, multi-objective optimization, dynamic planning, climate scenarios, enhanced carbon accounting

### Task 5.2 — User Experience Improvements
- Status: not_started
- Scope: interactive notebooks, GUI wrapper, tutorial videos, FAQ section, migration guide

### Task 5.3 — Performance Optimization
- Status: not_started
- Scope: solver tuning, memory profiling, parallel processing, incremental solving, caching

### Task 5.4 — Integration Enhancements
- Status: not_started
- Scope: fhops integration, FEMIC integration, FreshForge workflows, SpaDES coupling, API endpoints

### Task 5.5 — Production Deployment
- Status: not_started
- Scope: release packaging, CI/CD pipeline, versioning, changelog, community guidelines, support channels

### Task 5.6 — Additional How-To Guides
- Status: not_started
- Scope: advanced optimization, custom solvers, data validation, scenario analysis, reporting

### Task 5.7 — Textbook Expansion
- Status: not_started
- Scope: advanced spatial modeling, carbon accounting in detail, case studies, future directions, more exercises

### Task 5.8 — Testing and Validation
- Status: not_started
- Scope: unit tests, integration tests, performance tests, regression tests, documentation tests
