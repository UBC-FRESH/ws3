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
- Status: in_progress
- Branch: `feature/ws3-phase4-docs`

### Task 4.1 — Restructure documentation with audience-based navigation
- Status: complete
- Scope: rewrite `index.rst` landing page with audience navigation (new users, advanced users, LLM agents). Create `getting_started/`, `textbook/`, `howto/`, `reference/`, `guides/` sections.
- Verification: Docs build successfully, landing page has audience navigation, all section indexes created.

### Task 4.2 — Create Getting Started section
- Status: not_started
- Scope: installation guide, quickstart tutorial, first model walkthrough, architecture overview.

### Task 4.3 — Create textbook for forest estate modelling
- Status: complete
- Scope: 16-chapter textbook covering forest inventory, growth/yield, actions/transitions, optimization, spatial allocation, financial analysis, uncertainty, advanced topics, carbon modelling, FEMIC integration, fhops integration, FreshForge workflow automation, SpaDES integration, disturbance modelling (stub), and supply chain integration (stub). Each chapter includes learning objectives, worked examples, and exercises.
- Progress: All 16 chapters complete (ch01-ch16). Chapters 15-16 are stubs awaiting detailed content.

### Task 4.4 — Create how-to guides
- Status: in_progress
- Scope: operational guides for data preparation, curve definition, action definition, optimization, parallel optimization, spatial allocation, libcbm callbacks, financial scenarios, custom selectors, custom growth functions, model validation, and reproducibility.
- Progress: Guide index created. Individual guide content pending.

### Task 4.5 — Create agent-friendly contract pages
- Status: in_progress
- Scope: compact technical contracts for LLM coding agents: repo/runtime invariants, module responsibilities, class hierarchy, solver options. Follow femic/fhops pattern.
- Progress: Coding agent onboarding guide complete. Additional contract pages (module details, solver options) pending.

### Task 4.6 — Create coding agent onboarding guide
- Status: complete
- Scope: guide for LLM coding agents to understand ws3 architecture, data flows, and conventions. Include purpose, use cases, quick contract table, and platform-specific notes.
- Verification: Guide includes module map, class hierarchy (mermaid), data flow diagram, common patterns, and platform notes.

### Task 4.7 — Create troubleshooting and limitations guides
- Status: in_progress
- Scope: known issues, recovery procedures, honest documentation of boundaries and external dependencies.
- Progress: Guide index created. Content pending.

### Task 4.8 — Update conf.py and verify docs build
- Status: not_started
- Scope: enhance Sphinx configuration with new extensions, verify all docs build successfully, ensure GitHub Pages deployment works.
