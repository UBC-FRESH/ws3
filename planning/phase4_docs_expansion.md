# Phase 4: Documentation Expansion and Agent-Friendly Docs

**Parent Issue**: #58 (open)  
**Status**: complete  
**Branch**: `feature/ws3-phase4-docs`
**Completion date**: 2026-07-26

## Overview

Expand ws3 documentation to serve three audiences: new users, advanced users, and LLM coding agents. The docs should also double as an introduction to forest estate modelling for students.

## Goals

1. **New users**: Quick path to installation and first simulation
2. **Students**: Textbook covering forest estate modelling concepts with ws3 examples
3. **Advanced users**: Step-by-step operational guides for complex workflows
4. **LLM coding agents**: Compact technical contracts and onboarding guide

## Structure

```
docs/source/
├── index.rst                          # Landing page with audience navigation
├── getting_started/                   # Installation, quickstart, first model
│   ├── index.rst
│   ├── installation.rst
│   ├── quickstart.rst
│   ├── first_model.ipynb
│   └── architecture_overview.rst
├── textbook/                          # 9-chapter textbook
│   ├── index.rst
│   ├── ch01_forest_estate_models.rst
│   ├── ch02_forest_inventory.rst
│   ├── ch03_growth_and_yield.rst
│   ├── ch04_disturbance_and_actions.rst
│   ├── ch05_optimization_fundamentals.rst
│   ├── ch06_spatial_allocation.rst
│   ├── ch07_financial_analysis.rst
│   ├── ch08_uncertainty_and_scenarios.rst
│   └── ch09_advanced_topics.rst
├── howto/                             # Operational guides
│   ├── index.rst
│   ├── data-preparation.rst
│   ├── curve-definition.rst
│   ├── action-definition.rst
│   ├── running-optimization.rst
│   ├── parallel-optimization.rst
│   ├── spatial-schedule-allocation.rst
│   ├── libcbm-callbacks.rst
│   ├── financial-scenarios.rst
│   ├── custom-area-selector.rst
│   ├── custom-growth-function.rst
│   ├── model-validation.rst
│   └── reproducibility.rst
├── reference/                         # API reference and contracts
│   ├── index.rst
│   ├── data_contract.rst
│   ├── modules/                       # Autodoc-generated
│   └── contracts/                     # Agent-friendly compact facts
│       ├── index.rst
│       ├── repo-runtime-invariants.rst
│       ├── module-responsibilities.rst
│       ├── class-hierarchy.rst
│       └── solver-options.rst
└── guides/                            # Advanced guides
    ├── index.rst
    ├── developer-environment.rst
    ├── coding-agent-onboarding.rst
    ├── extending-ws3.rst
    ├── performance-tuning.rst
    ├── troubleshooting.rst
    ├── limitations-and-boundaries.rst
    └── contributing.rst
```

## Tasks

### Task 4.1 — Restructure documentation with audience-based navigation
- **Status**: Complete
- **Scope**: Rewrite `index.rst` landing page with audience navigation. Create section index files for `getting_started/`, `textbook/`, `howto/`, `reference/`, `guides/`.
- **Acceptance Criteria**: 
  - Landing page has clear navigation for 3 audiences
  - All section index files exist with proper toctrees
  - Legacy chapters still accessible

### Task 4.2 — Create Getting Started section
- **Status**: Complete
- **Scope**: Installation guide, quickstart tutorial, first model walkthrough, architecture overview.
- **Progress**: All 4 pages created with runnable examples and clear progression from installation to first model.

### Task 4.3 — Create textbook for forest estate modelling
- **Status**: Complete
- **Scope**: 9-chapter textbook covering forest inventory, growth/yield, actions/transitions, optimization, spatial allocation, financial analysis, uncertainty, and advanced topics.
- **Acceptance Criteria**: 
  - Each chapter has learning objectives, worked examples, exercises
  - Chapters follow prerequisite chain (visualized with mermaid)
  - Textbook suitable for university course use

### Task 4.4 — Create how-to guides
- **Status**: Complete
- **Scope**: Operational guides for data preparation, curve definition, action definition, optimization, parallel optimization, spatial allocation, libcbm callbacks, financial scenarios, custom selectors, custom growth functions, model validation, and reproducibility.
- **Progress**: All 12 how-to guides created with runnable examples and troubleshooting sections.

### Task 4.5 — Create agent-friendly contract pages
- **Status**: Complete
- **Scope**: Compact technical contracts for LLM coding agents: repo/runtime invariants, module responsibilities, class hierarchy, solver options. Follow femic/fhops pattern.
- **Progress**: All 4 contract pages created (data_contracts, runtime_invariants, module_boundaries, output_format_spec) with clear tables and code examples.

### Task 4.6 — Create coding agent onboarding guide
- **Status**: Complete
- **Scope**: Guide for LLM coding agents to understand ws3 architecture, data flows, and conventions.
- **Acceptance Criteria**: 
  - Explains module responsibilities and data flow
  - Documents class hierarchy and key patterns
  - Includes platform-specific notes (Linux/macOS/Windows)

### Task 4.7 — Create troubleshooting and limitations guides
- **Status**: Complete
- **Scope**: Known issues, recovery procedures, honest documentation of boundaries and external dependencies.
- **Progress**: Created troubleshooting.rst and limitations-and-boundaries.rst with comprehensive coverage of common issues, recovery procedures, and honest documentation of ws3 limitations.

### Task 4.8 — Update conf.py and verify docs build
- **Status**: In Progress
- **Scope**: Enhance Sphinx configuration with new extensions, verify all docs build successfully, ensure GitHub Pages deployment works.
- **Acceptance Criteria**: 
  - `sphinx-build -b html docs/source _build/html` succeeds
  - No warnings or errors
  - GitHub Actions workflow deploys successfully

## Reference

- **femic docs**: https://femic.readthedocs.io (agent-friendly pattern)
- **fhops docs**: https://fhops.readthedocs.io (agent-friendly pattern)
- **Current ws3 docs**: https://ubc-fresh.github.io/ws3/

## Design Principles

1. **Short, focused pages** — each page answers one question
2. **Copy-paste code blocks** — every guide has runnable examples
3. **Tables for specs** — data contracts use list-table for quick reference
4. **Explicit headings** — "Purpose", "Use This Guide For", "Quick Contract"
5. **Platform notes** — Linux/macOS/Windows where relevant
6. **Link forward, don't duplicate** — contracts link to guides, guides link to API
7. **Honest about limits** — document what doesn't work and why
8. **Agent-optimized** — clean RST, minimal cross-references in contracts, max code blocks

## Textbook Pedagogy

Each textbook chapter follows this pattern:

```rst
Chapter X: Title
================

Learning Objectives
-------------------
After reading this chapter, you should be able to:
- Explain what X is and why it matters
- Describe how Y works in the context of forest estate modelling
- Use ws3's Z class/module to implement Y

Key Concepts
------------
Definition-style explanations of core terms...

How It Works in ws3
-------------------
Code examples showing the ws3 implementation...

Worked Example
--------------
A complete, runnable notebook example...

Summary
-------
Bullet-point recap of key takeaways...

Exercises
---------
1. [Easy] ...
2. [Medium] ...
3. [Hard] ...

Further Reading
---------------
Links to textbook chapters, API docs, and external resources.
```

## Prerequisite Chain

```
ch01 (Forest Estate Models)
  → ch02 (Forest Inventory) + ch03 (Growth & Yield)  [parallel]
    → ch04 (Disturbance & Actions)
      → ch05 (Optimization) | ch06 (Spatial Allocation)  [parallel]
        → ch07 (Financial Analysis)
          → ch08 (Uncertainty & Scenarios)
            → ch09 (Advanced Topics)
```

## Audience Mapping

| Audience | Path | Entry Point |
|----------|------|-------------|
| Students / newcomers | Textbook | `textbook/ch01` |
| New ws3 users | Getting Started | `getting_started/quickstart` |
| Intermediate users | Tutorials | `tutorials/` notebooks |
| Advanced users | How-To + Guides | `howto/` + `guides/` |
| LLM coding agents | Contracts + Onboarding | `reference/contracts/` + `guides/coding-agent-onboarding` |
| Researchers | Textbook + Reference | `textbook/` + `reference/` |