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
- Branch: `feature/ws3-typed-python-refactor` (merged to dev)
- PR: #57 (squash merged)

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
<<<<<<< HEAD

---

## Phase 3 — Enhanced Features and Optimizations

- Parent issue: to be created
- Status: active
- Branch: `feature/ws3-phase3-enhancements`

### Task 3.1 — Performance optimizations for critical paths
- Status: planned
- Scope: profile and optimize performance-critical code paths, particularly in forest simulation and optimization modules.
- Child issue: to be created

### Task 3.2 — Enhanced validation and error handling
- Status: planned
- Scope: add runtime validation for critical operations, improve error messages, and add debugging tools.
- Child issue: to be created

### Task 3.3 — Documentation and examples
- Status: planned
- Scope: create comprehensive user documentation, API reference, and usage examples.
- Child issue: to be created

### Task 3.4 — Additional features and integrations
- Status: planned
- Scope: explore and implement additional features based on user feedback and project needs.
- Child issue: to be created
=======
>>>>>>> origin/dev
