# AGENTS.md

This file is the working contract for AI coding agents in this repository.

## Repository Purpose

`ws3` is a Python package for landscape-level wood supply simulation and related forest planning analyses. The durable output of this repository is not only code, but also reproducible models, tests, documentation, examples, and clear project workflow records.

The repo should stay focused on scientific software quality, maintainability, and transparent development practice. Keep the implementation grounded in the package’s domain while preserving clear engineering discipline and evidence-based verification.

## Working Principles

- Evidence over trust. Treat prose reports and local assumptions as unverified until they are checked against the repository, tests, or runtime output.
- Scope discipline. Keep changes aligned with the active roadmap phase and issue. Avoid broad detours unless the maintainer explicitly expands scope.
- Preserve uncertainty. A result is only as strong as its declared inputs, assumptions, and verification evidence.
- Prefer small, reviewable changes over large speculative rewrites.
- Keep public repo content free of private data, credentials, machine-specific paths, and unpublished notes.
- Treat examples, notebooks, and generated artifacts as local working material unless they are intentionally committed as sanitized, tracked examples.

## Repository Layout

- `README.md` — public overview and package entry point.
- `CONTRIBUTING.md` — contributor workflow and repository norms.
- `ROADMAP.md` — current roadmap phases, tasks, and progress tracking.
- `CHANGELOG.md` — append-only narrative of notable changes.
- `pyproject.toml` — packaging and dependency metadata.
- `src/ws3/` — importable package implementation.
- `tests/` — automated tests.
- `docs/` — Sphinx documentation source.
- `examples/` — public-safe examples and tutorials.

## Planning Workflow

This repository follows the UBC-FRESH phase/task/subtask workflow:

- `ROADMAP.md` is the current plan and issue tracker map.
- One roadmap phase maps to one GitHub parent issue and one feature branch.
- One roadmap task maps to one child GitHub issue linked from the parent issue body.
- Subtasks should be tracked as checklist items inside the child issue body unless they are large enough to warrant separate implementation issues.
- Keep `ROADMAP.md`, `CHANGELOG.md`, planning notes, issue bodies, and pull request descriptions synchronized.
- Before starting a non-trivial task, document the plan in `ROADMAP.md` under the relevant phase/task structure rather than keeping the plan only in chat.

## Development Workflow Constraints

- Use a dedicated feature branch for each roadmap phase.
- Prefer branch names such as `feature/ws3-<phase-id>-<short-name>`.
- Keep the implementation scoped to the current phase and child task.
- Update the roadmap and changelog as progress is made.
- Make verification part of the work rather than an afterthought.

## GitHub Issue and Comment Formatting

- Use short section labels on their own lines, such as `Roadmap task: P1.1`, `Parent phase issue: #N`, `Status: active`, and `Checklist:`.
- Use real Markdown task lists, not inline pseudo-checklists.
- Wrap branch names, file paths, commands, and commit hashes in backticks.
- Write issue bodies so another contributor can understand the goal, scope, acceptance criteria, and verification steps without reading the conversation transcript.

## Verification

Default local checks should include:

```bash
python -m pytest
python -m ruff check .
python -m build
sphinx-build -b html docs _build/html -W
```

For user-facing behaviour changes, add or update documentation in the same change set whenever practical.
