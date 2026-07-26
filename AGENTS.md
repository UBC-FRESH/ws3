# AGENTS.md

This file is the working contract for AI coding agents in this repository.

## Repository Purpose

`ws3` is a Python package for landscape-level wood supply simulation and related forest planning analyses. The durable output of this repository is not only code, but also reproducible models, tests, documentation, examples, and clear project workflow records.

The repo should stay focused on scientific software quality, maintainability, and transparent development practice. Keep the implementation grounded in the package’s domain while preserving clear engineering discipline and evidence-based verification.
Stay generic across the UBC-FRESH ecosystem. Do not encode private project assumptions as core rules.
## Working Principles

**Evidence over trust.** Treat a worker's or supervisor's prose report as untrusted until you verify the underlying repo, filesystem, or GitHub state. Require evidence for completion claims: diffs, command output, issue URLs, or inspected artifacts. Never trust a "done" without the proof.

**One termination artifact.** The Coordinator produces the final gate and writes (or synthesizes) the result that gets committed — workers produce intermediate output, but the Coordinator owns the deliverable. One bounded repair per worker task: if it fails, issue exactly one concrete repair follow-up naming the specific defect and exact files/lines to fix. If the second attempt fails, escalate to the developer — do not try a third time or do the work yourself.

**Signal over enforcement.** Quantitative metrics (yield gates, budget caps, quality thresholds) are informative signals, not enforcement cliffs. The thin Coordinator lays them down as tripwires — not rails that the workflow must bend to. Gating is a mirror, not a hammer.

**Single model, bounded authority.** All roles (Coordinator, Supervisor, Worker, Advisor) share one configured remote vLLM model. Role separation comes from bounded instructions and authority, not from pretending the underlying model is deterministic. Concurrency is free; serialization is policy.

**Gates that measure the right thing.** Design metrics to isolate what you care about. A yield gate on content-bearing records measures extraction quality. A yield gate on all records (including structural scaffolding like TOC entries) measures document structure noise. The gate should measure the thing you're gating.

**Workflow as scaffold, not retrofit.** Build the scaffolding (issues, branch, planning note) before the work, not after. The workflow exists so the scaffolding is the foundation, not the fire escape you retrofit when you realize you're in the air.

**Structured handoff, not memory trace.** Issue bodies, planning notes, and CHANGE_LOG entries are the durable handoff surface. A new person (or agent) should pick up the thread from the artifacts, not from a conversation memory trace.

**Preserve uncertainty.** If evidence is missing, report a blocker. A workflow result is only as strong as its declared inputs, and verification is only as strong as the evidence it inspects.

**Sanitized output.** Treat raw transcripts, credentials, and private paths as local working material. Promote only sanitized, public-safe findings to the tracked repository.

**Scope discipline.** Keep changes scoped to the active roadmap phase and issue. Do not drift into adjacent work without explicit maintainer approval.

## Repository Layout

- `README.md` — public overview and package entry point.
- `CONTRIBUTING.md` — contributor workflow and repository norms.
- `ROADMAP.md` — current roadmap phases, tasks, and progress tracking.
- `CHANGELOG.md` — append-only narrative of notable changes.
- `pyproject.toml` — packaging and dependency metadata.
- `ws3/` — importable package implementation.
- `tests/` — automated tests.
- `docs/` — Sphinx documentation source.
- `examples/` — public-safe examples and tutorials.
- `planning/` — sanitized planning notes.
- `tmp/`, `runtime/`, `local/`, `outputs/` — ignored local working areas.

Do not claim the repo contains a package, CI, benchmark harness, or extension until a later phase records that evidence.

## Planning Workflow

This repository follows the UBC-FRESH phase/task/subtask workflow:

- `ROADMAP.md` is the current plan and issue tracker map.
- One roadmap phase maps to one GitHub parent issue and one feature branch.
- One roadmap task maps to one child GitHub issue linked from the parent issue body.
- Subtasks usually stay as checklist items inside the child issue body.
- Use at most three issue levels: phase, task, implementation subtask.
- Record issue numbers beside roadmap phases and tasks once created.
- Keep `ROADMAP.md`, `CHANGELOG.md`, planning notes, issue bodies, and PR descriptions synchronized.
- Open a PR from the phase branch to `main` only after phase tasks, tests, docs, and closeout notes are complete or explicitly deferred.

## Strict Development Workflow

Use this workflow for active development from the first phase boundary onward:

- One active roadmap phase should generally correspond to one GitHub parent issue and one feature branch.
- Create or activate the GitHub parent issue before starting a roadmap phase.
- Create the feature branch from current `main` for that parent issue.
- Create child issues for roadmap tasks under the parent issue.
- Document task subtasks as checklist steps inside the child issue body unless they are large enough to deserve third-level implementation issues.
- Work child issues one at a time where practical, usually in roadmap order.
- Before closing a child issue, update every issue-body checklist item to checked, or rewrite the issue body to make explicitly clear which items were superseded or are not applicable.
- Close each child issue only after its repo changes, documentation, issue-body checklist, and verification for that task are complete.
- Keep `ROADMAP.md`, `CHANGELOG.md`, and issue comments synchronized as task state changes.
- Open a PR from the phase branch back to `main` when the parent issue's child issues are complete or explicitly deferred.
- Close the parent issue only after the PR has merged back to `main`.
- Do not start a new active parent issue and branch until the current parent issue is closed, unless the maintainer explicitly approves a parallel lane.

## GitHub Issue And Comment Formatting

Formatting matters. GitHub issue bodies and comments must be readable as rendered Markdown, not flattened prose.

Rules:

- Use short section labels on their own lines, such as `Roadmap task: P3.1`, `Parent phase issue: #18`, `Status: active`, and `Checklist:`.
- Use real GitHub task-list syntax, with one checklist item per line.
- Never write inline pseudo-checklists such as `Checklist: [ ] first. [ ] second.`
- Wrap branch names, file paths, commands, and commit hashes in backticks.
- For parent phase issues, list child issues as task-list bullets with issue numbers and task IDs.
- Before creating or editing several issues, prepare bodies as multi-line Markdown strings or temporary body files.

## GitHub Issue Body Quality Standard

Issue bodies are part of the project specification and onboarding material. Write them so a new lab student, external collaborator, or coding agent can understand the task, implement it, verify it, and close it without reading the original chat transcript.

Parent phase issues must include phase identifier, status, branch name, roadmap links, goal, scope, out-of-scope boundaries, architecture notes, child task checklist, acceptance criteria, verification, and closeout requirements.

Child task issues must include task identifier, parent phase issue, status, related planning links, goal, scope, out-of-scope boundaries, subtasks, acceptance criteria, verification commands, artifacts, risks, and completion metadata once closed.

Do not create placeholder issue bodies with only a title and a short checklist unless the maintainer explicitly asks for a placeholder.

## Development Workflow Constraints

- Do not commit, push, create PRs, or mutate GitHub without approval.

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
