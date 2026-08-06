# AGENTS.md

This file is the working contract for AI coding agents in this repository.

## Repository Purpose

`ws3` is a Python package for landscape-level wood supply simulation and related forest planning analyses. The durable output of this repository is not only code, but also reproducible models, tests, documentation, examples, and clear project workflow records.

The repo should stay focused on scientific software quality, maintainability, and transparent development practice. Keep the implementation grounded in the package’s domain while preserving clear engineering discipline and evidence-based verification.
Stay generic across the UBC-FRESH ecosystem. Do not encode private project assumptions as core rules.
## Working Principles

**Evidence over trust.** Treat a worker's or supervisor's prose report as untrusted until you verify the underlying repo, filesystem, or GitHub state. Require evidence for completion claims: diffs, command output, issue URLs, or inspected artifacts. Never trust a "done" without the proof.

**One termination artifact.** The Coordinator produces the final gate and writes (or synthesizes) the result that gets committed — workers produce intermediate output, but the Coordinator owns the deliverable. When validation fails, identify the concrete defect and the evidence needed to establish success. Continue, reassign, or retry while work remains evidence-based and within authority and safety constraints. Escalate only for an actual blocker, unsafe or ambiguous action, authority boundary, or required developer decision.

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

## Operating ws3 As An Agent

**Prefer the capability surface over hand-written API calls.**

`ws3.agent` exposes a small set of operations whose output is validated against
real ws3 state before you ever see it. When one of them covers what you need, use
it. Composing the Python API from memory is the fallback, not the default.

| Capability | What it validates |
|---|---|
| `build_mask` | The proposed mask resolves against the `ForestModel` to at least one development type |
| `explain_exception` | Every ws3 symbol the explanation cites actually exists in the installed package |
| `diagnose_import` | The suggested fix is applied to a scratch copy and the section genuinely re-imports |
| `rtfm` | The capability name returned is real; cited doc URLs return HTTP 200 |
| `ws3_hint` | Every cited ws3 symbol exists; every cited doc URL returns HTTP 200 |
| `inspect_model` | Live ``ForestModel`` metadata (base year, horizon/periods, period length, theme/action/dtype counts, total area at period 1) — read-only, validated against the actual in-memory model object |

Available over MCP:

```bash
ws3-agent-mcp --model-path <dir> --model-name <name>
```

#### IPython / Jupyter

In any IPython kernel or Jupyter notebook where a ``ForestModel`` named ``fm`` is
in scope::

```python
%load_ext ws3.agent.ipython_magics
%ws3_capabilities
%ws3_inspect_model
%ws3_hint How do I add a fire disturbance?
%build_mask all dead stands
%explain_exception KeyError: 'theme not found'
```

The ``fm`` object is discovered automatically. No explicit model argument is needed.
Requires ``pip install ws3[agent]``.

For agent-workbench coordinators, add to your VS Code or Claude Desktop `mcpServers` config:

```json
{
  "mcpServers": {
    "ws3": {
      "command": "ws3-agent-mcp",
      "args": ["--model-path", "/srv/shared-data/gep/jupyterhub04-projects/ws3/examples/data/woodstock_model_files_tsa24_clipped", "--model-name", "tsa24_clipped"]
    }
  }
}
```

Or from Python:

```python
import ws3.agent

if ws3.agent.available():
    result = ws3.agent.run('build_mask', 'mature spruce stands', context=fm)
    if result.ok:
        ...              # result.value is validated
    else:
        ...              # result.errors says why every attempt was rejected
```

Requires `pip install ws3[agent]`. Optional: core ws3 modelling never needs it,
and `import ws3` never loads it.

### What the guarantee is, and what it is not

A capability returns validated output or it returns nothing. It never returns a
best guess. `result.ok is False` means every attempt was rejected by the validator,
and `result.errors` says why — that is information, not an error to route around.

Capabilities are **advisory**. They return proposals; applying them is the
caller's decision. Nothing mutates a model in place.

### The rule for adding one

> **No oracle, no capability.**

A capability is a prompt plus a validator plus an evidence-driven continuation policy. The validator must
check the proposal against real state — resolve the mask, re-parse the file,
confirm the symbol exists. Validating model output against another model, against
a regex over its own text, or against a mock proves nothing.

Write the validator first. If you cannot write one that can actually fail, the
thing you are building is not a capability, and adding it would quietly convert a
trustworthy surface into a plausible-sounding one.

This is not a stylistic preference. Fabricated APIs reached the documentation
(Phase 6), the test suite (Phase 7.5), and shipped module code (Phase 7.6) in this
repository. The `explain_exception` validator exists specifically because that
failure mode is well attested here.


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
