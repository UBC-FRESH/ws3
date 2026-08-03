# Phase 10: FEMIC Model Contract and Verification Oracles

Parent program: [FEMIC #305](https://github.com/UBC-FRESH/femic/issues/305)

Companion issue: [ws3 #121](https://github.com/UBC-FRESH/ws3/issues/121)

Branch: `feature/p10-femic-model-contract`

Status: active; implementation slice complete, closeout in progress

## Purpose

Provide the narrow ws3 domain contract that FEMIC can call while FreshForge owns
the surrounding workflow graph. ws3 remains the forest-estate engine and
validator; it does not become a second workflow-orchestration system.

## Scope

- Define a serializable contract or adapter for themes, areas, yields, actions,
  transitions, outputs, horizon, and period length.
- Extract deterministic state from a known-good imported model where that state
  is recoverable.
- Provide deterministic emission or adapter hooks for FEMIC without asking a
  model to generate raw Woodstock syntax.
- Provide verification oracles for lint, import, theme-vector arity,
  development-type/area bindings, yield coverage, action references, transition
  closure, and bounded compile/solve smoke checks where supported.
- Preserve the lazy optional agent boundary and the existing validator-first
  capability contract.

## Boundary

FEMIC owns the Coordinator-facing request, approval policy, workspace manifest,
and domain-facing adapter. FreshForge owns workflow graphs, planning, execution,
and evidence records. ws3 owns the engine-level state and checks that require
real ws3 state.

The model may fill leaf values in a typed structure. Deterministic ws3/FEMIC
code emits files that ws3 reads. Model output is never executed and never enters
the trusted path as unchecked source syntax.

## Adapter ownership decision

The typed construction path is split at the domain boundary:

- **ws3 owns** `ModelSpec`, `ModelBuilder`, deterministic section emitters,
  `ForestModel` import, and engine-level verification that requires real ws3
  state.
- **FEMIC owns** translation from FEMIC CSV/domain inputs into `ModelSpec` or
  bridge section inputs, the Coordinator-facing request, approval policy, and
  workflow/smoke orchestration.
- The FEMIC typed adapter is the authoritative construction path. The legacy
  CSV-to-section bridge may remain as a compatibility path, but it must not
  become a second ws3 workflow engine or bypass the typed boundary silently.

This split keeps ws3 responsible for the engine contract while FEMIC remains
responsible for translating its own domain inputs and coordinating a build.

## Field audit result

The closeout audit found an explicit outcome for every declared surface:

| Surface | Outcome |
| --- | --- |
| Themes, areas, yields | Typed, validated, deterministically emitted and imported. |
| Actions | Typed and imported; `target_age`, `lock_exempt`, and `description` are intentionally not emitted and are reported in `BuildResult.loss`. |
| Transitions | Typed and imported; unsupported `theme_append` and invalid `theme_replace` are rejected before emission; `theme_mask` is reported in `BuildResult.loss` because this slice does not import it. |
| Outputs | Typed, deterministically emitted and imported, including normalized theme indices and output groups. |
| Horizon and period length | Typed and applied during fresh-model construction and period conversion. |
| Extraction and verification | Provided by `ModelContract` and its source/compile-solve oracles. |
| Optional agent boundary | Remains lazy; core `import ws3` does not require `fresh-agent-core` or a live endpoint. |

The loss records are intentional contract evidence, not silent drops. Callers
must inspect `BuildResult.loss` when they provide fields outside the supported
emission subset.

## Verification ladder

- L0: contract/schema validation
- L1: Woodstock keyword and dataset lint
- L2: section import
- L3: structural invariants: theme arity, development types, area bindings,
  yield coverage, action references, and transition closure
- L4: action compilation
- L5: bounded one-period solve or schedule smoke, when optional runtime support is
  available

Every result reports the highest tier cleared. Import success alone is not
model-build success.

## Verified current state

The first implementation slice is complete in the current worktree:

- `ws3.agent.spec` defines a serializable `ModelSpec` construction contract for
  themes, areas, yields, actions, transitions, outputs, horizon, and period
  length, with validation and JSON round-tripping.
- `ws3.agent.emitter` deterministically emits landscape, areas, yields, actions,
  transitions, and outputs section files.
- `ws3.agent.builder` validates the spec, rejects unsupported transition
  features before emission, imports a fresh `ForestModel`, applies period-to-year
  conversion, and reports unsupported action metadata in `BuildResult.loss`.
- `ModelContract` already provides deterministic extraction plus source and
  compile/solve verification oracles.
- Output parsing now handles unthemed outputs, one-based Woodstock theme indices,
  comma-separated output groups, and the final output at end-of-file.
- The active P10 slice has 76 focused tests, 406 passing full-suite tests with
  9 skips, and 10 passing FEMIC bridge tests when FEMIC is run with its local
  `src` directory on `PYTHONPATH`.

The implementation is still uncommitted on
`feature/p10-femic-model-contract`. Existing local `runtime/`, `tmp/`, and the
empty `=` file are working-tree material and are not part of this closeout.

## Closeout plan

### Task 10.1 — Synchronize the Phase 10 record

- [x] Replace the original immediate-task placeholder with the implemented
  construction boundary and verification evidence.
- [x] Record the current branch and implementation status in the roadmap.
- [x] Update GitHub issue #121 with the same verified state and child-task
  checklist.

### Task 10.2 — Audit the typed contract boundary

- [x] Compare every declared Phase 10 field with `ModelSpec`, `ModelBuilder`,
  `BuildResult`, and the FEMIC adapter call site.
- [x] Decide whether adapter placement belongs in ws3, FEMIC, or both, and record
  the ownership rule without adding a second workflow engine to ws3.
- [x] Identify any intentionally lossy fields and require an explicit loss
  record or rejection for each one.
- [x] Confirm no concrete boundary gap requires an implementation edit; do not
  broaden the contract based on hypothetical Woodstock features.

### Task 10.3 — Preserve and extend verification evidence

- [x] Keep the ws3 focused and full-suite results reproducible from this branch.
- [x] Keep the FEMIC bridge regression byte-for-byte comparison and imported
  action/transition assertions reproducible without making FEMIC a ws3 runtime
  dependency.
- [x] Add or link the cross-repository bridge evidence to FEMIC #305 and ws3
  #121.
- [x] Confirm optional agent imports remain lazy and core `import ws3` remains
  independent of `fresh-agent-core`.

Evidence recorded on 2026-08-03:

- `python -m pytest tests/test_spec.py -q`: 76 passed.
- `python -m pytest -q`: 406 passed, 9 skipped, 4 warnings.
- Focused Ruff check over the P10 modules and tests: all checks passed.
- `cd ../femic && PYTHONPATH=src python -m pytest tests/test_ws3_bridge.py -q`:
  10 passed.
- `python -c "import ws3; import ws3.agent; ..."`: core import succeeded and
  `ws3.agent.available()` returned `False` without requiring the optional agent
  runtime.

### Task 10.4 — Resolve bounded implementation gaps

- [x] Audit evidence for defects demonstrated by Task 10.2 or Task 10.3; none
  were found.
- [x] Record an explicit no-op rather than inventing a repair or speculative
  contract expansion.
- [x] Preserve regression tests for output round-trip and
  period conversion behavior.
- [x] Do not mix Phase 11 Ruff debt cleanup into this branch or reformat
  unrelated legacy code.

Task result: no implementation edit is warranted by the completed audit. The
existing action/transition loss records and pre-emission rejection behavior are
the documented contract for this slice.

### Task 10.5 — Prepare phase closeout

- [x] Run the final ws3 and FEMIC verification commands and capture results.
- [x] Update `ROADMAP.md`, `CHANGELOG.md`, this plan, and issue #121 so their
  status and evidence agree.
- [x] Review the complete diff, preserving unrelated user worktree material.
- [x] Ask the maintainer before the commit/PR step; approval was received.
  Merge and parent-issue closeout remain pending review and merge.

Final closeout evidence recorded on 2026-08-03:

- `python -m build`: built `ws3-1.1.0a4.tar.gz` and
  `ws3-1.1.0a4-py3-none-any.whl` successfully.
- `git diff --check`: passed.
- The Sphinx HTML build generated output but reported 302 pre-existing warnings
  under `-W`; this is a documentation backlog and is not a Phase 10 code
  failure.
- Worktree review found five modified tracked files and the expected new P10
  implementation/test/plan files. Existing local artifacts (`=`, `runtime/`,
  and `tmp/foo.txt`) remain untouched.
- Commit `8369a68` contains 10 intended files, 3,631 insertions, and 12
  deletions. The branch is published as
  `feature/p10-femic-model-contract` and PR #127 is open against `main`.

Maintainer decision gate: approval was received for the commit and PR. No
merge or parent issue closure has been performed.

Closeout state: implementation and technical verification are complete; PR #127
is open for review. The Sphinx warning backlog should be handled separately from
Phase 10.

The next bounded action is review and merge decision for PR #127. The parent
issue remains open until that merge is complete.

## Acceptance criteria

- [x] Serializable contract covers themes, areas, yields, actions, transitions,
  outputs, horizon, and period length.
- [x] Deterministic extraction and structural verification are available.
- [x] Deterministic emission/import builds a fresh model without unchecked model
  output entering the trusted path.
- [x] Verification ladder coverage is explicit through L4, with L5 deferred
  when no optimization problem or optional runtime is available.
- [x] Unsupported features are rejected or recorded as loss; they are not
  silently discarded.
- [x] Existing optional agent boundaries and validator-first capabilities remain
  intact.
- [x] Adapter ownership and any remaining lossy fields are documented.
- [x] Cross-repository evidence is linked from the phase records.
- [x] Final verification and closeout records are synchronized.

## Verification commands

```bash
python -m pytest tests/test_spec.py -q
python -m pytest -q
python -m ruff check ws3/agent/__init__.py ws3/agent/spec.py \
  ws3/agent/emitter.py ws3/agent/builder.py tests/test_spec.py
cd ../femic
PYTHONPATH=src python -m pytest tests/test_ws3_bridge.py -q
```

The full repository Ruff backlog is tracked separately in Phase 11. It is not
a Phase 10 acceptance gate for this implementation branch.

## Risks

- Woodstock sections may not contain enough information for a lossless round
  trip; document any required contract superset.
- Compile/solve checks may depend on optional solver/runtime dependencies; report
  unavailable tiers explicitly rather than treating them as passes.
- Existing untracked runtime files in the checkout are local working material
  and must remain untouched.
