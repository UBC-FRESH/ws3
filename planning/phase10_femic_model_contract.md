# Phase 10: FEMIC Model Contract and Verification Oracles

Parent program: [FEMIC #305](https://github.com/UBC-FRESH/femic/issues/305)

Companion issue: [ws3 #121](https://github.com/UBC-FRESH/ws3/issues/121)

Branch: `feature/p10-femic-model-contract`

Status: planned

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

## Immediate bounded task

Define the typed contract and workspace-facing adapter boundary, then write the
first public-fixture extraction/import verification test. Do not add an LLM
provider in this task.

## Risks

- Woodstock sections may not contain enough information for a lossless round
  trip; document any required contract superset.
- Compile/solve checks may depend on optional solver/runtime dependencies; report
  unavailable tiers explicitly rather than treating them as passes.
- Existing untracked runtime files in the checkout are local working material
  and must remain untouched.
