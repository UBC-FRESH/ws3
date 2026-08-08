# Task 8.7 Discovery — WS3 Read-Only AAM Vertical Slice

**Date**: 2026-08-06
**Status**: active implementation
**Parent issue**: [#105](https://github.com/UBC-FRESH/ws3/issues/105)
**Child issue**: [#149](https://github.com/UBC-FRESH/ws3/issues/149)

This report records discovery evidence and the approved first implementation
slice. It is not MCP deployment approval, package-integration approval, or
approval to enable mutating, expensive, or long-running operations.

## Recommended Journey

Orient a user to an explicitly selected WS3 model and preview a natural-language
stand selection without applying actions, schedules, optimization, or other model
changes.

The proposed conceptual flow is:

1. Inspect metadata for the explicitly selected model.
2. Accept a natural-language stand-selection description.
3. Build a Woodstock-style mask and validate it against live model state.
4. Summarize verified metadata, the returned mask, validation status,
   attempts/errors, and provenance.
5. Provide a review-only next-step draft without executing it.

The first slice stops after the reviewable summary and draft. A human must review
the interpretation and decide whether any later scientific or modelling action is
appropriate.

## Verified Existing Surfaces

The following repository surfaces were inspected or supplied as verified discovery
evidence:

- [`inspect_model`](../ws3/agent/capabilities/inspect_model.py) is a bounded,
  read-only metadata executor.
- [`build_mask`](../ws3/agent/capabilities/build_mask.py) builds a proposed mask
  and validates it through [`ForestModel.unmask`](../ws3/forest.py); it does not
  apply the mask.
- [`ws3_hint`](../ws3/agent/capabilities/ws3_hint.py) is advisory. Its partial
  oracle validates cited symbols and documentation URLs, not semantic truth or
  completeness.
- [`__init__.py`](../ws3/agent/capabilities/__init__.py) contains the actual
  seven-capability current registry: the historical six (`build_mask`,
  `explain_exception`, `diagnose_import`, `rtfm`, `ws3_hint`, and
  `inspect_model`) plus `report_scenario_inventory_products`.
- The public capability contract and IPython usage are documented in
  [`agent-capabilities.rst`](../docs/source/guides/agent-capabilities.rst).
- [`agent_capability_example.py`](../examples/agent_capability_example.py)
  provides an offline worked example.
- Relevant evidence is covered by
  [`test_agent_inspect_model.py`](../tests/test_agent_inspect_model.py),
  [`test_agent_capabilities.py`](../tests/test_agent_capabilities.py), and
  [`test_agent_mcp.py`](../tests/test_agent_mcp.py), including inspection
  no-mutation, mask validation/retry/provenance, schema/context, and transport
  concerns.

These links establish existing package surfaces and evidence locations; they do
not establish a new summary API, skill package, server deployment, or cross-package
contract.

## MCP Tool And Skill Responsibilities

### Existing capability/tool responsibilities

- `inspect_model` supplies bounded metadata grounded in a live `ForestModel`.
- `build_mask` supplies a proposed mask only after package-owned live-state
  validation. A failed validation remains a failure with its attempts/errors;
  it is not a best guess.
- `ws3_hint` may provide a citation-checked advisory hint, but its validator does
  not prove the hint's semantic completeness and it is not required for the first
  slice.
- The capability registry and existing MCP-facing contract remain the source of
  truth for what is executable. No new tool name or schema is proposed here.

### Proposed skill/composition responsibilities

A workflow skill would manage the user goal, explicit model selection, sequence,
review points, provenance presentation, uncertainty, and the stop condition. It
could compose the existing inspection and mask-validation capabilities, then
format a review-only summary and next-step draft. That composition is proposed
discovery design, not an existing API or implementation.

The skill must not claim that a mask was applied, run schedules or optimization,
interpret scientific meaning on the user's behalf, or convert an advisory hint
into a validated result. Summary/provenance artifact format and the boundary
between a loaded notebook `ForestModel` and MCP-side loading remain unresolved.

## Safety And Trust Boundary

- The model path and model name must be explicit and limited to the selected
  artifact; path-redaction policy is still an open decision.
- Inspection and mask validation are read-only. No action, schedule, simulation,
  optimization, plotting, model-file mutation, or arbitrary time-series inspection
  is included.
- No credentials or hidden provider configuration are accepted, exposed, or
  written to provenance.
- Package-owned validation must remain in the evidence path. Generated prose is
  not an oracle.
- Attempts, validation errors, uncertainty, and provenance must remain visible;
  an unsupported or unresolved request must stop rather than produce a plausible
  fallback.
- Human review is required before scientific interpretation or downstream use.
- MCP deployment, package integration, and mutating or expensive operations need
  a later explicit approval.

## Evaluation And Acceptance Signals

Discovery should be considered successful only when evidence shows that:

- the user supplies an explicit model selection and the reported metadata matches
  live model state;
- a representative stand-selection description produces a nonempty, validated
  mask or a clear structured failure;
- no model or model-file state changes occur during inspection and validation;
- the summary distinguishes verified observations, attempts/errors, provenance,
  assumptions, and the review-only draft;
- invalid or unsupported selections stop with actionable evidence rather than a
  guessed mask;
- setup burden and any difference between notebook-loaded and MCP-side models are
  recorded; and
- a human can decide whether to continue, revise, or stop without treating the
  draft as executed work.

## Focused Validation Evidence

Prior bounded worker evidence reports:

```text
./.venv/bin/python -m pytest tests/test_agent_inspect_model.py tests/test_agent_capabilities.py -q
107 passed in 1.26s
```

This report records that focused result; it does not claim that the full proposed
journey or a new composition has been implemented or tested.

## Implementation Slice Evidence

Developer approval was recorded for the first implementation slice on 2026-08-06.
The bounded implementation is a package-local, read-only composition at
[`ws3/agent/workflows.py`](../ws3/agent/workflows.py), exposed as
`preview_model_selection`. It requires one explicitly selected, already-loaded
`ForestModel`, an explicit matching `model_name` and `model_path`, and a nonempty
mask expression. It calls the existing `inspect_model` capability for live
metadata, then routes the explicit mask through the existing `build_mask`
capability and its `ForestModel.unmask` validator. It returns typed verified model
facts, mask-validation facts, capability attempts/errors and provenance IDs,
advisory uncertainty, and a `review_only` draft with `executed=False`.

Focused implementation evidence:

```text
python -m pytest tests/test_agent_workflows.py -q
10 passed in 1.15s

python -m pytest tests/test_agent_workflows.py tests/test_agent_inspect_model.py tests/test_agent_capabilities.py -q
117 passed in 2.19s

python -m ruff check ws3/agent/workflows.py tests/test_agent_workflows.py
All checks passed
```

The focused tests cover successful live metadata and nonempty mask validation,
missing or ambiguous model context, explicit path requirements, empty and invalid
masks, unsupported or malformed inspection results, advisory path/credential
redaction, provenance/error preservation, and no calls into action, schedule,
simulation, optimization, or draft-execution paths. A nonempty resolution is
explicitly reported as insufficient proof of scientific forestry meaning, and
human review remains required. No existing capability module, registry, MCP
deployment, provider configuration, model file, or model state was changed.

This is a partial implementation slice, not Task 8.7 closeout. Remaining work is
to review the public composition contract and evidence with the maintainer,
decide whether the artifact should be documented or exposed through an approved
client surface, and complete any additional acceptance evidence before closing
the child issue.

## Field-Test Implementation Slice

On 2026-08-06, the approved first implementation slice was revised to a concrete
scenario-report journey around the bundled
``examples/data/woodstock_model_files_tsa24_clipped`` model and its
``tsa24_clipped.seq`` schedule. The callable entry point is
``ws3.agent.report_scenario_inventory_products`` and the registry/MCP tool uses
the explicit name ``report_scenario_inventory_products``. Its request contains
only the selected model directory, model base name, and optional sibling schedule
path.

The workflow loads a fresh ``ForestModel``, imports the landscape, areas, yields,
actions, transitions, and schedule sections, then reports initial area with
``ForestModel.inventory(0)``. After applying the bounded source schedule only to
that fresh in-memory model, each row is computed with
``ForestModel.compile_product(period, '1.', acode='harvest')``,
``ForestModel.compile_product(period, 'totvol', acode='harvest')``, and
``ForestModel.inventory(period, 'totvol')``. The result includes model identity,
schedule provenance, rows, warnings/errors, source-file hash status, and the
explicit statement that no source model file was mutated. The importer now also
accepts the fixture's valid schedule form where the optional event-type token is
absent.

The direct Python workflow is deterministic and offline; the registered MCP
adapter ignores the shared provider and performs the computation host-side. The
schedule may mutate the isolated in-memory model, but it never receives a caller
model object and it does not write model outputs or source files. No mask,
provider-generated actions, optimization, plotting, arbitrary action input,
cross-package integration, or destructive/expensive operation is included.

Evidence:

- [Runnable offline field test](../examples/agent_scenario_report.py)
- [Scenario report tests](../tests/test_agent_scenario_report.py)
- [Agent/MCP regression tests](../tests/test_agent_mcp.py)
- `7 passed` for the focused scenario report tests
- `148 passed` for the existing agent, inspection, workflow, and MCP tests

This records the bounded implementation slice; broader MCP deployment/expansion,
cross-package work, and destructive or expensive operations remain gated by a
separate approval.

## Field-Test Acceptance Evidence (2026-08-07)

The exact public Python callable ``ws3.agent.report_scenario_inventory_products``
was exercised directly with the bundled ``tsa24_clipped`` model and its sibling
``tsa24_clipped.seq`` schedule; the runnable example was not used as the test
entry point. The call returned ``ok=True`` with model identity, initial area
``1366.737737577``, 24 ``harvest`` schedule entries across periods 1--10, and
10 report rows. Harvested area totalled ``1000.0`` and harvested volume
``164838.080298028``; standing volume was positive in every row.

The focused command ``python -m pytest tests/test_agent_scenario_report.py -q``
returned ``7 passed``. Its API-tracing test and the implementation source verify
the real ``ForestModel.inventory`` and ``ForestModel.compile_product`` calls.
The request has no age or mask input and performs no age stratification. A safe
missing-model-directory input returned ``ok=False`` with a structured
``ValueError`` result. The fixture contained 14 files and 14440 bytes before and
after the call; file sizes and SHA-256 hashes were equal, and the result reported
source-file integrity true. The registered descriptor remained a host-side
offline capability. A disposable stdio JSON-RPC probe also passed `initialize`,
`notifications/initialized`, `tools/list` (7 tools), and `tools/call` for
`report_scenario_inventory_products`; this verifies disposable transport only,
not Copilot host attachment or deployment approval.

This evidence supports the bounded scenario-report field test only. Task 8.7 is
complete within this approved slice; deployment, transport, broader client
exposure, and any mutating, expensive, or age-based expansion remain subject to
the existing approval boundary.

## Unresolved Decisions

- Should the first slice use a notebook-loaded `ForestModel`, MCP-side loading, or
  document both as separate adapters?
- Should the acceptance guarantee require only a nonempty resolution, or also
  expose matched development-type keys?
- What summary and provenance artifact format is durable and reviewable?
- Should `ws3_hint` be excluded from the first slice because its oracle is only
  citation-complete, or used only as a clearly labelled advisory fallback?
- What path-redaction policy protects sensitive local layouts without hiding the
  explicit model identity needed for reproducibility?

## Next Bounded Action And Approval Gate

The next bounded action is maintainer review of the first composition slice and
its focused evidence. Any client exposure, MCP deployment, package integration,
mutating operation, or expensive or long-running workflow still requires separate
explicit approval.

## Final Scenario-Report Acceptance Evidence (2026-08-07)

This final evidence record supersedes the earlier transport-availability note
for the current field test. The approved journey is a forestry analyst or
developer selecting the bundled ``tsa24_clipped`` model and its sibling
``tsa24_clipped.seq`` schedule to review a deterministic inventory/products
scenario. Inputs are the selected model directory, model base name, and
optional schedule path. Review points are model and schedule identity, initial
area, schedule-entry provenance, per-period harvested area, harvested volume,
standing volume, warnings/errors, and source-file integrity. The journey stops
on structured input/import/validation failure, missing or unsupported artifacts,
any failed integrity check, or before scientific interpretation or downstream
action. Success stops after the reviewable report: no user-supplied action,
optimization, plotting, mutation, or expensive or long-running operation
follows from this slice.

### Authoritative Contract Map

| Contract or behavior | Authoritative source | Evidence boundary |
| --- | --- | --- |
| Fresh model loading and model identity | [`ws3/agent/capabilities/scenario_report.py`](../ws3/agent/capabilities/scenario_report.py), [`ws3/forest.py`](../ws3/forest.py) | Fresh `ForestModel` from the explicit model directory/base name; no caller model is accepted or mutated. |
| Sibling `.seq` schedule import | [`ws3/agent/capabilities/scenario_report.py`](../ws3/agent/capabilities/scenario_report.py), [`examples/data/woodstock_model_files_tsa24_clipped/tsa24_clipped.seq`](../examples/data/woodstock_model_files_tsa24_clipped/tsa24_clipped.seq) | The default schedule is the selected model's sibling schedule and applies only to fresh in-memory state. |
| Inventory values | [`ws3/forest.py`](../ws3/forest.py) | Initial and standing values come from `ForestModel.inventory`. |
| Product values | [`ws3/forest.py`](../ws3/forest.py) | Harvested area and volume come from `ForestModel.compile_product` with verified product codes and `acode='harvest'`. |
| Serialization and registry/MCP exposure | [`ws3/agent/capabilities/__init__.py`](../ws3/agent/capabilities/__init__.py), [`ws3/agent/capabilities/scenario_report.py`](../ws3/agent/capabilities/scenario_report.py), [`tests/test_agent_mcp.py`](../tests/test_agent_mcp.py) | Existing descriptor/adapter serialization; no new server or transport is claimed. |
| Source integrity checks | [`ws3/agent/capabilities/scenario_report.py`](../ws3/agent/capabilities/scenario_report.py), [`tests/test_agent_scenario_report.py`](../tests/test_agent_scenario_report.py) | Before/after file sizes and SHA-256 hashes are compared and reported. |

### Claims Status

| Claim | Status | Evidence or limitation |
| --- | --- | --- |
| Direct callable and registered descriptor return the deterministic report. | Verified | Direct and registry calls succeeded; focused scenario tests returned `7 passed`. |
| The report uses live `ForestModel.inventory` and `ForestModel.compile_product` values. | Verified | API-tracing tests and source inspection cover both calls; field result: initial area `1366.737737577`, 10 rows, harvested area `1000.0`, harvested volume `164838.080298028`. |
| Invalid paths produce structured failure and source fixtures remain unchanged. | Verified | Missing-directory failure returned structured `ValueError`; 14 files, sizes, and SHA-256 hashes were unchanged. |
| Disposable stdio MCP initialize/tools/list/call succeeds. | Verified | A disposable stdio JSON-RPC probe passed `initialize`, `notifications/initialized`, `tools/list` (7 tools), and `tools/call` for `report_scenario_inventory_products`; this does not establish Copilot host attachment or approve a new server, deployment, or transport. |
| Age-based masking or stratification belongs in this journey. | Unsupported / outside approved slice | No age or mask input is exposed. Age is a stand attribute or state variable, not a stratification variable by itself; stratification requires an explicit classification rule and validated model fields. The earlier age/mask journey is outside this approved slice. |
| The report establishes universal package-version semantics or scientific forestry interpretation. | Unresolved / not claimed | It records package/model/schedule identity plus warnings/errors, but no universal package-version field or semantic certainty. This is accepted for this bounded slice; a future gate must define version fields and obtain maintainer/domain review of semantic meaning. |

### Setup, Risks, And Recommendation

Setup burden is bounded but nonzero: use the repository environment, select the
bundled model directory/base name, resolve the sibling schedule, and invoke the
existing Python registry/MCP descriptor. No credentials or provider selection is
needed. Observed risks are schedule/model drift, ambiguity in forestry meaning,
the distinction between host-side registry execution and deployment, and the
possibility that numeric success is mistaken for scientific validation. The
recommendation is **continue** only to maintainer review and closeout discussion
for this exact slice; **revise** if stronger version or semantic evidence is
required; **stop** before any expansion lacking separate approval. Broader
exclusions remain provider-generated actions, arbitrary actions, age
masks/stratification, optimization, plotting, mutation,
destructive/expensive/long-running work, cross-package work, new
server/transport, and credential selection.