# Phase 8 — AAM Tools and Skills

**Date**: 2026-08-06

**Status**: Task 8.7 bounded implementation complete; broader expansion remains gated

**Approval record (2026-08-06)**

- **Authority**: Developer.
- **Approved scope**: Developer approval dated 2026-08-06 covers the bounded deterministic
  Task 8.7 scenario-report implementation/field-test slice through the existing package-local
  registry/MCP descriptor surface:
  - callable `ws3.agent.report_scenario_inventory_products`;
  - existing registry/MCP descriptor exposure under `report_scenario_inventory_products`;
  - fresh in-memory model loading from an explicit model directory/base name and sibling
    `.seq` schedule;
  - live `ForestModel.inventory(...)` and `ForestModel.compile_product(...)` reporting;
  - structured provenance, warnings, errors, and source-file integrity status; and
  - the offline runnable example and focused tests.
  This approval does not authorize a new MCP server or transport, and expansion beyond this
  exact slice remains gated.
- **Candidate conceptual actions**: inspect, validate, summarize, and generate an
  inspectable hint/snippet draft, where supported. Actual APIs and package capabilities
  must be verified before they are claimed.
- **Explicit exclusions**: deploying a new MCP server or changing transport; broad
  cross-package integrations; changing package or model APIs or formats; provider-generated
  schedules/actions or arbitrary action inputs; age-based masking/stratification UX; plotting
  or optimization; mutating, destructive, irreversible, expensive, or long-running
  operations; credential, provider, vendor, client, or deployment selection; publishing or
  distributing a skills library; and a universal ontology or schema.
- **Next approval gate**: Any expansion beyond this exact bounded deterministic
  scenario-report implementation/field-test slice requires separate Developer approval,
  including new MCP server or transport work, broader client exposure, package integrations,
  provider-generated schedules/actions or arbitrary action inputs, age-based
  masking/stratification UX, plotting, optimization, mutation, or expensive/long-running
  operations. The approval must name the scope, owner, evidence, safety limits, and
  exclusions.

**Raw source**: [phase8-ideas.md](../tmp/phase8-ideas.md) (local/ignored working note)

## Current implementation slice

The first field-testable Task 8.7 slice is the host-side
``report_scenario_inventory_products`` workflow. Its callable Python entry point
is ``ws3.agent.report_scenario_inventory_products`` and its discoverable MCP
capability uses the same name. It accepts an explicit model directory and model
base name, defaults to that model's sibling ``.seq`` schedule, and returns model
identity, schedule provenance, initial inventory, per-period products and
standing volume, warnings, errors, and source-file integrity status.

The workflow imports a fresh in-memory ``ForestModel`` and computes values with
``inventory(0)``, ``compile_product(period, '1.', acode='harvest')``,
``compile_product(period, 'totvol', acode='harvest')``, and
``inventory(period, 'totvol')``. Applying the schedule changes only that fresh
in-memory state. Source model files are hashed before and after the run and the
result states that they were not mutated. No selection mask or provider-generated
action schedule is accepted, and the direct Python entry point runs offline
without credentials.

Evidence is linked from [the runnable example](../examples/agent_scenario_report.py)
and [the focused tests](../tests/test_agent_scenario_report.py). Future MCP
server expansion, cross-package work, optimization, plotting, arbitrary action
inputs, and destructive or expensive operations remain separately gated.

## Field-Test Result (2026-08-07)

The exact public Python callable ``ws3.agent.report_scenario_inventory_products``
was run directly against the bundled ``tsa24_clipped`` model and sibling
``tsa24_clipped.seq`` schedule. It returned ``ok=True`` with initial area
``1366.737737577``, 24 ``harvest`` entries across periods 1--10, 10 rows,
harvested area ``1000.0``, harvested volume ``164838.080298028``, and positive
standing volume in every row. The focused test command returned ``7 passed``;
its call-tracing test and the implementation source verify the real
``ForestModel.inventory`` and ``ForestModel.compile_product`` APIs.

A safe missing-model-directory input returned ``ok=False`` with a structured
``ValueError``. The 14 fixture files (14440 bytes) had identical sizes and
SHA-256 hashes before and after, and the report's source-file integrity flag was
true. No age input, age mask, or age stratification is exposed or used. This is
an offline host-side registry/callable result; a disposable stdio JSON-RPC probe
also passed ``initialize``, ``notifications/initialized``, ``tools/list`` (7
tools), and ``tools/call`` for ``report_scenario_inventory_products``. That
verifies disposable transport only, not Copilot host attachment or deployment
approval. The child issue is closed; parent issue #105 remains active/open
pending a reviewable PR and merge.

---

> **Interpretation boundary**
>
> This document expands raw Developer ideas into options that can be reviewed.
> It is not an approved plan, roadmap commitment, implementation specification,
> or claim that any named package already provides the capabilities discussed below.

## Agent-Assisted Modelling

Agent-Assisted Modelling (AAM) is an approach in which a user works with an agent to
understand, construct, inspect, run, and communicate about scientific models. The agent may
coordinate software, documentation, data, and validation, but modelling packages and their
explicit contracts remain the source of truth.

The desired outcome is to lower the practical barrier to complex UBC-FRESH workflows. A
researcher or practitioner should be able to state an intent, inspect its interpretation,
and use supported capabilities without memorizing every command, object, and format. The
user must still see what will happen, verify results, and judge scientific meaning.

## Objectives

- Explore a coherent AAM interface across candidate UBC-FRESH packages and make supported
  capabilities discoverable without relying on API recall.
- Separate executable package operations from reusable workflow guidance.
- Move from modelling intent to inspectable steps while preserving package-owned semantics and validation.
- Support provenance, reproducibility, useful failures, and a narrow pre-investment test.

## Non-Goals

- Approving an architecture, server, integration, implementation phase, or common API.
- Replacing scientific judgment, model review, package documentation, or explicit contracts.
- Granting arbitrary execution or hiding conversion, mutation, cost, or failure details.
- Treating plausible snippets as correct, standardizing uninspected internals, or defining roadmap commitments.

## MCP Tools and Skills

MCP tools and skills address different layers of an AAM experience.

An **MCP tool** is a structured, discoverable, executable capability. It declares inputs and
outputs, invokes a bounded operation, and returns structured results or errors grounded in
the package or an owning adapter rather than in generated API guesses.

A **skill** is workflow and procedural knowledge supplied to an agent. It explains when and
how to use tools, what evidence to inspect, and when to stop or request human judgment. It
may compose several tools, but is not proof that an operation succeeded.

| Dimension | MCP tool | Skill |
| --- | --- | --- |
| Primary role | Execute a bounded capability | Guide a workflow or decision process |
| Contract | Structured inputs, outputs, errors | Procedural steps, constraints, evidence |
| Grounding | Package state, adapter, or validator | Documentation and accepted practice |
| Typical scope | One inspect, validate, translate, run, or summarize operation | A multi-step modelling outcome |
| Failure mode | Explicit tool or validation error | Incomplete, inapplicable, or stale guidance |
| User control | Review parameters and returned effects | Review the proposed sequence and choices |
| Versioning concern | Tool schema and package capability | Workflow assumptions and referenced tools |

The surfaces compose when a skill turns user intent into transparent MCP calls and checks
each result. Tools provide executable facts; skills provide workflow logic. Neither should
silently fill gaps in the other.

## Candidate MCP Server Shape

One option is a **UBC-FRESH AAM server** with discoverable package-oriented toolboxes. A
client could discover packages and versions, then inspect a toolbox. This is not a topology
or ownership decision; discovery must report real constraints and preserve package boundaries.

### Architecture alternatives

| Alternative | Potential benefit | Risk or tradeoff | Question to test |
| --- | --- | --- | --- |
| One monolithic server | One endpoint and unified discovery | Coupled releases, dependencies, and permissions | Can package isolation remain clear? |
| One server per package | Strong ownership and independent versioning | More client configuration and discovery work | Is setup manageable for new users? |
| Thin hub over package servers | Unified catalog with package-owned execution | Routing, compatibility, and failure complexity | Does the hub add enough value? |
| One server with package toolboxes | Coherent user surface and shared policy | Shared host may blur ownership or fault isolation | Can schemas expose boundaries honestly? |

Other hybrids may be preferable. Evidence about workflows, installation, trust boundaries,
and maintenance should decide; no option is locked in here.

## Candidate Toolbox Domains

These candidate surfaces do not assert MCP support, readiness, installability, API stability,
or package capabilities.

| Candidate toolbox | Questions for discovery | Candidate verbs, where supported |
| --- | --- | --- |
| WS3 | Which model structures and workflows can be exposed with strong validation? | inspect, validate, translate, run, summarize |
| FHOPS | Which package-owned operations are suitable for bounded agent use? | inspect, validate, translate, run, summarize |
| FreshForge | Which orchestration or workflow surfaces are public and stable? | inspect, validate, translate, run, summarize |
| ModelWright | What artifacts and contracts exist, and which are discoverable? | inspect, validate, translate, run, summarize |
| FABLE Pyculator | What inputs, outputs, and validation boundaries are supported? | inspect, validate, translate, run, summarize |
| Nemora | Which operations can be safely represented as structured tools? | inspect, validate, translate, run, summarize |
| SpaDES-WS3 Bridge | Which interoperability boundaries are explicit and testable? | inspect, validate, translate, run, summarize |

These candidate verbs are not proposed API names. A package might support a subset, use
different language, or expose more specific operations. Translation should state what is
preserved, changed, omitted, or unresolved rather than assert unevidenced equivalence.

Potential cross-package concerns include capability and version discovery; metadata
inspection; package-owned validation; translation with loss reporting; bounded execution
with resource expectations; and summaries linked to structured provenance.

## Candidate Skills Library

### Onboarding and get-started
Move from an installed environment and stated goal through capability discovery, a minimal
example, and a verified result. Test availability instead of recalling commands.

### Complex model-building workflows
Split an objective into inspectable stages, identify domain decisions, call available tools,
and preserve artifacts. Include evidence requirements and stopping conditions.

### Debugging and validation
Collect minimal context, use package-owned checks, distinguish tool from model failures, and
return an evidenced diagnosis. Do not route around a failed validator.

### Translation and interoperability
Inspect endpoint capabilities, preview mappings, report unsupported constructs, and validate
artifacts. Keep loss and ambiguity visible rather than silently choosing a meaning.

### Reproducibility
Record versions, inputs, parameters, environment facts, provenance, outputs, and validation
results needed to repeat or audit a workflow. The record format remains undecided.

### Snippet and hint generation
Produce focused CLI or API drafts when no direct tool covers the need. Snippets are
inspectable drafts, not answers to execute blindly. Cite their basis, mark assumptions,
exclude secrets, and invite controlled validation.

## Safety and Trust Boundaries

- Prefer dry-run or preview where supported; require explicit paths, parameters, package selection, and destinations.
- Expose filesystem, model-state, environment, and remote mutation.
- Link provenance to inputs, versions, and tools; keep package and scientific checks in the evidence path.
- Distinguish invalid input, unsupported capability, validation, dependency, permission, and runtime failures.
- Confirm destructive, irreversible, expensive, or long-running work under policies still to be defined.
- Keep secrets host-managed and out of prompts, snippets, provenance, and model artifacts.
- Discover tool, schema, server, adapter, and package versions; discovery reports capability, not permission.
- Preserve partial results and uncertainty without relabeling them as success.

## Staged Exploration Path

These stages are questions to investigate, not approved tasks or a delivery plan.

### Discovery
**Entrance questions**: Which journeys are difficult? Which package contracts, validators,
and docs are authoritative? What environments, clients, and risks matter?

**Exit questions**: Is there a high-value bounded journey with credible validation? Are
ownership, permissions, and evidence clear enough for a non-generalized prototype?

### Narrow WS3 vertical slice
**Entrance questions**: Can the scenario use a package-grounded WS3 capability, remain
read-only or preview-first, and be measured without trusting generated prose?

**Exit questions**: Were guesses reduced; calls, assumptions, and results inspectable;
failures actionable; and setup burden proportionate to benefit?

### Evaluate
**Entrance questions**: Is there representative user and artifact evidence? Can tool value
be separated from skill quality and agent behavior? Are safety records reviewable?

**Exit questions**: Does evidence support continuing, revising, or stopping? Which
assumptions survived, and what maintenance or governance costs appeared?

### Expand
**Entrance questions**: Has an owner approved expansion? Has each package been assessed for
contracts, validators, risks, and demand? Are compatibility policies adequate?

**Exit questions**: Does each integration have acceptance evidence, preserve package
boundaries, explain semantic loss, and remain supportable?

## Candidate Vertical-Slice Scenario

A candidate scenario helps a user understand an existing WS3 model and draft a safe next
step. The names below describe concepts, not fabricated tool APIs.

1. An onboarding skill asks for the modelling goal and explicit model-artifact path.
2. The skill requests server and WS3 capability discovery for the active environment.
3. With reviewed parameters, it invokes an available read-only inspection capability and
  receives structured metadata or a structured failure.
4. If discovery shows an appropriate validation capability, the user authorizes its call.
5. The skill separates verified observations from interpretation and links tool evidence.
6. Where no tool exists, it drafts but does not execute an assumption-labelled hint.
7. The user revises inputs, investigates, reviews the draft, or stops. Mutation or expense
  requires a new decision.

The skill manages intent, sequence, and evidence; MCP tools supply bounded facts. This does
not require a universal agent or imply the sequence works for another package.

## Decision Questions

### Product
- Who is the first user, and which modelling barrier matters most?
- Should the experience optimize for learning, throughput, or reliability, and which outcomes
  require direct review?
- How should uncertainty appear, and is a coding-agent client the first surface?

### Architecture
- Which server topology best preserves package ownership and simple discovery?
- Where should schemas, adapters, validators, and shared policies live, and how are optional
  packages isolated?
- What compatibility contract spans client, server, tool, and package versions, including
  local, remote, and long-running execution?

### Skill design
- What makes a skill authoritative enough to publish and maintain?
- How should skills bind to discovered versions, divide deterministic steps from judgment,
  and avoid brittleness?
- How do snippets prove APIs, disclose assumptions, and test realistic variants and failures?

### Governance and operations
- Who owns each toolbox, skill, schema, and compatibility decision?
- What review governs mutation, expense, external connections, provenance, privacy, telemetry,
  and redaction?
- How are capabilities deprecated, incidents handled, stale skills withdrawn, and support
  shared across repositories?

## Evaluation Criteria and Signals

- **Grounding**: claims and operations trace to discovered tools, package state, validators,
  or selected documentation.
- **Task success**: representative users complete and explain the journey; domain correctness
  is reviewed separately from agent fluency.
- **Failure quality**: unsupported or invalid operations fail explicitly with useful evidence.
- **Safety**: no hidden mutation, secret exposure, or unconfirmed destructive or expensive act.
- **Reproducibility**: evidence identifies inputs, versions, parameters, outputs, and checks.
- **Usability**: effort falls for the target user without concealing important decisions.
- **Maintainability**: owners can update and test tools and skills as packages evolve.
- **Interoperability**: translation reports preserved, changed, lost, and unresolved semantics.

Metrics and thresholds should be selected during evaluation design, not invented
here. Both successful and failed attempts should inform the decision.

## Approval Gates

No broader implementation should be inferred. Explicit Developer approval is needed before each transition beyond the approved slice:

1. Treating this exploration as roadmap scope or creating implementation issues.
2. Selecting a server topology or assigning cross-repository ownership.
3. Building and exposing a WS3 vertical slice beyond the approved deterministic scenario-report slice.
4. Publishing or distributing a skill, tool schema, adapter, or server.
5. Adding another package integration or enabling mutating or expensive operations.

An approval should identify scope, owner, evidence, safety limits, and exclusions. Passing one
gate does not imply approval of later gates.

## Dependencies and Unknowns

- Public contracts, versions, installation modes, and maintainers need package-by-package verification.
- Representative users, journeys, and artifacts are not selected.
- Target MCP clients, protocol support, authentication, and deployment are not established.
- Compatibility, discovery, deprecation, compute, cancellation, concurrency, and long-running jobs need study.
- Scientific validation oracles may not exist for every desirable operation.
- Data licensing, privacy, provenance, remote-service policy, ownership, and releases remain undecided.
- Existing package-specific agent surfaces must be verified, not inferred here.
- Authoritative MCP- and Copilot-specific documentation still needs to be selected and verified.

## Out of Scope

- Implementing or configuring new servers, clients, dependencies, adapters, tools, skills,
  schemas, or endpoints beyond the approved deterministic scenario-report implementation
  and existing registry/MCP descriptor surface.
- Changing package APIs, model formats, validation rules, or release processes.
- Selecting vendors, services, model providers, credentials, or a universal ontology.
- Benchmarking unselected agents, asserting unprotocolled targets, or promising any named
  package integration.

## Task 8.7 Current Discovery Result

The durable discovery record is [planning/phase8_aam_task_8_7_discovery.md](phase8_aam_task_8_7_discovery.md).
It selects a read-only journey that inspects an explicitly selected WS3 model,
validates a natural-language stand-selection mask against live state, summarizes
verified results and provenance, and produces a review-only next-step draft without
executing model changes. The verified first-slice surfaces are `inspect_model` and
`build_mask`; `ws3_hint` remains advisory and is not required for the first slice.

This is active discovery evidence for the exploratory journey; the approved implementation
slice is recorded above. No new MCP deployment, cross-package integration, mutating
operation, or expensive or long-running operation is approved. The approval boundary and
exploratory alternatives above remain in force.

## References

- [OpenAI tools guide](https://developers.openai.com/api/docs/guides/tools)
- [OpenAI Code Interpreter tools guide](https://developers.openai.com/api/docs/guides/tools-code-interpreter)
- [Raw Phase 8 Developer ideas](../tmp/phase8-ideas.md) (local/ignored; not a durable tracked source)

## Final Task 8.7 Evidence Repair (2026-08-07)

The approved user journey is a forestry analyst or developer selecting the
bundled `tsa24_clipped` model and sibling `tsa24_clipped.seq` schedule,
reviewing a deterministic inventory/products report, and stopping before
scientific interpretation or downstream action. Inputs are the explicit model
directory, model base name, and optional sibling schedule path. Review points
are model/schedule identity, initial area, schedule provenance, per-period
products and standing volume, warnings/errors, and source integrity. Structured
failure, unsupported artifacts, or a failed integrity check are stop conditions;
success stops after the reviewable report.

### Contract Sources And Claim Status

The authoritative loading and schedule-import implementation is
[`ws3/agent/capabilities/scenario_report.py`](../ws3/agent/capabilities/scenario_report.py),
with model behavior owned by [`ws3/forest.py`](../ws3/forest.py). Inventory and
product facts therefore come from `ForestModel.inventory` and
`ForestModel.compile_product` respectively. Structured serialization and
registry exposure are governed by
[`ws3/agent/capabilities/__init__.py`](../ws3/agent/capabilities/__init__.py)
and existing MCP regression coverage in
[`tests/test_agent_mcp.py`](../tests/test_agent_mcp.py). Source integrity is
implemented and tested in
[`tests/test_agent_scenario_report.py`](../tests/test_agent_scenario_report.py)
using before/after file sizes and SHA-256 hashes. These sources establish the
current bounded contract; they do not establish a new server or transport.

| Claim | Status |
| --- | --- |
| Direct callable and registered descriptor succeed for the bundled scenario; focused scenario tests return `7 passed`. | Verified |
| Live inventory/product APIs produce initial area `1366.737737577`, 10 rows, harvested area `1000.0`, and harvested volume `164838.080298028`. | Verified |
| Missing model input returns structured `ValueError` failure and the 14 fixture files remain byte/hash identical. | Verified |
| Disposable stdio MCP initialize/tools/list/call succeeds. | Verified by a disposable stdio JSON-RPC probe: `initialize`, `notifications/initialized`, `tools/list` (7 tools), and `tools/call` for `report_scenario_inventory_products` passed; this is not Copilot host attachment or approval for new deployment or transport. |
| Age masks or age stratification belong in this slice. | Unsupported and outside the approved slice. Age is a stand attribute/state variable, not a stratification variable by itself; stratification requires an explicit classification rule and validated model fields. |
| A universal package-version field or semantic certainty about forestry interpretation is provided. | Unresolved and not claimed. The report records package/model/schedule identity and warnings/errors, but not a universal package-version field or semantic interpretation. |

The version/uncertainty limitation is accepted for this bounded field-test slice.
A future approval gate must define the version contract and require maintainer
and domain review of forestry semantics before broader client exposure.

### Setup, Risks, And Recommendation

Setup requires the repository environment, the bundled model selection, sibling
schedule resolution, and the existing Python registry/MCP descriptor; it needs
no credentials or provider selection. Observed risks are model/schedule drift,
semantic over-interpretation of numeric outputs, and confusion between this
host-side descriptor evidence and a deployed MCP service. **Continue** to
maintainer review and closeout discussion for the exact approved slice;
**revise** if version or semantic evidence is required; **stop** before any
broader exposure or excluded operation. Exclusions remain provider-generated or
arbitrary actions, age masks/stratification, optimization, plotting, mutation,
destructive/expensive/long-running work, cross-package work, new
server/transport, and credential selection.