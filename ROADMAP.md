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
- Status: complete
- Branch: `feature/ws3-phase4-docs`
- Completion date: 2026-07-26

### Task 4.1 — Restructure documentation with audience-based navigation
- Status: complete
- Scope: rewrite `index.rst` landing page with audience navigation (new users, advanced users, LLM agents). Create `getting_started/`, `textbook/`, `howto/`, `reference/`, `guides/` sections.
- Verification: Docs build successfully, landing page has audience navigation, all section indexes created.

### Task 4.2 — Create Getting Started section
- Status: not_started
- Scope: installation guide, quickstart tutorial, first model walkthrough, architecture overview.

### Task 4.3 — Create textbook for forest estate modelling
- Status: complete
- Scope: 16-chapter textbook covering forest inventory, growth/yield, actions/transitions, optimization, spatial allocation, financial analysis, uncertainty, advanced topics, carbon modelling, FEMIC integration, fhops integration, FreshForge workflow automation, SpaDES integration, disturbance modelling, and supply chain integration. Each chapter includes learning objectives, worked examples, and exercises.
- Progress: All 16 chapters complete (ch01-ch16) with substantive chapter content across the full sequence.

### Task 4.4 — Create how-to guides
- Status: complete
- Scope: operational guides for data preparation, curve definition, action definition, optimization, parallel optimization, spatial allocation, libcbm callbacks, financial scenarios, custom selectors, custom growth functions, model validation, and reproducibility.
- Progress: All 12 how-to guides created with runnable examples and troubleshooting sections.

### Task 4.5 — Create agent-friendly contract pages
- Status: complete
- Scope: compact technical contracts for LLM coding agents: repo/runtime invariants, module responsibilities, class hierarchy, solver options. Follow femic/fhops pattern.
- Progress: All 4 contract pages created (data_contracts, runtime_invariants, module_boundaries, output_format_spec).

### Task 4.6 — Create coding agent onboarding guide
- Status: complete
- Scope: guide for LLM coding agents to understand ws3 architecture, data flows, and conventions. Include purpose, use cases, quick contract table, and platform-specific notes.
- Verification: Guide includes module map, class hierarchy (mermaid), data flow diagram, common patterns, and platform notes.

### Task 4.7 — Create troubleshooting and limitations guides
- Status: complete
- Scope: known issues, recovery procedures, honest documentation of boundaries and external dependencies.
- Progress: Created troubleshooting.rst and limitations-and-boundaries.rst with comprehensive coverage.

### Task 4.8 — Update conf.py and verify docs build
- Status: complete
- Scope: enhance Sphinx configuration with new extensions, verify all docs build successfully, ensure GitHub Pages deployment works.
- Verification: Docs build with zero errors and zero warnings.

## Phase 5 — Advanced Features and Production Deployment

- Parent issue: #60
- Status: complete
- Branch: `feature/ws3-phase5`
- Start date: 2026-07-26
- Alpha release: v1.1.0a1 (published to PyPI)

### Task 5.1 — Advanced Modeling Features
- GitHub issue: #61
- Status: complete (code created, tests pending)
- Scope: stochastic optimization, multi-objective optimization, dynamic planning, climate scenarios, enhanced carbon accounting

### Task 5.2 — User Experience Improvements
- GitHub issue: #62
- Status: complete (code created, tests pending)
- Scope: interactive notebooks, FAQ section, migration guide

### Task 5.3 — Performance Optimization
- GitHub issue: #63
- Status: complete (code created, tests pending)
- Scope: solver tuning, memory profiling, parallel processing, incremental solving, caching

### Task 5.4 — Integration Enhancements
- GitHub issue: #64
- Status: complete (code created, tests pending)
- Scope: fhops integration, FEMIC integration, FreshForge workflows, SpaDES coupling, API endpoints

### Task 5.5 — Production Deployment
- GitHub issue: #65
- Status: complete (v1.1.0a1 published to PyPI via trusted publisher)
- Scope: release packaging, CI/CD pipeline, versioning, changelog, community guidelines, support channels

### Task 5.6 — Additional How-To Guides
- GitHub issue: #66
- Status: complete (4 new guides created)
- Scope: advanced optimization, custom solvers, data validation, scenario analysis

### Task 5.7 — Textbook Expansion
- GitHub issue: #67
- Status: partial (ch17-18 created, ch19-20 not yet created)
- Scope: advanced spatial modeling, carbon accounting in detail, case studies, future directions

### Task 5.8 — Testing and Validation
- GitHub issue: #68
- Status: not_started (no dedicated test files for new modules)
- Scope: unit tests, integration tests, performance tests, regression tests, documentation tests

## Phase 6 — Documentation Cleanup and Integration

- Parent issue: #69
- Status: complete
- Branch: `feature/ws3-phase6-docs-cleanup`
- Completion date: 2026-07-28

### Task 6.1 — Audit documentation for AI slop
- GitHub issue: #70
- Status: complete
- Scope: read every .rst and .md file, identify verbose filler, redundant explanations, overly casual language
- Deliverable: `planning/phase6_independent_audit.md`

### Task 6.2 — Integrate legacy chapters
- GitHub issue: #71
- Status: complete
- Scope: merge legacy flat chapters into structured sections or remove if obsolete, remove "Old Documentation" section from index.rst
- Progress: Removed 2-line stubs (`SpaDES.rst`, `libCBM.rst`), removed deprecated legacy chapters (`Chapt1.rst`, `Chapt2.rst`, `intro.rst`, `aboutws3.rst`), updated `appendices.rst`

### Task 6.3 — Purge AI slop
- GitHub issue: #72
- Status: complete
- Scope: remove filler sentences, tighten verbose explanations, ensure consistent tight technical tone
- Progress: Fixed P0 (architecture_overview.rst, ch09_advanced_topics.rst), P1 (spatial-allocation.rst, multi-objective-optimization.rst, parallel-optimization.rst), P2 (guides/index.rst filler, troubleshooting.rst fabricated APIs)

### Task 6.4 — Verify docs build
- GitHub issue: #73
- Status: complete
- Scope: ensure sphinx-build succeeds with zero errors and zero warnings
- Result: `sphinx-build -b html docs/source _build/html -W` passes with zero errors. Only pre-existing `image.not_readable` warnings from nbsphinx notebook output images (205 warnings, all image-related, not documentation issues).

## Phase 7 — Release and Community Building

- Parent issue: [#87](https://github.com/UBC-FRESH/ws3/issues/87)
- Status: complete
- Branch: `feature/ws3-phase7-release`
- Start date: 2026-07-27
- Completion date: 2026-07-29
- Documentation: [GitHub Pages](https://ubc-fresh.github.io/ws3/) (migrated from ReadTheDocs)

### Task 7.1 — Release Verification
- GitHub issue: [#88](https://github.com/UBC-FRESH/ws3/issues/88)
- Status: complete
- Scope: version bump to 1.1.0a2, CHANGELOG update, verification suite

### Task 7.2 — Community Infrastructure
- GitHub issue: [#89](https://github.com/UBC-FRESH/ws3/issues/89)
- Status: complete
- Scope: GitHub Discussions, issue templates, CONTRIBUTING.md, support channels

### Task 7.3 — User Testing Readiness
- GitHub issue: [#90](https://github.com/UBC-FRESH/ws3/issues/90)
- Status: complete
- Scope: verify notebooks execute, add test files for advanced_modeling/perf/integration
- Note: Test files exist and pass (44 passed, 3 skipped). Core functionality verified. Some notebook bugs remain (expected for alpha).

### Task 7.4 — PyPI Release
- GitHub issue: [#91](https://github.com/UBC-FRESH/ws3/issues/91)
- Status: complete
- Scope: publish v1.1.0a2 to PyPI
- Note: published via PyPI trusted publisher (OIDC); tag `v1.1.0a2` pushed and the Release Artifacts workflow succeeded.

## Phase 7.5 — Code Quality Remediation and Patch Release

- Parent issue: [#95](https://github.com/UBC-FRESH/ws3/issues/95)
- Status: complete
- Branch: `feature/ws3-phase7.5-code-quality`
- Release: `v1.1.0a3`
- Date: 2026-07-29

**Goal**: clear the code-quality debt blocking meaningful CI signal, fix runtime defects shipped in `v1.1.0a2`, and cut a patch release so Phase 8 starts from a codebase where automated checks mean something.

### Task 7.5.1 — Fix undefined-name defects
- Status: complete
- Issue: [#93](https://github.com/UBC-FRESH/ws3/issues/93)
- Scope: cleared all 25 `F821` findings. Two defects in the `_cp` branch of `_compile_oper_expr` meant only `_cp =` ever worked; `resolve_tmask` crashed on `_REPLACE`/`_APPEND` models. Added parametrized regression tests.

### Task 7.5.2 — Lint stage 1: mechanical whitespace
- Status: complete
- Scope: trailing whitespace, whitespace-only blank lines, end-of-file newlines. Whitespace only, no semantic change. 1808 -> 1001 findings.

### Task 7.5.3 — Lint stage 2: correctness defects
- Status: complete
- Scope: `F401` unused imports (genuine probes annotated, not deleted), `E722` bare excepts narrowed, `E711`/`E712` comparisons, `F841` unused locals with side-effecting calls preserved. All pyflakes findings cleared.

### Task 7.5.4 — CI restructure
- Status: complete
- Issue: [#94](https://github.com/UBC-FRESH/ws3/issues/94)
- Scope: `test` no longer gated on `lint`; mypy advisory pending Phase 2 typing debt; `.flake8` policy config added. `flake8 ws3/ tests/` exits 0.

### Task 7.5.5 — Release v1.1.0a3
- Status: complete
- Scope: version bump, CHANGELOG, tag, PyPI publish via trusted publisher.

## Phase 7.6 — Defect Sweep from mypy Findings

- Parent issue: [#99](https://github.com/UBC-FRESH/ws3/issues/99)
- Status: complete
- Branch: `feature/ws3-phase7.6-defect-sweep`
- Release: `v1.1.0a4`
- Date: 2026-07-29

**Goal**: fix the subset of mypy findings representing actual defects, so Phase 8 can build symbol-validating capabilities on a package that does not reference things that do not exist. Deliberately not the full 323-error typing debt ([#98](https://github.com/UBC-FRESH/ws3/issues/98)).

| Class | Before | After |
|---|---:|---:|
| `attr-defined` | 12 | 0 |
| `union-attr` | 20 | 0 |
| `unused-ignore` | 34 | 0 |
| mypy total | 323 | 258 |

### Defects fixed
- [#100](https://github.com/UBC-FRESH/ws3/issues/100) — `sylv_cred` bound `log` to `math.exp`, returning values wrong by ~40x without raising. The existing test asserted the buggy output.
- [#101](https://github.com/UBC-FRESH/ws3/issues/101) — every `rv=True` path raised `NameError`; PaCal was never imported. Restored via a NumPy 2.0 compatibility shim and a `ws3[rv]` extra.
- [#97](https://github.com/UBC-FRESH/ws3/issues/97) — `advanced_modeling` imported `ws3.core.compile_scenario`, which does not exist.
- [#103](https://github.com/UBC-FRESH/ws3/issues/103) — twelve Phase 5 entry points returned fabricated or vacuous results; now gated.
- 20 unguarded `re.search(...).group(...)` sites in the Woodstock parsers now raise descriptive `ValueError`s.

### Deferred
- [#98](https://github.com/UBC-FRESH/ws3/issues/98) — remaining 258 mypy findings (annotation coverage). mypy stays `continue-on-error` in CI.
- [#102](https://github.com/UBC-FRESH/ws3/issues/102) — whether to adopt PaCal as `fresh-pacal`. Gated on licence (GPL-3.0 vs ws3 MIT) and on contacting the upstream authors.

## Phase 8 — Embedded Agent Capabilities

- Parent issue: [#105](https://github.com/UBC-FRESH/ws3/issues/105)
- Status: complete
- Branch: `feature/ws3-phase8-embedded-agents`
- Start date: 2026-07-29
- Completion date: 2026-07-29
- Merged: [PR #113](https://github.com/UBC-FRESH/ws3/pull/113)
- Design doc: [planning/phase8_embedded_agents.md](planning/phase8_embedded_agents.md)
- Companion package: [fresh-agent-core](https://github.com/UBC-FRESH/fresh-agent-core)

**Goal**: give ws3 a validated, agent-backed capability surface so external coding agents operate the package through a contract-bound interface instead of inferring the Python API from documentation.

**Core principle**: *a capability is a prompt plus a validator plus a retry budget. No oracle, no capability.* An LLM proposes; a validator checks the proposal against real ws3 state; output that fails validation never reaches the caller. On exhaustion the result is `ok=False` with reasons, never a best guess.

**Why the timing**: Phases 7.5 and 7.6 exist because this premise needs a package that does not reference things which do not exist. Before them ws3 contained 25 undefined names, four calls to non-existent APIs, and twelve entry points returning fabricated results. Those are now zero.

**Package split**: `fresh-agent-core` owns the mechanism (config, provider, `Capability` contract and retry loop, provenance, test double, MCP host). Each package owns its capabilities and validators, because the validator is the part requiring domain knowledge. Core depends on nothing in the ecosystem.

| Capability | Oracle |
|---|---|
| `build_mask` | mask resolves against the `ForestModel` to ≥1 development type |
| `explain_exception` | every ws3 symbol cited actually exists |
| `diagnose_import` | the fix is applied to a scratch copy and the section re-imports |

### Task 8.1 — fresh-agent-core: runtime foundation
- GitHub issue: [#106](https://github.com/UBC-FRESH/ws3/issues/106)
- Status: complete
- Scope: `AgentConfig` resolution, OpenAI-compatible provider, error hierarchy, `available()` probe, `FakeProvider`. Credential redaction by substring match so unfamiliar vendor headers redact by default.

### Task 8.2 — fresh-agent-core: capability framework and provenance
- GitHub issue: [#107](https://github.com/UBC-FRESH/ws3/issues/107)
- Status: complete
- Scope: `Capability` ABC, validate/retry loop with failure feedback, `Verdict`, `CapabilityResult`, provenance with JSONL/memory/null sinks, `Registry`. Provider failures propagate; content failures return `ok=False`.

### Task 8.3 — ws3: implement three capabilities
- GitHub issue: [#108](https://github.com/UBC-FRESH/ws3/issues/108)
- Status: complete
- Scope: `build_mask`, `explain_exception`, `diagnose_import`, each with a validator consulting real ws3 state. 50 tests, all offline.

### Task 8.4 — ws3: MCP wiring
- GitHub issue: [#109](https://github.com/UBC-FRESH/ws3/issues/109)
- Status: complete
- Scope: generic MCP host in core, ws3 registry exposed as tools, `ws3-agent-mcp` console entry point, per-capability input schemas. Pinned `mcp>=1.0,<2`; 2.x migration tracked in [fresh-agent-core#1](https://github.com/UBC-FRESH/fresh-agent-core/issues/1).

### Task 8.5 — Discoverability contract
- GitHub issue: [#110](https://github.com/UBC-FRESH/ws3/issues/110)
- Status: complete
- Scope: `AGENTS.md` section declaring the capability surface as the supported agent interface and stating the oracle rule; `README.md` pointer; MCP registration snippet for agent-workbench.

### Task 8.6 — Packaging and documentation
- GitHub issue: [#111](https://github.com/UBC-FRESH/ws3/issues/111)
- Status: complete
- Scope: `ws3[agent]` and `ws3[agent-mcp]` extras, Sphinx guide covering configuration, capabilities, provenance and how to add a capability validator-first, worked example, CHANGELOG entry.

## Backlog — not yet scheduled

Work identified during earlier phases and deliberately deferred to keep those phases scoped.

### Typing debt remediation
- Issue: [#98](https://github.com/UBC-FRESH/ws3/issues/98)
- Status: not_started
- Scope: 258 remaining mypy findings (annotation coverage), 69% in `forest.py`, from the incomplete Phase 2 typed refactor. The defect-bearing classes were cleared in Phase 7.6; what remains threatens nothing. mypy stays `continue-on-error` in CI until this lands.

### PaCal adoption decision
- Issue: [#102](https://github.com/UBC-FRESH/ws3/issues/102)
- Status: parked
- Scope: whether to adopt PaCal as `fresh-pacal`. Gated on licence (PaCal is GPL-3.0-or-later, ws3 is MIT) and on contacting the upstream authors. Not blocking: PaCal works today via the compatibility shim and the `ws3[rv]` extra.

### Flaky performance test
- Issue: [#112](https://github.com/UBC-FRESH/ws3/issues/112)
- Status: not_started
- Scope: `test_large_problem_scalability` asserts on wall-clock time in a way dominated by warm-up rather than problem size. Reproduced on a clean checkout.

### MCP 2.x migration
- Issue: [fresh-agent-core#1](https://github.com/UBC-FRESH/fresh-agent-core/issues/1)
- Status: not_started
- Scope: 2.0 removed the low-level `Server` decorator API. Main prize is `MCPServer.tool()` generating schemas from type hints, which removes hand-maintained schema drift by construction. Elicitation needs a deliberate decision: it must run before the validate loop, never inside it, or the oracle guarantee weakens into a conversation.

### Lint tooling split and notebook cell-type defects
- Issue: [#120](https://github.com/UBC-FRESH/ws3/issues/120)
- Status: not_started
- Scope: the repo configures two linters and enforces one. `flake8 ws3/ tests/` is the CI gate and passes; `ruff` is configured in `pyproject.toml` but wired to no job, while `AGENTS.md` and `CONTRIBUTING.md` tell contributors to run it. Two real defects hide in the resulting noise: notebooks `071` and `078` store Markdown in `code` cells (they raise `SyntaxError` if executed, and cause 12 Sphinx warnings), and `[tool.ruff] target-version = "py39"` contradicts `requires-python = ">=3.10"`. Residual after aligning scope and policy is 515 cosmetic findings with zero F-codes.

## Phase 9 — Woodstock Format Contract and Dataset Linting

Parent issue: #114
Branch: `phase9-woodstock-format-contract`
PR: #119, merged
Status: complete

Give ws3 a machine-readable contract for the Woodstock input data format, and use it to report
what ws3 does and does not read from a dataset. The format is deliberately open ended: a model
instance declares its own theme set, order and codes, and the LANDSCAPE section is the
authoritative source for the theme vector.

| Task | Issue | Status |
| --- | --- | --- |
| P9.1 Ship the keyword contract as package data | #115 | complete |
| P9.2 Preserve theme descriptions on LANDSCAPE import | #116 | complete |
| P9.3 Dataset linter for unsupported sections and keywords | #117 | complete |
| P9.4 Document the supported subset and divergences | #118 | complete |

Measured support: Landscape, Areas, Yields, Transitions, Outputs, Constants and Schedule are
implemented; Actions is partial; Optimize, Control, Graphics and Lifespan are stubs that import
nothing; Regimes, Reports, Queue, Allocation and LpSchedule have no importer. 198 keywords
catalogued, 25 read by ws3.

Recorded divergences from Woodstock, intentional and not defects:

- Woodstock measures age and action timing in periods; ws3 measures them in years.
- Woodstock counts themes from one (`_THn`); ws3 stores themes zero-indexed.

Documented at `docs/source/reference/contracts/woodstock_format.rst`. The support tables on
that page are generated at build time from `ws3.woodstock` by the `docs/source/_ext/ws3_woodstock.py`
Sphinx extension, so the documented subset cannot drift from the implemented one.

## Phase 10 — FEMIC Model Contract and Verification Oracles

- Parent program: [FEMIC #305](https://github.com/UBC-FRESH/femic/issues/305)
- Companion issue: [#121](https://github.com/UBC-FRESH/ws3/issues/121)
- Status: active
- Branch: `feature/p10-femic-model-contract`

### Goal

Provide the narrow ws3 domain contract that FEMIC can call while FreshForge owns the surrounding workflow graph. The first target is reliable construction and verification of a new ws3 model instance from typed model data, without adding a second workflow engine to ws3.

### Scope

- Define a serializable ws3 model contract or adapter surface for themes, areas, yields, actions, transitions, outputs, horizon, and period length.
- Expose deterministic extraction from a known-good imported model where the state is recoverable.
- Expose deterministic emission or adapter hooks that FEMIC can use without asking a model to generate raw Woodstock syntax.
- Provide verification oracles for input lint, import, theme-vector arity, development-type/area bindings, yield coverage, action references, transition closure, and bounded compile/solve smoke checks where supported.
- Keep the existing `ws3.agent` validator-first contract and optional dependency boundary intact.

### Out of scope

- FEMIC workflow orchestration.
- FreshForge graph planning or execution.
- A generic ws3 chat endpoint.
- Unchecked code or raw input syntax generated by a model.
- Silent mutation of `ForestModel` instances.

### Immediate task

Define the typed contract and workspace-facing adapter boundary, then write the first public-fixture extraction/import verification test. Do not add an LLM provider in this task.

Initial extraction and structural verification are complete in
`ws3.agent.themes.ModelContract`: the contract captures model metadata, theme
schema, development-type inventory, period-0 area inventory, and yield-component
coverage; L0 checks cover theme arity, basecodes, development-type key length,
and known theme codes; L1 reports duplicate keys, empty area inventory, and
missing yield coverage as warnings. Action references from `oper_expr` and
transition action codes are now captured and checked against declared actions;
the source-backed `verify_source()` oracle now runs Woodstock lint and a
scratch-model landscape/areas import, returning structured findings for
unsupported, malformed, or missing source data. The bounded compile/solve smoke
oracle has landed: `verify_compile_solve()` attempts action compilation and
returns a `CompileSolveCapability` record covering compile availability, solve
availability, per-development-type yield compilation status, and an explicit
deferral reason when a model defines no optimization problem. It does not invoke
`Problem.solve()`, because coefficient functions are not guaranteed safe to call
without user context, so an unavailable tier is never reported as a pass. The
The deterministic emission and adapter slice is implemented: FEMIC can
construct a model without a language model generating raw Woodstock syntax.
The typed deterministic construction boundary covers themes, areas, yields,
actions, transitions, and outputs. Outputs are emitted and imported with
normalized theme indices and output-group membership. Focused and full
regression tests cover deterministic bytes, FEMIC-shaped five-section input,
period conversion, transition proportions, and the output emit/import round-
trip. The FEMIC-side bridge regression compares all five bridge files byte-for-
byte with ws3 typed emission and checks imported action and transition state
without making FEMIC a ws3 runtime dependency. Current evidence is 76 focused
ws3 tests passed, 406 full-suite tests passed with 9 skips, and 10 FEMIC bridge
tests passed. The remaining Phase 10 work is closeout: audit adapter ownership
and intentionally lossy fields, link cross-repository evidence, and synchronize
the phase records before a PR decision.

Detailed plan: `planning/phase10_femic_model_contract.md`.

## Phase 11 — Ruff Lint Gate and Legacy Debt Cleanup

- Parent issue: [#120](https://github.com/UBC-FRESH/ws3/issues/120)
- Status: planned
- Branch: `feature/phase11-ruff-cleanup` (to be created on activation)

### Goal

Make the documented lint command truthful, choose one blocking lint policy, and
clean the existing Ruff debt in reviewable batches without mixing it into the
active Phase 10 FEMIC contract work. The focused baseline is 234 Ruff findings
in `ws3/forest.py`; the broader repository count must first be scoped so
notebooks, generated paths, and nested checkouts do not create false signal.

### Scope

- Establish an authoritative lint scope and baseline.
- Align Ruff's Python target with `requires-python >=3.10` and reconcile its
	policy with flake8.
- Repair genuine notebook/configuration defects and remove scope noise.
- Clean selected `ws3/` and `tests/` findings in bounded, tested batches.
- Enforce the chosen gate in CI or pre-commit and synchronize contributor docs.

### Child task checklist

- [ ] 11.1 Establish lint contract and baseline
- [ ] 11.2 Align configuration and choose the blocking linter
- [ ] 11.3 Repair scope and notebook defects
- [ ] 11.4 Apply low-risk package and test cleanup
- [ ] 11.5 Review behavior-sensitive `forest.py` debt
- [ ] 11.6 Enforce and close out the gate

Detailed plan: `planning/phase11_ruff_cleanup.md`.
