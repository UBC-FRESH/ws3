# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Common Changelog](https://common-changelog.org/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- Phase 11c (mypy Stage 4) complete: non-forest.py mypy errors reduced from 284 to 24.
  Fixed `opt.py` (status() return type, missing return, arg-type fixes, type: ignore on
  optional imports), `common.py` (harv_cost annotation, union-attr fix), `perf.py`
  (return type -> Any), `integration.py` (FastAPI decorators, integrator calls),
  `agent/capabilities/build_mask.py` and `diagnose_import.py` (Capability[Any] subclass
  type: ignore). 3 PaCal operator errors and 21 ForestModel/import_* no-untyped-call
  errors remain — blocked on Stage 5 (forest.py) and PaCal stubs. PR #140 merged.

- Phase 11 complete: 676 → 0 gate errors. Replaced flake8 CI gate with `ruff check ws3/ tests/`. Cleaned Ruff debt in `ws3/forest.py` (E741, E701, UP031, B904, E402 fixes), `ws3/spatial.py`, `ws3/common.py`, `ws3/core.py`, `ws3/forest_helper.py`, and `tests/`. Full suite passes: 406 tests, 9 skipped. Package v1.1.0a4. Branch `feature/phase11-ruff-cleanup`.

### Planned

- Phase 10 closeout is tracked in
  `planning/phase10_femic_model_contract.md`. The implementation, adapter
  ownership audit, cross-repository evidence, and final verification are
  complete. PR #127 merged into `main` as
  `ce95c16953fc8a283e4c8f376af19a19b761edd4`.
- Planned Phase 11, Ruff lint gate and legacy debt cleanup, under parent issue
  #120. The phase will first establish the authoritative lint scope and resolve
  the `py39` Ruff target versus `requires-python >=3.10` mismatch, then address
  the existing `forest.py` backlog in tested batches without mixing it into the
  active Phase 10 FEMIC contract work. Detailed plan:
  `planning/phase11_ruff_cleanup.md`.

### Added

- Started Phase 10 companion work for FEMIC #305 in issue #121 and branch
  `feature/p10-femic-model-contract`.
- Defined the ws3 responsibility as a typed model contract and deterministic
  verification-oracle surface behind FEMIC, while FreshForge owns workflow
  orchestration.
- Added `ws3.agent.themes.ModelContract`, a reference-free JSON-serializable
  extraction surface for model metadata, theme schema, and development-type
  inventory.
- Added structured L0 verification findings for theme arity, theme basecodes,
  development-type key length, and known theme codes, plus an L1 duplicate-key
  warning. Focused and regression validation passes with 33 tests passed and 1
  pre-existing skip.
- Extended `ModelContract` with typed development-type entries containing
  period-0 area inventory and yield-component coverage, plus L1 warnings for
  empty area inventory and missing yield coverage. The full ws3 suite passes:
  306 tests passed and 9 skipped.
- Added deterministic action and transition inventory to each development-type
  entry, with L1 warnings for references not present in the model's declared
  action set. The full ws3 suite passes again: 315 tests passed and 9 skipped.
- Added the source-backed `ModelContract.verify_source()` oracle. It runs
  `ws3.woodstock.lint_dataset`, attempts a real scratch-model landscape/areas
  import when lint permits, records source provenance, and returns structured
  findings instead of raising for missing, malformed, or unsupported input.
  The full ws3 suite passes: 323 tests passed and 9 skipped.

- `ws3.woodstock`: machine-readable contract for the Woodstock input data format (198 keywords)
  and `lint_dataset`, which reports sections and keywords ws3 does not read. Previously these
  were ignored silently, so a dataset could import cleanly and produce a model that was not the
  model that was written.
- New reference page, *Woodstock Format: What ws3 Reads*, documenting the supported subset and
  the two deliberate divergences from Woodstock (periods versus years, one-based versus
  zero-based theme indexing). Its support tables are generated from the contract at docs build
  time rather than transcribed, so they cannot drift from the importers.

### Fixed

- `ForestModel.import_landscape_section` now preserves theme descriptions from the `*THEME`
  declaration instead of discarding them. These are the only statement in a dataset of what a
  theme position means.
- `ws3.agent` capabilities now assemble development-type masks at the model's real theme count
  rather than asking a language model to reproduce it, and `ws3.agent.run` passes the model
  through to capability construction.

### Added
- **Agent capabilities** (`ws3.agent`, optional). A small set of operations designed to be driven by an AI coding agent, where the output is validated against real model state before it is returned.

  | Capability | Oracle |
  |---|---|
  | `build_mask` | the mask resolves against the `ForestModel` to at least one development type |
  | `explain_exception` | every ws3 symbol the explanation cites actually exists |
  | `diagnose_import` | the suggested fix is applied to a scratch copy and the section genuinely re-imports |

  A capability returns validated output or nothing at all — never a plausible guess. On exhaustion it returns `ok=False` with the reasons every attempt was rejected. Capabilities are advisory: they return proposals and never mutate a model in place.

  The design rule is *no oracle, no capability*. If a proposal cannot be checked against real state, it does not become a capability. This is recorded in `AGENTS.md` with its evidence: fabricated APIs reached this repository's documentation (Phase 6), test suite (Phase 7.5), and shipped module code (Phase 7.6) before being caught. `explain_exception` exists because that failure mode is well attested here.

- `ws3[agent]` and `ws3[agent-mcp]` extras, plus a `ws3-agent-mcp` console entry point exposing the capabilities as MCP tools. Tools in a tool list get called; conventions in documentation get ignored.
- Provenance: every attempt is recorded, including failures — model, prompt digest, raw output, verdict, attempt number. No field is capable of holding a credential; the endpoint host is recorded rather than the URL, which can carry credentials, and a SHA-256 rather than the prompt body, which can embed user data.
- New guide: `docs/source/guides/agent-capabilities.rst`, covering configuration, provenance, and how to add a capability validator-first.
- Companion package [fresh-agent-core](https://github.com/UBC-FRESH/fresh-agent-core), which owns the shared mechanism. Each package owns its own capabilities and validators, since the validator is the part requiring domain knowledge.

### Fixed
- Eight RST errors in `docs/source/guides/troubleshooting.rst`: directives and nested lists lacking a required preceding blank line. The docs now build with zero errors.

### Notes
`import ws3` never loads the agent package, and `pip install ws3` needs none of it. The MCP dependency is pinned `<2`; the 2.x SDK removed the low-level `Server` decorator API this host is written against, and migration is tracked in [fresh-agent-core#1](https://github.com/UBC-FRESH/fresh-agent-core/issues/1).

## 1.1.0a4 - 2026-07-29 (alpha)

**Status**: Defect sweep. Fixes a silent numerical error, restores probabilistic financial analysis, and gates code paths that returned meaningless results.

### Fixed
- **`sylv_cred` returned values wrong by ~40x, silently.** Seven of eight hand-copied binding pairs in `ws3/financial.py` bound `log` to `math.exp` rather than `math.log`, so `exp(C7d*log(vp)+C8d)` evaluated as `exp(C7d*exp(vp)+C8d)`. Only `harv_cost` was correct, which established the intent. Evaluated independently at `P=10, vr=2, vp=1`: correct `80.83076`, as-coded `126.33232`. The existing test asserted `126.33` — the buggy output — because it had been written by running the code and recording the result. The eight duplicated pairs are now a single `_math_funcs()` helper.
- **All `rv=True` paths raised `NameError`.** `PACAL_BROKEN = True` guarded the `import pacal` permanently, so the name was never bound. Neither flake8 nor mypy could see it: pyflakes treats a guarded import as binding the name.
- **Unguarded optional access in the Woodstock parsers.** 20 sites, mostly `re.search(...).group(...)`, which raise a bare `AttributeError` on malformed input. Now raise `ValueError` naming the construct, the expected pattern, and the offending text.

### Changed
- **PaCal support restored.** The standing note said to patch `numpy.fft.fftpack` in `pacal/utils.py`, but that describes PaCal 1.6 — 1.6.1 already imports from `numpy.fft`. The real blockers were NumPy 2.0 alias removals (`Inf`, `NaN`, `asfarray`, `product`) and an undeclared `sympy` dependency. A small additive compatibility shim makes PaCal 1.6.1 work on NumPy 2.5. Available via `pip install ws3[rv]`. PaCal is GPL-3.0-or-later while ws3 is MIT, so it remains an optional dependency the user installs and is never bundled.
- **Non-functional Phase 5 code paths are gated.** Twelve entry points across `advanced_modeling`, `perf`, and `integration` now raise `NotImplementedError` naming what is missing, rather than returning fabricated results. Most seriously, `StochasticOptimizer.solve_stochastic` reported `expected_value`, `variance`, and `std_dev` while `_apply_scenario` was a bare `pass` — scenarios were generated and never applied, so the variance across N identical solves was always exactly 0. Data structures, scenario generation, `MemoryProfiler`, `ResultCache`, and benchmarking are unaffected and still work.
- mypy's `warn_unused_ignores` disabled: with `ignore_missing_imports`, an absent optional dependency types as `Any`, so the check reports differently depending on whether PaCal is installed.

### Added
- `ws3[rv]` extra for probabilistic financial analysis.
- `ws3.financial.pacal_available()`.
- `tests/test_experimental_gates.py` — 29 tests asserting every gate fires with an actionable message and that the working parts still work.
- Regression tests for the `log`/`exp` mixup and the parser error messages.

### Notes
Several existing tests asserted the defective behaviour and had to be rewritten. They built `MagicMock` problems whose `get_objective_value()` and `get_solution()` returned canned values — neither method exists on `ws3.opt.Problem`, but `MagicMock` supplies any attribute requested, so they passed against an API that was never written. One asserted `result["objective_values"] == {}`, pinning a stub's hardcoded empty dict as correct. Tests written by recording what the code returned cannot catch the defect they were born from.

## 1.1.0a3 - 2026-07-29 (alpha)

**Status**: Patch release. Clears code-quality debt that was blocking meaningful CI signal, and fixes runtime defects shipped in 1.1.0a2.

### Fixed
- **`_cp` period conditions were broken for `>=` and `<=`.** Two separate defects in `DevelopmentType._compile_oper_expr` meant only `_cp =` ever worked:
  - the `<=` branch referenced an undefined `rel_opertors` (missing `a`), raising `NameError`
  - the bound was then folded in with an unguarded `max(_plo, plo), min(_phi, phi)`; a one-sided comparison leaves the opposite bound `None`, raising `TypeError`. The parallel `_age` branch already guarded for this.
- `resolve_tmask` called `resolve_treplace` and `resolve_tappend` without `self.`, so any model using `_REPLACE` or `_APPEND` theme expressions crashed.
- `_evaluate_basic` referenced a bare `parent` instead of `self.parent`.
- `resolve_replace`'s exception handler printed seven out-of-scope names, raising `NameError` inside the handler and masking the original error.
- A debug print in the target-age resolver referenced `age` where the variable is `sage`.
- Removed `_expand_action`, unreachable dead code that could not execute under any input.

### Changed
- **Minimum Python is now 3.10.** `ws3/opt.py` has used `match` statements (structural pattern matching, 3.10+) for some time while `requires-python` still claimed `>=3.9`, so installing on 3.9 produced an immediate `SyntaxError` on import. The metadata now states what the code actually requires. Python 3.9 reached end-of-life in October 2025.
- **CI no longer collapses on a lint failure.** `test`, `docs`, and `build` were transitively gated on `lint`; with the lint job permanently red, none of them ever ran. They are now independent. mypy is advisory pending the Phase 2 typing debt.
- Added a `dev` extra to `pyproject.toml`. CI ran `pip install -e ".[dev]"` against an extra that was never declared, so pytest was never installed — masked because the test job never ran.
- Added `.flake8` codifying an explicit policy: correctness enforced (all pyflakes codes, `E7xx` logic errors, syntax, deprecations), style codes that conflict with long-standing project conventions documented and ignored.
- flake8 across `ws3/` and `tests/` now exits clean, down from 1808 findings.

### Added
- Parametrized regression tests covering all three `_cp` relational operators and the invalid-operator path.

### Notes
Three tests were passing without testing anything, and are now explicitly skipped with the reason stated rather than silently vacuous:
- `test_operate` was decorated `@pytest.fixture`, so pytest collected it as a fixture and never ran it; it also requests an `area_selector` fixture that does not exist.
- `test_yield_curve_interpolation` calls `ws3.core.interpolate_curves`, which does not exist; absent test data triggered an early skip before the undefined name was evaluated.
- `test_documentation` computed a section-header flag and never asserted it. The implied assertion is now present and passing.

## 1.1.0a2 - 2026-07-29 (alpha)

**Status**: Second alpha release. Documentation cleanup complete (Phase 6). Ready for user testing.

### Changed
- **Phase 6 complete**: All documentation fabricated APIs purged, legacy stubs removed, troubleshooting guide fixed
- Documentation now uses real ws3 APIs throughout (verified against source)
- `sphinx-build -b html docs/source _build/html -W` passes with zero errors

### Removed
- Deprecated legacy chapters (`Chapt1.rst`, `Chapt2.rst`, `intro.rst`, `aboutws3.rst`)
- 2-line stubs (`SpaDES.rst`, `libCBM.rst`) — content covered by textbook

## 1.1.0a1 - 2026-07-26 (alpha)

**Status**: Alpha release for smoke testing. Not yet stable.

## Phase 7 — Release and Community Building (2026-07-27)

**Status**: Complete (2026-07-29)

### Added
- Issue templates (bug_report, feature_request, question) for GitHub
- Community channels: GitHub Discussions enabled, CONTRIBUTING.md updated with support info
- Phase 7 planning docs and GitHub issues (#87-#91)
- Feature branch `feature/ws3-phase7-release`

### Added
- Phase 5 interactive notebooks (070-078) for advanced workflows
- FAQ section in documentation (`docs/source/howto/faq.rst`)
- Migration guide from Woodstock to ws3 (`docs/source/howto/migration_from_woodstock.rst`)
- Multi-objective optimization examples
- Spatial constraints and adjacency modeling
- Parallel optimization and performance benchmarking
- Scenario analysis and comparison workflows
- New modules: `ws3.advanced_modeling`, `ws3.perf`, `ws3.integration`
- Textbook chapters 17-18 (advanced spatial, carbon accounting)
- How-to guides: advanced-optimization, custom-solvers, data-validation, scenario-analysis
- CI/CD pipeline with lint, test, docs, and publish jobs

### Changed
- Updated ROADMAP.md to reflect Phase 5 progress
- Enhanced documentation with 18 how-to guides (was 14)
- Exported new modules in `ws3.__all__`

### Fixed
- Removed broken duplicate notebook 072 (carbon accounting)
- Resolved DT key/mask matching issues in yield curve loading

### Known Issues
- Textbook chapters 19-20 not yet created (planned but not implemented)
- No dedicated test files for new modules (only import checks)
- CI/CD publish job not configured with PyPI secrets
- Version was previously misreported as 2.0.0 in some documents (now corrected to 1.1.0a1)

## 1.0.0 - 2024-11-24

### Added
- Docstring coverage
- Multiple-format automatic technical documentation system implemented via Sphinx and readthedocs
- Unit tests
- Examples of different purposes
- Add an integrated PuLP optimization module
- A feature to get left handside values of constraints in the optimization module

### Fix
- Fix ws3/cbm connection
- Fix error on populating themes

## 0.0.1 - 2021-10-19
- Initial release
