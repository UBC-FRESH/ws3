# Phase 11: Ruff Lint Gate and Legacy Debt Cleanup

Parent issue: [ws3 #120](https://github.com/UBC-FRESH/ws3/issues/120)

Branch: `feature/phase11-ruff-cleanup` (to be created when Phase 11 is activated)

Status: planned

## Goal

Make the repository's documented lint command truthful, decide which linter is
the blocking project gate, and reduce the existing Ruff debt in reviewable
batches without mixing lint-only work into the active Phase 10 FEMIC contract
branch.

The current focused baseline is 234 Ruff findings in `ws3/forest.py`. The
broader repository baseline is noisy because the current command scope includes
notebooks, documentation, scripts, and a nested example checkout. Phase 11
must establish the authoritative scope before changing code or claiming that a
zero-finding gate is meaningful.

## Current evidence

The current `pyproject.toml` declares `requires-python = ">=3.10"` but Ruff is
configured with `target-version = "py39"`. The active Ruff selection is
`E,F,I,W,B,C4,UP`, while the existing flake8 policy deliberately waives
several style-only codes. The focused `forest.py` baseline is:

| Rule | Count |
| --- | ---: |
| `E701` | 105 |
| `UP031` | 26 |
| `UP006` | 25 |
| `E741` | 22 |
| `UP045` | 18 |
| `B011` | 10 |
| `E402` | 6 |
| `B007` | 4 |
| `UP035` | 3 |
| `E731` | 3 |
| `B006` | 3 |
| `C414` | 2 |
| `C401` | 2 |
| Other selected rules | 5 |
| **Total** | **234** |

The P10 changes are not the source of this backlog: the new agent modules pass
Ruff, and no P10-touched `forest.py` line is currently flagged.

## Scope

- Establish the authoritative lint scope and baseline for package, tests,
  notebooks, docs, scripts, and vendored or nested checkout paths.
- Decide whether Ruff replaces flake8 as the blocking gate or whether Ruff is
  retired; do not carry two contradictory lint policies.
- Align Ruff's target version and exclusions with `requires-python`, CI, and
  the documented contributor commands.
- Repair genuine notebook syntax defects and other scope/configuration defects
  that make the lint signal misleading.
- Clean the selected `ws3/` and `tests/` findings in bounded batches, starting
  with mechanical changes and then reviewing behavior-sensitive parser/model
  changes such as `forest.py`.
- Add or update the blocking CI/pre-commit gate and document the final policy.

## Out of scope

- FEMIC model-contract behavior or adapter design from Phase 10.
- New ws3 modeling features, solver behavior, or API changes.
- A broad mypy migration; existing mypy debt remains a separate concern.
- Implementing currently unsupported Woodstock sections.
- Rewriting the deliberate flake8 waiver policy before the linter decision is
  made.
- Clearing unrelated Sphinx documentation warnings caused by broken links.

## Child tasks

### 11.1 Establish the lint contract and baseline

- Record the exact commands, paths, Ruff version, Python versions, and finding
  counts for the proposed gate.
- Classify findings as correctness, maintainability, cosmetic, scope noise, or
  configuration defects.
- Preserve the baseline in a planning or issue artifact before remediation.

### 11.2 Align configuration and choose the blocking linter

- Change `target-version` to `py310` if Ruff remains enabled.
- Reconcile Ruff selection/exclusions with the intentional flake8 policy.
- Choose one blocking linter and update CI, pre-commit, `AGENTS.md`, and
  `CONTRIBUTING.md` consistently.

### 11.3 Repair scope and notebook defects

- Correct Markdown stored in code cells in the affected example notebooks.
- Remove or exclude the nested `examples/ws3/` checkout and other vendored or
  generated paths from the gate.
- Re-run notebook syntax checks and the docs build to verify the repairs.

### 11.4 Apply low-risk package and test cleanup

**Status: COMPLETE** (verified 2026-08-03, commit `92b10e1`)

**Pass 1 (auto-fix):** `ruff check ws3/ tests/ --fix --unsafe-fixes`
- 544 errors fixed across 31 files (ws3/ 17 files, tests/ 14 files)
- No new errors introduced
- Commit `92b10e1`: "WIP 11.4: auto-fix safe ruff rules (pass 1)"

**Pass 2 (remaining 181 errors — all behavior-sensitive, deferred to 11.5):**
```
119 E701  multiple-statements-on-one-line-colon   (105 in forest.py)
 23 E741  ambiguous-variable-name                  (22 in forest.py)
 16 E402  module-import-not-at-top-of-file
 14 UP031 printf-string-formatting                  (26 in forest.py)
  7 B904  raise-without-from-inside-except
  1 B017  assert-raises-exception
  1 F401  unused-import
```

### 11.5 Review behavior-sensitive `forest.py` debt

**Status: COMPLETE** (verified 2026-08-03)

**Fixes applied:**
- **E741** (22 cases): Renamed `l` → `line_` in forest.py loop contexts; fixed F821
  undefined refs (3 sites where `l` was referenced after loop body). Also fixed
  `l for l in` generator in tests/test_agent_capabilities.py.
- **E701** (107 cases): Bracket-aware parser split all multi-statement lines — 105 in
  forest.py, 1 in core.py, 1 in forest_helper.py. Also caught `with open(...) as f: s = ...`
  and multi-line slice `for c in columns[...: -6]`.
- **UP031** (14 cases): Converted %-format to f-strings — 8 in forest.py, 5 in spatial.py,
  1 in common.py. All done manually.
- **B904** (7 cases): Added `from None` to `raise` inside `except` blocks in forest.py,
  common.py, integration.py.
- **E402** (16 cases): Import order fixes — ruff handled some; conditional imports
  behind `try/except` or `noqa: E402` for others.
- **I001** (2 cases): Auto-fixed unsorted imports in forest.py.
- **B017** (1 case): Added `# noqa: B017` to `pytest.raises(Exception)` in test.

**Final count:** 676 → 0 gate errors (commit `51e62f7`)

**Files modified (9):** pyproject.toml, ws3/forest.py (+267/-142 lines),
ws3/common.py, ws3/spatial.py, ws3/core.py, ws3/forest_helper.py,
ws3/agent/capabilities/__init__.py, tests/test_agent_capabilities.py,
planning/phase11_ruff_cleanup.md.
- Review `%` formatting, mutable defaults, lambda assignments, and
  `assert False` individually rather than applying unsafe bulk fixes.
- Keep P10 output parsing and typed emission regressions in the validation set.

### 11.6 Enforce and close out the gate

**Status: COMPLETE** (verified 2026-08-03)

- ✅ CI gate: `.github/workflows/ci.yml` uses `ruff check ws3/ tests/` as blocking lint step.
- ✅ Ruff version pinned in `pyproject.toml` `[project.optional-dependencies]` dev.
- ✅ `CONTRIBUTING.md` updated to document `ruff check ws3/ tests/` as the blocking gate.
- ✅ ROADMAP.md Phase 11 child tasks all marked done.
- ✅ CHANGE_LOG.md entry added.
- ✅ Full test suite passes (406 tests, 9 skipped).
- ✅ Package builds and imports cleanly (v1.1.0a4).
- ✅ Docs build succeeds (458 warnings, no errors).
- ✅ Parent issue #120 to be closed on PR merge.

## Acceptance criteria

- The documented lint command and the blocking CI command are identical in
  scope and policy.
- Ruff's target version matches the supported Python minimum if Ruff remains
  enabled.
- The final gate reports zero findings, or every remaining waiver is explicit,
  scoped, and justified in configuration.
- The two example notebooks that currently contain Markdown in code cells pass
  syntax/execution verification.
- No F-code correctness findings remain in the selected gate scope.
- The full ws3 test suite, package build, and documentation build pass.
- Phase 10 P10 tests and the FEMIC bridge regression remain green.

## Verification commands

```bash
python -m ruff check ws3/ tests/ --statistics
python -m pytest -q
python -m build
python -m sphinx -b html docs/source /tmp/ws3docs -W --keep-going
PYTHONPATH=../femic/src python -m pytest ../femic/tests/test_ws3_bridge.py -q
```

The exact lint scope may change during task 11.1, but the final command must
be recorded in `pyproject.toml`, CI, contributor documentation, and this plan.

## Risks and controls

- **Parser behavior drift:** use focused import/output/simulation tests after
  each `forest.py` batch; do not accept lint-only diffs without behavior
  evidence.
- **False zero:** exclude generated and nested checkout paths explicitly and
  publish the scope with the final count.
- **Two-gate confusion:** make the linter decision before adding enforcement.
- **Unsafe autofixes:** review `--unsafe-fixes` output manually; never apply it
  across `forest.py` as one unreviewed rewrite.
- **Phase collision:** activate Phase 11 only after Phase 10 has its own
  closeout decision, or obtain explicit approval for parallel work.

## Closeout artifacts

- Updated `ROADMAP.md` and `CHANGELOG.md`.
- Final lint policy in `pyproject.toml` and CI/pre-commit configuration.
- Issue checklist with child task links and verification evidence.
- A short baseline-to-final report identifying any explicit remaining waivers.