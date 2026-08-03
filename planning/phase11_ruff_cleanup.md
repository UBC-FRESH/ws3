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

- Apply reviewed mechanical fixes for imports, modern typing aliases, unused
  loop variables, safe comprehensions, and equivalent formatting.
- Run focused tests after each batch and preserve behavior in public APIs.

### 11.5 Review behavior-sensitive `forest.py` debt

- Split one-line control flow and rename ambiguous variables in bounded parser
  sections, with tests for import, simulation, and output behavior.
- Review `%` formatting, mutable defaults, lambda assignments, and
  `assert False` individually rather than applying unsafe bulk fixes.
- Keep P10 output parsing and typed emission regressions in the validation set.

### 11.6 Enforce and close out the gate

- Make the selected lint command blocking in CI or pre-commit.
- Update the roadmap, changelog, contributor instructions, and parent issue
  checklist with measured final results.
- Verify the full test, build, and documentation gates.

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